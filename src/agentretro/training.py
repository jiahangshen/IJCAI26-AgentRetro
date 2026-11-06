from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from .backbones.base import BackboneAdapter
from .models.domain import DomainClassifier, GradientReversal


def release_fragmented_cuda_cache(
    device: torch.device,
    *,
    free_threshold_bytes: int = 4 * 1024**3,
    cached_threshold_bytes: int = 64 * 1024**2,
) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    free_bytes, _ = torch.cuda.mem_get_info(device)
    cached_bytes = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    if free_bytes >= free_threshold_bytes or cached_bytes < cached_threshold_bytes:
        return False
    torch.cuda.empty_cache()
    return True


def selective_save_on_cpu(
    minimum_bytes: int = 0,
    *,
    pin_memory: bool = True,
    device_type: str = "cuda",
):
    minimum_bytes = int(minimum_bytes)
    if minimum_bytes < 0:
        raise ValueError("minimum_bytes must be non-negative")
    device_module = getattr(torch, device_type, torch.cuda)

    def pack(tensor):
        size_bytes = tensor.numel() * tensor.element_size()
        should_offload = tensor.device.type == device_type and size_bytes >= minimum_bytes
        if not should_offload:
            return False, tensor.device, tensor
        if not pin_memory:
            return True, tensor.device, tensor.cpu()
        packed = torch.empty(
            tensor.size(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            pin_memory=(device_module.is_available() and not tensor.is_sparse),
        )
        packed.copy_(tensor)
        return True, tensor.device, packed

    def unpack(packed):
        was_offloaded, device, tensor = packed
        if not was_offloaded:
            return tensor
        return tensor.to(device, non_blocking=pin_memory)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


@dataclass
class TrainSettings:
    lambda_adv: float = 0.1
    lambda_meta: float = 0.5
    inner_lr: float = 1e-4
    meta_train_fraction: float = 0.8
    second_order: bool = True
    offload_second_order: bool = False
    offload_minimum_bytes: int = 0
    meta_split_seed: int = 2026

    gradient_clip: float = 25.0

def select_batch(batch: Any, indices: torch.Tensor) -> Any:
    if torch.is_tensor(batch):
        if batch.ndim > 0 and batch.shape[0] >= int(indices.max()) + 1:
            return batch.index_select(0, indices.to(batch.device))
        return batch
    if isinstance(batch, dict):
        return {key: select_batch(value, indices) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(select_batch(value, indices) for value in batch)
    if isinstance(batch, list):
        return [batch[int(i)] for i in indices]
    return batch


def split_meta_domains(
    domains: torch.Tensor,
    fraction: float = 0.8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_domains, test_domains = split_meta_domain_values(domains, fraction, generator)
    train_mask = torch.isin(domains, train_domains.to(domains.device))
    test_mask = torch.isin(domains, test_domains.to(domains.device))
    return train_mask.nonzero(as_tuple=False).flatten(), test_mask.nonzero(as_tuple=False).flatten()


def split_meta_domain_values(
    domains: torch.Tensor,
    fraction: float = 0.8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    unique = torch.unique(domains).cpu()
    if len(unique) < 2:
        raise ValueError("MLDG requires at least two domains in a batch")
    order = unique[torch.randperm(len(unique), generator=generator)]
    cut = min(len(order) - 1, max(1, int(round(len(order) * fraction))))
    return order[:cut], order[cut:]


class BilevelTrainer:
    def __init__(
        self,
        adapter: BackboneAdapter,
        model_optimizer: torch.optim.Optimizer,
        settings: TrainSettings,
        domain_optimizer: torch.optim.Optimizer | None = None,
        domain_lr: float = 1e-4,
    ):
        self.adapter = adapter
        self.settings = settings
        if hasattr(adapter, "set_text_enabled"):
            adapter.set_text_enabled(True)
        self.domain_classifier = DomainClassifier(adapter.feature_dim)
        self.domain_classifier.to(next(adapter.parameters()).device)
        self.grl = GradientReversal()
        self.model_optimizer = model_optimizer

        self.domain_optimizer = domain_optimizer or torch.optim.AdamW(
            self.domain_classifier.parameters(),
            lr=domain_lr,
            betas=(0.9, 0.98),
            weight_decay=0.01,
        )
        self.generator = torch.Generator().manual_seed(settings.meta_split_seed)
        self.step_number = 0

    def _domain_loss(self, batch: Mapping[str, Any], detach: bool) -> torch.Tensor:
        if detach:
            with torch.no_grad():
                features = self.adapter(batch, mode="encode")
        else:
            features = self.adapter(batch, mode="encode")
            features = self.grl(features)
        return nn.functional.cross_entropy(self.domain_classifier(features), batch["domain_id"].long())

    def _task_components(
        self, batch: Mapping[str, Any]
    ) -> list[tuple[Mapping[str, Any], float]]:
        return [(batch, 1.0)]

    def _fast_parameters(self, inner_loss: torch.Tensor) -> dict[str, torch.Tensor]:
        trainable = [
            (name, value) for name, value in self.adapter.meta_named_parameters() if value.requires_grad
        ]
        gradients = torch.autograd.grad(
            inner_loss,
            [value for _, value in trainable],
            create_graph=self.settings.second_order,
            allow_unused=True,
        )
        updates: dict[int, torch.Tensor] = {}
        for (name, value), gradient in zip(trainable, gradients):
            if gradient is not None:
                updates[id(value)] = value - self.settings.inner_lr * gradient

        return {
            name: updates.get(id(value), value)
            for name, value in self.adapter.named_parameters(remove_duplicate=False)
        }

    def _restore_parameter_aliases(self, parameters: list[tuple[str, nn.Parameter]]) -> None:
        for qualified_name, parameter in parameters:
            module_name, parameter_name = qualified_name.rsplit(".", 1)
            module = self.adapter.get_submodule(module_name)
            module._parameters[parameter_name] = parameter

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        return self.train_accumulated_step([batch])

    def train_accumulated_step(
        self, batches: Sequence[Mapping[str, Any]]
    ) -> dict[str, float]:
        batches = list(batches)
        if not batches:
            raise ValueError("Gradient accumulation requires at least one microbatch")

        uses_cuda = next(self.adapter.parameters()).is_cuda
        if self.settings.second_order and uses_cuda:
            with sdpa_kernel(SDPBackend.MATH):
                return self._train_accumulated_step_impl(batches)
        return self._train_accumulated_step_impl(batches)

    def _train_accumulated_step_impl(
        self, batches: Sequence[Mapping[str, Any]]
    ) -> dict[str, float]:
        self.adapter.train()
        self.domain_classifier.train()
        uses_cuda = next(self.adapter.parameters()).is_cuda
        sample_counts = [int(batch["domain_id"].numel()) for batch in batches]
        total_samples = sum(sample_counts)
        if total_samples <= 0:
            raise ValueError("Cannot train on empty microbatches")
        prepared = []
        all_domains = torch.cat([batch["domain_id"].long() for batch in batches])
        train_domains, test_domains = split_meta_domain_values(
            all_domains, self.settings.meta_train_fraction, self.generator
        )
        for batch in batches:
            domains = batch["domain_id"].long()
            train_idx = torch.isin(domains, train_domains.to(domains.device)).nonzero(
                as_tuple=False
            ).flatten()
            test_idx = torch.isin(domains, test_domains.to(domains.device)).nonzero(
                as_tuple=False
            ).flatten()
            prepared.append(
                (
                    batch,
                    self.adapter.select_batch(batch, train_idx) if train_idx.numel() else None,
                    self.adapter.select_batch(batch, test_idx) if test_idx.numel() else None,
                )
            )
        if not any(meta_train is not None for _, meta_train, _ in prepared):
            raise ValueError("MLDG virtual batch has no meta-train reactions")
        if not any(meta_test is not None for _, _, meta_test in prepared):
            raise ValueError("MLDG virtual batch has no meta-test reactions")

        meta_train_total = sum(
            int(meta_train["domain_id"].numel())
            for _, meta_train, _ in prepared
            if meta_train is not None
        )
        meta_test_total = sum(
            int(meta_test["domain_id"].numel())
            for _, _, meta_test in prepared
            if meta_test is not None
        )

        metric_totals = {
            "loss": 0.0,
            "task_loss": 0.0,
            "domain_classifier_loss": 0.0,
            "adversarial_loss": 0.0,
            "outer_loss": 0.0,
        }
        self.domain_optimizer.zero_grad(set_to_none=True)
        classifier_batches = [meta_train for _, meta_train, _ in prepared if meta_train is not None]
        classifier_total = sum(int(batch["domain_id"].numel()) for batch in classifier_batches)
        for classifier_batch in classifier_batches:
            weight = int(classifier_batch["domain_id"].numel()) / classifier_total
            domain_value = self._domain_loss(classifier_batch, detach=True)
            (domain_value * weight).backward()
            metric_totals["domain_classifier_loss"] += weight * float(domain_value.detach())
        self.domain_optimizer.step()

        self.model_optimizer.zero_grad(set_to_none=True)

        task_components = [
            component
            for batch, _, _ in prepared
            for component in self._task_components(batch)
        ]
        task_total = sum(
            int(batch["domain_id"].numel()) * supervision_weight
            for batch, supervision_weight in task_components
        )
        if task_total <= 0:
            raise ValueError("Virtual batch contains no real task-supervision rows")
        for batch, supervision_weight in task_components:
            weight = (
                int(batch["domain_id"].numel()) * supervision_weight / task_total
            )
            task = self.adapter(batch, mode="loss")
            (task * weight).backward()
            metric_totals["task_loss"] += weight * float(task.detach())

        offload = (
            selective_save_on_cpu(
                self.settings.offload_minimum_bytes,
                pin_memory=True,
            )
            if uses_cuda and self.settings.offload_second_order
            else nullcontext()
        )
        with offload:
            inner_total = None
            adversarial_total = None
            for _, meta_train, _ in prepared:
                if meta_train is None:
                    continue
                weight = int(meta_train["domain_id"].numel()) / meta_train_total
                inner = self.adapter(meta_train, mode="loss")
                adversarial = self._domain_loss(meta_train, detach=False)
                inner = inner + self.settings.lambda_adv * adversarial
                weighted_adversarial = adversarial * weight
                adversarial_total = (
                    weighted_adversarial
                    if adversarial_total is None
                    else adversarial_total + weighted_adversarial
                )
                weighted_inner = inner * weight
                inner_total = (
                    weighted_inner if inner_total is None else inner_total + weighted_inner
                )
            if inner_total is None:
                raise RuntimeError("MLDG failed to construct its virtual inner objective")
            fast = self._fast_parameters(inner_total)
            if adversarial_total is not None:
                metric_totals["adversarial_loss"] = float(adversarial_total.detach())

            outer_batches = [meta_test for _, _, meta_test in prepared if meta_test is not None]
            for index, meta_test in enumerate(outer_batches):
                weight = int(meta_test["domain_id"].numel()) / meta_test_total
                original_parameters = list(
                    self.adapter.named_parameters(remove_duplicate=False)
                )
                try:
                    outer = functional_call(
                        self.adapter,
                        fast,
                        (meta_test,),
                        {"mode": "loss"},
                        tie_weights=False,
                    )
                finally:
                    self._restore_parameter_aliases(original_parameters)
                weighted_outer = outer * weight
                if uses_cuda:
                    release_fragmented_cuda_cache(next(self.adapter.parameters()).device)
                (self.settings.lambda_meta * weighted_outer).backward(
                    retain_graph=index < len(outer_batches) - 1
                )
                metric_totals["outer_loss"] += float(weighted_outer.detach())
        metric_totals["loss"] = (
            metric_totals["task_loss"]
            + self.settings.lambda_meta * metric_totals["outer_loss"]
        )

        nn.utils.clip_grad_norm_(self.adapter.parameters(), self.settings.gradient_clip)
        self.model_optimizer.step()
        self.step_number += 1
        return metric_totals
