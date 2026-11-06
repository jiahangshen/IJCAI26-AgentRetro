from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn

from ..models.domain import TextFusion, masked_mean


class BackboneAdapter(nn.Module, ABC):
    feature_dim: int

    @abstractmethod
    def encode(self, batch: Mapping[str, Any]) -> torch.Tensor:
        pass

    @abstractmethod
    def task_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, batch: Mapping[str, Any], top_k: int = 10) -> list[list[str]]:
        pass

    def forward(self, batch: Mapping[str, Any], mode: str = "loss") -> torch.Tensor:
        if mode == "encode":
            return self.encode(batch)
        if mode == "loss":
            return self.task_loss(batch)
        raise ValueError(f"Unknown adapter mode: {mode}")

    def meta_named_parameters(self):
        return list(self.named_parameters())

    def select_batch(self, batch: Mapping[str, Any], indices: torch.Tensor) -> Mapping[str, Any]:
        from ..training import select_batch

        return select_batch(batch, indices)

    def save_checkpoint(self, path: str | Path, **metadata: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "metadata": metadata}, path)

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu")
        self.load_state_dict(payload["state_dict"], strict=strict)
        return payload.get("metadata", {})


class HookedBackboneAdapter(BackboneAdapter):
    def __init__(
        self,
        model: nn.Module,
        representation_module: nn.Module,
        feature_dim: int,
        loss_fn: Callable[[nn.Module, Mapping[str, Any]], torch.Tensor],
        predict_fn: Callable[[nn.Module, Mapping[str, Any], int], list[list[str]]],
        text_dim: int = 768,
        output_getter: Callable[[Any], tuple[torch.Tensor, Callable[[torch.Tensor], Any]]] | None = None,
    ):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.text_fusion = TextFusion(text_dim, feature_dim)
        self.loss_fn = loss_fn
        self.predict_fn = predict_fn
        self.output_getter = output_getter or self._default_output_getter
        object.__setattr__(self, "_representation_module_ref", representation_module)
        self.text_enabled = True
        self._current_text: torch.Tensor | None = None
        self._last_features: torch.Tensor | None = None
        self._handle = representation_module.register_forward_hook(self._representation_hook)

    @staticmethod
    def _default_output_getter(output: Any):
        if torch.is_tensor(output):
            return output, lambda value: value
        if isinstance(output, dict) and "graph_feature" in output:
            return output["graph_feature"], lambda value: {**output, "graph_feature": value}
        raise TypeError("Provide output_getter for this upstream representation")

    def _representation_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        extracted = self.output_getter(output)
        hidden, rebuild = extracted[:2]
        padding_mask = extracted[2] if len(extracted) == 3 else None
        if hidden.ndim == 3 and hidden.shape[0] != (self._current_text.shape[0] if self._current_text is not None else hidden.shape[0]):
            hidden = hidden.transpose(0, 1)
            transpose_back = True
        else:
            transpose_back = False
        if not self.text_enabled:
            self._last_features = masked_mean(hidden, padding_mask)
            return output
        fused = self.text_fusion(hidden, self._current_text)
        self._last_features = masked_mean(fused, padding_mask)
        return rebuild(fused.transpose(0, 1) if transpose_back else fused)

    def _set_context(self, batch: Mapping[str, Any]) -> None:
        self._current_text = batch.get("text_embedding") if self.text_enabled else None
        self._last_features = None

    def set_text_enabled(self, enabled: bool) -> None:
        self.text_enabled = bool(enabled)

    def meta_named_parameters(self):
        parameters = list(self.named_parameters())
        if self.text_enabled:
            return parameters
        text_ids = {id(parameter) for parameter in self.text_fusion.parameters()}
        return [(name, parameter) for name, parameter in parameters if id(parameter) not in text_ids]

    def encode(self, batch: Mapping[str, Any]) -> torch.Tensor:
        self._set_context(batch)
        self._run_representation(batch)
        if self._last_features is None:
            raise RuntimeError("Representation hook did not run")
        return self._last_features

    def _run_representation(self, batch: Mapping[str, Any]) -> Any:
        if hasattr(self.model, "encode"):
            return self.model.encode(batch)
        raise NotImplementedError("Subclass must implement _run_representation")

    def task_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        self._set_context(batch)
        loss = self.loss_fn(self.model, batch)
        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")
        return loss

    def predict(self, batch: Mapping[str, Any], top_k: int = 10) -> list[list[str]]:
        self._current_text = None
        return self.predict_fn(self.model, batch, top_k)
