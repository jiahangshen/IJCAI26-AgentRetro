from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from .base import HookedBackboneAdapter


def _group_hypotheses(hypotheses, batch_size: int, top_k: int):
    if len(hypotheses) == batch_size * top_k:
        hypotheses = [
            [candidate for row in hypotheses[start : start + top_k] for candidate in row]
            for start in range(0, len(hypotheses), top_k)
        ]
    if len(hypotheses) != batch_size:
        raise RuntimeError(
            f"EditRetro returned {len(hypotheses)} rows for batch size {batch_size} and TopK {top_k}"
        )
    return hypotheses


def _deterministic_inference(task, generator, model, batch):
    source = batch["net_input"]["src_tokens"]
    devices = [source.device.index] if source.is_cuda else []
    batch_seed = int(batch.get("id", source.new_zeros(1)).detach().long().sum().cpu())
    seed = int(getattr(generator, "random_seed", 1)) + batch_seed

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if source.is_cuda:
            torch.cuda.manual_seed_all(seed)
        return task.inference_step(generator, [model], batch)


def _editretro_output_getter(output: Any):
    hidden = output.encoder_out.transpose(0, 1)

    def rebuild(value: torch.Tensor):
        value = value.transpose(0, 1)
        if hasattr(output, "_replace"):
            return output._replace(encoder_out=value)
        output.encoder_out = value
        return output

    return hidden, rebuild, getattr(output, "encoder_padding_mask", None)


class EditRetroAdapter(HookedBackboneAdapter):
    def __init__(self, model, task, criterion, feature_dim: int, generator=None):
        def loss_fn(native_model, batch):
            sample = dict(batch)
            if "prev_target" not in sample:
                sample["prev_target"] = task.inject_noise(sample["target"])
            loss, _, _ = criterion(native_model, sample)
            return loss

        def predict_fn(native_model, batch, top_k):
            if generator is None:
                raise RuntimeError("EditRetro generator was not constructed")
            hypotheses = _deterministic_inference(task, generator, native_model, batch)
            dictionary = task.target_dictionary
            batch_size = int(batch["net_input"]["src_tokens"].size(0))

            generated_top_k = int(getattr(generator, "TOPK", top_k))
            hypotheses = _group_hypotheses(hypotheses, batch_size, generated_top_k)
            return [
                [dictionary.string(hypothesis["tokens"]) for hypothesis in sample[:top_k]]
                for sample in hypotheses
            ]

        super().__init__(
            model=model,
            representation_module=model.encoder,
            feature_dim=feature_dim,
            loss_fn=loss_fn,
            predict_fn=predict_fn,
            output_getter=_editretro_output_getter,
        )
        self.task = task
        self.criterion = criterion
        self.generator = generator

    def meta_named_parameters(self):
        prefixes = ("model.encoder.", "text_fusion.")
        selected = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name.startswith(prefixes)
            and (self.text_enabled or not name.startswith("text_fusion."))
        ]
        if not selected:
            raise RuntimeError("EditRetro MLDG found no feature-extractor parameters")
        return selected

    def predict_scored(self, batch: Mapping[str, Any], top_k: int = 10) -> list[list[tuple[str, float]]]:
        if self.generator is None:
            raise RuntimeError("EditRetro generator was not constructed")
        self._current_text = None
        hypotheses = _deterministic_inference(self.task, self.generator, self.model, batch)
        batch_size = int(batch["net_input"]["src_tokens"].size(0))
        generated_top_k = int(getattr(self.generator, "TOPK", top_k))
        hypotheses = _group_hypotheses(hypotheses, batch_size, generated_top_k)
        dictionary = self.task.target_dictionary
        output = []
        for sample in hypotheses:
            rows = []
            for hypothesis in sample[:top_k]:
                score = hypothesis.get("score")
                rows.append(
                    (
                        dictionary.string(hypothesis["tokens"]),
                        float(score) if score is not None else float("-inf"),
                    )
                )
            output.append(rows)
        return output

    def _run_representation(self, batch: Mapping[str, Any]):
        net = batch["net_input"]
        return self.model.encoder(net["src_tokens"], src_lengths=net.get("src_lengths"))


def load_editretro_checkpoint(checkpoint: str | Path, data_dir: str | Path | None = None) -> EditRetroAdapter:
    root = Path(__file__).resolve().parents[3]
    upstream = root / "third_party" / "editretro"
    for path in (upstream, upstream / "fairseq"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import editretro
    from fairseq import checkpoint_utils, tasks

    overrides = {"data": str(data_dir)} if data_dir else None
    state = checkpoint_utils.load_checkpoint_to_cpu(str(checkpoint), overrides)
    args = state["args"]
    if getattr(args, "arch", "") == "pretrain_mlm_editretro":
        args.arch = "editretro_nat"
        args.task = "translation_retro"
        args.criterion = "nat_loss"
        args.noise = "random_delete_shuffle"
        args.data = str(data_dir)
        args.alpha_ratio = 0.5
        args.dae_ratio = 0.5

        args.dropout = 0.3
        args.attention_dropout = 0.0
        args.label_smoothing = 0.0
        args.apply_bert_init = True
        task = tasks.setup_task(args)
        model = task.build_model(args)
        pretrained = dict(state["model"])
        for key in (
            "encoder.lm_head_transform_weight.weight",
            "encoder.lm_head_transform_weight.bias",
            "encoder.layer_norm1.weight",
            "encoder.layer_norm1.bias",
        ):
            pretrained.pop(key, None)
        current = model.state_dict()
        compatible = {}
        for key, value in pretrained.items():
            if key in current:
                if value.ndim == current[key].ndim and value.shape[1:] == current[key].shape[1:]:
                    if value.shape[0] == current[key].shape[0] + 1:
                        value = value[:-1]
                if value.shape == current[key].shape:
                    compatible[key] = value
        model.load_state_dict(compatible, strict=False)
    else:
        models, args, task = checkpoint_utils.load_model_ensemble_and_task(
            [str(checkpoint)], arg_overrides=overrides
        )
        model = models[0]

    args.init_src = True
    args.TOPK = 20
    args.repos_beam = 5
    args.mask_beam = 1
    args.token_beam = 4
    args.iter_decode_max_iter = 10
    args.iter_decode_with_beam = 1
    args.iter_decode_eos_penalty = 0.0

    criterion = task.build_criterion(args)
    generator = task.build_generator(args)
    feature_dim = int(getattr(model.args, "encoder_embed_dim", 512))
    return EditRetroAdapter(model, task, criterion, feature_dim, generator)
