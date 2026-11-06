from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from .base import HookedBackboneAdapter


class RetroComposerAdapter(HookedBackboneAdapter):
    def __init__(
        self,
        model,
        feature_dim: int,
        loss_fn: Callable[[Any, Mapping[str, Any]], torch.Tensor],
        predict_fn: Callable[[Any, Mapping[str, Any], int], list[list[str]]],
        representation_module=None,
    ):
        representation_module = representation_module or model.graph_embedding
        super().__init__(model, representation_module, feature_dim, loss_fn, predict_fn)

    def _run_representation(self, batch: Mapping[str, Any]):
        return self.model.run_gnn(batch["native_batch"], typed=bool(batch.get("typed", False)))

    def select_batch(self, batch: Mapping[str, Any], indices: torch.Tensor) -> Mapping[str, Any]:
        from torch_geometric.data import Batch

        native = batch["native_batch"]
        positions = indices.detach().cpu().tolist()
        examples = native.index_select(positions)
        selected_native = Batch.from_data_list(examples).to(native.x.device)

        for name in (
            "sequences",
            "reaction_center_atom_indexes",
            "reaction_center_cands",
            "reaction_center_cands_labels",
            "reaction_center_cands_smarts",
            "atom_len",
        ):
            if hasattr(native, name):
                values = getattr(native, name)
                setattr(selected_native, name, [values[position] for position in positions])
        reduced = {key: value for key, value in batch.items() if key != "native_batch"}
        result = dict(super().select_batch(reduced, indices))
        result["native_batch"] = selected_native
        return result


def retrocomposer_native_loss(model, batch: Mapping[str, Any]) -> torch.Tensor:
    output = model(batch["native_batch"], typed=bool(batch.get("typed", False)), decode=False)
    if torch.is_tensor(output) and output.ndim == 0:
        return output
    if isinstance(output, Mapping) and "loss" in output:
        return output["loss"]
    if isinstance(output, tuple) and len(output) >= 2 and all(torch.is_tensor(x) for x in output[:2]):
        return output[0] + output[1]
    raise TypeError("RetroComposer wrapper expected a scalar or {'loss': tensor}")
