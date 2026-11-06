from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from .base import HookedBackboneAdapter


class GDiffRetroAdapter(HookedBackboneAdapter):
    def __init__(
        self,
        center_task,
        graph_encoder,
        feature_dim: int,
        predict_fn: Callable[[Any, Mapping[str, Any], int], list[list[str]]],
    ):
        def loss_fn(task, batch):
            loss, _ = task(batch["native_batch"])
            return loss

        super().__init__(center_task, graph_encoder, feature_dim, loss_fn, predict_fn)
        self.graph_encoder = graph_encoder

    def _run_representation(self, batch: Mapping[str, Any]):
        product = batch["native_batch"]["graph"][1]
        return self.graph_encoder(product, product.node_feature.float())

    def select_batch(self, batch: Mapping[str, Any], indices: torch.Tensor) -> Mapping[str, Any]:
        native = batch["native_batch"]
        reduced = {key: value for key, value in batch.items() if key != "native_batch"}
        result = dict(super().select_batch(reduced, indices))
        selected_native = dict(super().select_batch(native, indices))
        positions = indices.detach().cpu().tolist()
        selected_native["graph"] = tuple(graph[positions] for graph in native["graph"])
        result["native_batch"] = selected_native
        return result
