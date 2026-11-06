from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from .chemistry import morgan_bits, reaction_difference_bits


def _tanimoto_block(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    query = query.astype(np.float32, copy=False)
    database = database.astype(np.float32, copy=False)
    intersections = query @ database.T
    qsum = query.sum(axis=1, keepdims=True)
    dsum = database.sum(axis=1, keepdims=True).T
    unions = qsum + dsum - intersections
    return np.divide(intersections, unions, out=np.ones_like(intersections), where=unions != 0)


def exact_tanimoto_knn(features: np.ndarray, k: int = 10, block_size: int = 256) -> sparse.csr_matrix:
    n = len(features)
    if n < 2:
        raise ValueError("At least two samples are required")
    k = min(k, n - 1)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        similarity = _tanimoto_block(features[start:stop], features)
        similarity[np.arange(stop - start), np.arange(start, stop)] = -1
        indices = np.argpartition(similarity, -k, axis=1)[:, -k:]
        scores = np.take_along_axis(similarity, indices, axis=1)
        order = np.argsort(-scores, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        scores = np.take_along_axis(scores, order, axis=1)
        rows.append(np.repeat(np.arange(start, stop), k))
        cols.append(indices.reshape(-1))
        vals.append(scores.reshape(-1))
    graph = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return graph.maximum(graph.T)


def spectral_partition(
    affinity: sparse.csr_matrix,
    n_clusters: int,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = laplacian(affinity, normed=True)
    _, vectors = eigsh(
        normalized, k=n_clusters, which="SM", tol=1e-5,
        v0=np.full(normalized.shape[0], 1.0 / np.sqrt(normalized.shape[0])),
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    embedding = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    labels = capacity_constrained_kmeans(embedding, n_clusters, seed=seed)
    return labels.astype(np.int16), embedding.astype(np.float32)


def capacity_constrained_kmeans(
    embedding: np.ndarray,
    n_clusters: int,
    seed: int = 2026,
    max_iter: int = 50,
) -> np.ndarray:
    embedding = np.asarray(embedding)
    n_samples = len(embedding)
    if not 1 < n_clusters <= n_samples:
        raise ValueError("n_clusters must be between 2 and the sample count")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    initial = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=20,
    ).fit(embedding)
    centroids = initial.cluster_centers_.copy()
    capacities = np.full(n_clusters, n_samples // n_clusters, dtype=np.int64)
    capacities[: n_samples % n_clusters] += 1
    previous: np.ndarray | None = None

    for _ in range(max_iter):
        distances = ((embedding[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        preferences = np.argsort(distances, axis=1, kind="stable")
        ordered_distances = np.take_along_axis(distances, preferences, axis=1)
        confidence = ordered_distances[:, 1] - ordered_distances[:, 0]
        point_order = np.argsort(-confidence, kind="stable")

        labels = np.full(n_samples, -1, dtype=np.int16)
        remaining = capacities.copy()
        for row in point_order:
            for cluster in preferences[row]:
                if remaining[cluster] > 0:
                    labels[row] = cluster
                    remaining[cluster] -= 1
                    break
        if np.any(labels < 0) or np.any(remaining != 0):
            raise RuntimeError("Capacity-constrained k-means assignment is incomplete")

        centroids = np.stack(
            [embedding[labels == cluster].mean(axis=0) for cluster in range(n_clusters)]
        )
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
    return labels


@dataclass
class SDEResult:
    frame: pd.DataFrame
    product_features: np.ndarray
    template_features: np.ndarray
    product_embedding: np.ndarray
    template_embedding: np.ndarray
    product_affinity: sparse.csr_matrix
    template_affinity: sparse.csr_matrix


def fit_sde_frame(
    frame: pd.DataFrame,
    km: int = 4,
    kt: int = 3,
    k_neighbors: int = 10,
    seed: int = 2026,
) -> SDEResult:
    train = frame[frame["split"] == "train"].copy().reset_index(drop=True)
    if len(train) < max(km, kt, k_neighbors + 1):
        raise ValueError("Training split is too small for requested SDE settings")
    product_features = np.stack([morgan_bits(x) for x in train["product"]])
    template_features = np.stack(
        [reaction_difference_bits(p, r) for p, r in zip(train["product"], train["reactants"])]
    )
    product_affinity = exact_tanimoto_knn(product_features, k=k_neighbors)
    template_affinity = exact_tanimoto_knn(template_features, k=k_neighbors)
    mol_labels, mol_embedding = spectral_partition(product_affinity, km, seed=seed)
    template_labels, template_embedding = spectral_partition(template_affinity, kt, seed=seed)
    train["molecular_domain"] = mol_labels
    train["template_domain"] = template_labels
    train["domain_id"] = mol_labels * kt + template_labels + 1
    return SDEResult(
        frame=train,
        product_features=product_features,
        template_features=template_features,
        product_embedding=mol_embedding,
        template_embedding=template_embedding,
        product_affinity=product_affinity,
        template_affinity=template_affinity,
    )
