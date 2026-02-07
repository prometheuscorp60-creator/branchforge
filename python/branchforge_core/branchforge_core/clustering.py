from __future__ import annotations

from typing import Tuple
import numpy as np


def _weighted_choice(rng: np.random.Generator, probs: np.ndarray) -> int:
    probs = np.asarray(probs, dtype=float)
    s = probs.sum()
    if s <= 0:
        return int(rng.integers(0, len(probs)))
    probs = probs / s
    return int(rng.choice(len(probs), p=probs))


def weighted_kmeans(points: np.ndarray, weights: np.ndarray, k: int, n_iter: int = 25, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(points)
    if k <= 0:
        raise ValueError("k must be >= 1")
    if n == 0:
        raise ValueError("No points provided")
    k = min(k, n)

    rng = np.random.default_rng(seed)

    # k-means++ init
    centroids = np.zeros((k, 2), dtype=float)
    idx0 = _weighted_choice(rng, weights)
    centroids[0] = points[idx0]
    d2 = np.sum((points - centroids[0])**2, axis=1)

    for i in range(1, k):
        probs = weights * d2
        idx = _weighted_choice(rng, probs)
        centroids[i] = points[idx]
        d2 = np.minimum(d2, np.sum((points - centroids[i])**2, axis=1))

    # Lloyd iterations (weighted)
    for _ in range(n_iter):
        # Assign
        dist2 = np.sum((points[:, None, :] - centroids[None, :, :])**2, axis=2)  # (n,k)
        labels = np.argmin(dist2, axis=1)
        # Update
        new_c = centroids.copy()
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                continue
            w = weights[mask]
            s = w.sum()
            if s <= 0:
                continue
            new_c[j] = np.sum(points[mask] * w[:, None], axis=0) / s
        shift = np.sqrt(np.sum((new_c - centroids)**2))
        centroids = new_c
        if shift < 1e-6:
            break

    return centroids


def sample_weighted(points: np.ndarray, weights: np.ndarray, max_samples: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(points)
    if n <= max_samples:
        return points, weights
    rng = np.random.default_rng(seed)
    probs = weights.clip(min=0)
    s = probs.sum()
    if s <= 0:
        idx = rng.choice(n, size=max_samples, replace=False)
        return points[idx], weights[idx]
    probs = probs / s
    idx = rng.choice(n, size=max_samples, replace=False, p=probs)
    return points[idx], weights[idx]
