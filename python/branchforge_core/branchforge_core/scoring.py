from __future__ import annotations

from typing import List, Tuple, Dict, Any
import math
import numpy as np
import networkx as nx

from .sizing import WATER_CP
from .heatmap import grid_cell_centers


def assign_heat_to_leaves(
    plate_w_mm: float,
    plate_h_mm: float,
    heat_grid_watts: np.ndarray,
    leaf_xy: List[Tuple[float,float]],
) -> np.ndarray:
    """Assign each heatmap cell to nearest leaf centroid and sum watts per leaf."""
    pts = grid_cell_centers(plate_w_mm, plate_h_mm, heat_grid_watts)  # (ny,nx,2)
    flat_pts = pts.reshape(-1, 2)
    flat_heat = heat_grid_watts.reshape(-1)
    leaf = np.array(leaf_xy, dtype=float)  # (k,2)
    # distance to each leaf
    dist2 = np.sum((flat_pts[:, None, :] - leaf[None, :, :])**2, axis=2)  # (n,k)
    labels = np.argmin(dist2, axis=1)
    k = len(leaf_xy)
    sums = np.zeros(k, dtype=float)
    for i in range(len(flat_heat)):
        sums[labels[i]] += float(flat_heat[i])
    return sums


def leaf_deltaT(leaf_watts: np.ndarray, leaf_mdot_kg_s: np.ndarray, cp_j_kgk: float = WATER_CP) -> np.ndarray:
    mdot = np.maximum(leaf_mdot_kg_s, 1e-9)
    return leaf_watts / (mdot * cp_j_kgk)


def path_bend_count(path: List[Tuple[float,float]], angle_threshold_deg: float = 10.0) -> int:
    if len(path) < 3:
        return 0
    thresh = math.radians(angle_threshold_deg)
    cnt = 0
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        v1 = (b[0]-a[0], b[1]-a[1])
        v2 = (c[0]-b[0], c[1]-b[1])
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cosang = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
        cosang = max(-1.0, min(1.0, cosang))
        ang = math.acos(cosang)
        if ang >= thresh:
            cnt += 1
    return cnt


def manufacturing_score(
    all_edge_paths: Dict[Tuple[str,str], List[Tuple[float,float]]],
    routing_success: Dict[Tuple[str,str], bool],
) -> float:
    bends = 0
    failures = 0
    for e, p in all_edge_paths.items():
        bends += path_bend_count(p)
        if not routing_success.get(e, True):
            failures += 1
    # simple scoring:
    # - failures are brutal
    # - bends are mild
    score = 1.0
    score *= math.exp(-0.15*bends)
    score *= math.exp(-1.5*failures)
    # clamp
    return float(max(0.0, min(1.0, score)))
