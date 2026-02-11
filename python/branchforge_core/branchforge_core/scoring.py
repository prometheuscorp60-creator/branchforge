from __future__ import annotations

from typing import List, Tuple, Dict, Any
import math
import numpy as np
import networkx as nx

from .sizing import WATER_CP
from .heatmap import grid_cell_centers


# ---------------------------------------------------------------------------
# Surface-area metrics (Meng et al. Nature 649, 315-322, 2026)
# ---------------------------------------------------------------------------

def total_wetted_surface_area_mm2(
    edge_paths: Dict[Tuple[str, str], List[Tuple[float, float]]],
    edge_widths: Dict[Tuple[str, str], float],
    depth_mm: float,
) -> float:
    """Total wetted surface area of the channel network [mm^2].

    For a rectangular channel, wetted perimeter = 2*(width + depth),
    and wetted surface area = perimeter * length.

    This is the quantity the Nature paper's branching networks minimize.
    """
    sa = 0.0
    for e, path in edge_paths.items():
        if len(path) < 2:
            continue
        w = edge_widths.get(e, 1.0)
        perimeter = 2.0 * (w + depth_mm)
        length = 0.0
        for a, b in zip(path[:-1], path[1:]):
            length += math.hypot(b[0] - a[0], b[1] - a[1])
        sa += perimeter * length
    return float(sa)


def murray_law_deviation(
    G: nx.DiGraph,
    edge_widths: Dict[Tuple[str, str], float],
    depth_mm: float,
    exponent: float = 3.0,
) -> float:
    """Measure deviation from Murray's law (generalized) across all bifurcations.

    Murray's law: d_parent^n = sum(d_child_i^n)
    For the Nature paper's surface-minimization networks, n=3 is the classical
    exponent. The generalized form from Meng et al. may use a modified exponent
    depending on the objective (surface area vs. dissipation).

    Returns the RMS relative deviation: 0.0 = perfect Murray's law adherence.
    """
    deviations = []

    for node in G.nodes:
        kind = G.nodes[node].get("kind")
        if kind not in ("internal", "root"):
            continue

        # Find parent edge (incoming)
        parent_edges = list(G.in_edges(node))
        if not parent_edges:
            continue

        # Find child edges (outgoing)
        child_edges = list(G.out_edges(node))
        if len(child_edges) < 2:
            continue

        # Parent hydraulic diameter
        pe = parent_edges[0]
        w_p = edge_widths.get(pe, 0.0)
        if w_p <= 0:
            continue
        d_p = 2.0 * w_p * depth_mm / (w_p + depth_mm)  # hydraulic diameter

        # Children hydraulic diameters
        d_children_sum = 0.0
        for ce in child_edges:
            w_c = edge_widths.get(ce, 0.0)
            if w_c <= 0:
                continue
            d_c = 2.0 * w_c * depth_mm / (w_c + depth_mm)
            d_children_sum += d_c ** exponent

        if d_children_sum <= 0:
            continue

        d_p_n = d_p ** exponent
        # Relative deviation: |d_p^n - sum(d_c^n)| / d_p^n
        dev = abs(d_p_n - d_children_sum) / max(d_p_n, 1e-12)
        deviations.append(dev)

    if not deviations:
        return 0.0
    return float(np.sqrt(np.mean(np.array(deviations) ** 2)))


def surface_area_efficiency(
    total_sa_mm2: float,
    total_channel_volume_mm3: float,
) -> float:
    """Surface-to-volume ratio efficiency.

    Lower SA/V is better (the paper's networks minimize surface area for a
    given volume). Returns the ratio SA/V, which the optimizer should minimize.
    """
    if total_channel_volume_mm3 <= 0:
        return float('inf')
    return float(total_sa_mm2 / total_channel_volume_mm3)


def channel_volume_mm3(
    edge_paths: Dict[Tuple[str, str], List[Tuple[float, float]]],
    edge_widths: Dict[Tuple[str, str], float],
    depth_mm: float,
) -> float:
    """Total volume of all channels [mm^3]."""
    vol = 0.0
    for e, path in edge_paths.items():
        if len(path) < 2:
            continue
        w = edge_widths.get(e, 1.0)
        length = 0.0
        for a, b in zip(path[:-1], path[1:]):
            length += math.hypot(b[0] - a[0], b[1] - a[1])
        vol += w * depth_mm * length
    return float(vol)


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
