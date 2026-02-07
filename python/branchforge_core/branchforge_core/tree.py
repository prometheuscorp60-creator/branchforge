from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import networkx as nx

from .clustering import weighted_kmeans
from .geometry import clamp_point_to_allowed


def build_supply_tree(
    root_xy: Tuple[float, float],
    leaf_xy: List[Tuple[float, float]],
    leaf_mass_flows: List[float],
    allowed_region,
    seed: int = 0,
) -> nx.DiGraph:
    """Build a directed binary tree from root -> leaves.

    Uses recursive weighted k=2 clustering to create Steiner-like internal nodes at centroids.

    Node attribute conventions:
    - kind: root|internal|leaf
    - xy: (x_mm, y_mm)
    - leaf_mdot_kg_s: only for leaf nodes
    """
    G = nx.DiGraph()

    # root
    G.add_node("root", xy=root_xy, kind="root")

    # leaves
    for i, (xy, mdot) in enumerate(zip(leaf_xy, leaf_mass_flows)):
        nid = f"leaf_{i}"
        G.add_node(nid, xy=xy, kind="leaf", leaf_mdot_kg_s=float(mdot))

    leaf_ids = [f"leaf_{i}" for i in range(len(leaf_xy))]

    def rec(parent_id: str, group_leaf_ids: List[str], depth: int, local_seed: int):
        if len(group_leaf_ids) == 1:
            G.add_edge(parent_id, group_leaf_ids[0])
            return

        pts = np.array([G.nodes[n]["xy"] for n in group_leaf_ids], dtype=float)
        wts = np.array([G.nodes[n].get("leaf_mdot_kg_s", 1.0) for n in group_leaf_ids], dtype=float)

        centers = weighted_kmeans(pts, wts, k=2, n_iter=30, seed=local_seed)
        dist2 = np.sum((pts[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = np.argmin(dist2, axis=1)

        group_a = [group_leaf_ids[i] for i in range(len(group_leaf_ids)) if labels[i] == 0]
        group_b = [group_leaf_ids[i] for i in range(len(group_leaf_ids)) if labels[i] == 1]

        if len(group_a) == 0 or len(group_b) == 0:
            order = sorted(group_leaf_ids, key=lambda n: G.nodes[n]["xy"][0])
            mid = len(order)//2
            group_a = order[:mid]
            group_b = order[mid:]

        def make_internal_node(group: List[str], tag: str):
            pts_g = np.array([G.nodes[n]["xy"] for n in group], dtype=float)
            w_g = np.array([G.nodes[n].get("leaf_mdot_kg_s", 1.0) for n in group], dtype=float)
            s = w_g.sum()
            if s <= 0:
                c = pts_g.mean(axis=0)
            else:
                c = (pts_g * w_g[:, None]).sum(axis=0) / s
            cxy = (float(c[0]), float(c[1]))
            cxy = clamp_point_to_allowed(cxy, allowed_region)
            node_id = f"int_{depth}_{tag}_{len(G.nodes)}"
            G.add_node(node_id, xy=cxy, kind="internal")
            return node_id

        node_a = make_internal_node(group_a, "A") if len(group_a) > 1 else group_a[0]
        node_b = make_internal_node(group_b, "B") if len(group_b) > 1 else group_b[0]

        if node_a != parent_id:
            G.add_edge(parent_id, node_a)
        if node_b != parent_id:
            G.add_edge(parent_id, node_b)

        if G.nodes[node_a]["kind"] == "internal":
            rec(node_a, group_a, depth + 1, local_seed + 1)
        if G.nodes[node_b]["kind"] == "internal":
            rec(node_b, group_b, depth + 1, local_seed + 2)

    rec("root", leaf_ids, depth=0, local_seed=seed)
    return G


def build_return_tree(
    root_xy: Tuple[float, float],
    leaf_xy: List[Tuple[float, float]],
    leaf_mass_flows: List[float],
    allowed_region,
    seed: int = 0,
) -> nx.DiGraph:
    return build_supply_tree(root_xy, leaf_xy, leaf_mass_flows, allowed_region, seed=seed)


def postorder_nodes(G: nx.DiGraph) -> List[str]:
    order = list(nx.topological_sort(G))
    return order[::-1]


def compute_edge_mass_flows(G: nx.DiGraph) -> Dict[Tuple[str, str], float]:
    """Edge flow = sum of leaf flows in the child subtree (kg/s)."""
    node_flow = {}
    for n in G.nodes:
        if G.nodes[n]["kind"] == "leaf":
            node_flow[n] = float(G.nodes[n].get("leaf_mdot_kg_s", 0.0))
        else:
            node_flow[n] = 0.0

    for n in postorder_nodes(G):
        for p in G.predecessors(n):
            node_flow[p] += node_flow[n]

    flow = {}
    for u, v in G.edges:
        flow[(u, v)] = node_flow[v]
    return flow


def node_leaf_mass_flow(G: nx.DiGraph, leaf_id: str) -> float:
    return float(G.nodes[leaf_id].get("leaf_mdot_kg_s", 0.0))
