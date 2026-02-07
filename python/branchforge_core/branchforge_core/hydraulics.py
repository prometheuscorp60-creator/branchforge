from __future__ import annotations

from typing import Dict, Tuple, List, Any
import math
import networkx as nx

from .sizing import WATER_RHO, WATER_MU, hydraulic_diameter_rect, flow_velocity_rect


def _polyline_length_mm(path: List[Tuple[float,float]]) -> float:
    if len(path) < 2:
        return 0.0
    s = 0.0
    for a, b in zip(path[:-1], path[1:]):
        s += math.hypot(b[0]-a[0], b[1]-a[1])
    return s


def friction_factor(Re: float) -> float:
    Re = max(1e-6, float(Re))
    if Re < 2300.0:
        return 64.0 / Re
    # Blasius smooth turbulent
    return 0.3164 * (Re ** -0.25)


def pressure_drop_edge_pa(
    mdot_kg_s: float,
    width_mm: float,
    depth_mm: float,
    length_mm: float,
    rho_kg_m3: float = WATER_RHO,
    mu_pa_s: float = WATER_MU,
) -> float:
    L = length_mm / 1000.0
    Dh = hydraulic_diameter_rect(width_mm, depth_mm)
    v = flow_velocity_rect(mdot_kg_s, width_mm, depth_mm, rho_kg_m3=rho_kg_m3)
    Re = rho_kg_m3 * v * Dh / mu_pa_s
    f = friction_factor(Re)
    dp = f * (L / max(1e-9, Dh)) * (rho_kg_m3 * v*v / 2.0)
    return float(dp)


def minor_loss_pa(
    mdot_kg_s: float,
    width_mm: float,
    depth_mm: float,
    K: float,
    rho_kg_m3: float = WATER_RHO,
) -> float:
    v = flow_velocity_rect(mdot_kg_s, width_mm, depth_mm, rho_kg_m3=rho_kg_m3)
    return float(K * (rho_kg_m3 * v*v / 2.0))


def estimate_network_dp_kpa(
    G_supply: nx.DiGraph,
    G_return: nx.DiGraph,
    supply_paths: Dict[Tuple[str,str], List[Tuple[float,float]]],
    return_paths: Dict[Tuple[str,str], List[Tuple[float,float]]],
    supply_mdot: Dict[Tuple[str,str], float],
    return_mdot: Dict[Tuple[str,str], float],
    supply_w: Dict[Tuple[str,str], float],
    return_w: Dict[Tuple[str,str], float],
    depth_mm: float,
    junction_meta: Dict[str, Dict[str, Any]] | None = None,
) -> float:
    """Compute worst-case inlet->leaf->outlet pressure drop (kPa).

    junction_meta: optional dict keyed by node_id with fields like:
      - regime: 'sprout'|'branch'
      - rho
      - theta_thick_deg
    """
    if junction_meta is None:
        junction_meta = {}

    # Precompute edge dp (Pa)
    dp_edge_supply = {}
    for e, path in supply_paths.items():
        mdot = supply_mdot[e]
        w = supply_w[e]
        L = _polyline_length_mm(path)
        dp = pressure_drop_edge_pa(mdot, w, depth_mm, L)
        dp_edge_supply[e] = dp

    dp_edge_return = {}
    for e, path in return_paths.items():
        mdot = return_mdot[e]
        w = return_w[e]
        L = _polyline_length_mm(path)
        dp = pressure_drop_edge_pa(mdot, w, depth_mm, L)
        dp_edge_return[e] = dp

    # Junction losses: assign to outgoing edges (child edge) from each internal node
    dp_minor_supply = {e: 0.0 for e in dp_edge_supply}
    for u in G_supply.nodes:
        if u == "root":
            continue
        if G_supply.nodes[u].get("kind") not in ("internal", "root"):
            continue
        meta = junction_meta.get(u, {})
        regime = meta.get("regime", "sprout")
        theta = float(meta.get("theta_thick_deg", 180.0))
        baseK = 0.3 if regime == "sprout" else 0.6
        # angle penalty: sharper junction => higher K
        # Use 180-theta (steering) normalized.
        steer = max(0.0, (180.0 - theta) / 60.0)  # 0..1 across typical range
        K = baseK + 0.8*(steer**2)
        for p in G_supply.predecessors(u):
            pass
        for v in G_supply.successors(u):
            e = (u, v)
            if e not in dp_minor_supply:
                continue
            mdot = supply_mdot[e]
            w = supply_w[e]
            dp_minor_supply[e] += minor_loss_pa(mdot, w, depth_mm, K)

    dp_minor_return = {e: 0.0 for e in dp_edge_return}
    for u in G_return.nodes:
        if u == "root":
            continue
        if G_return.nodes[u].get("kind") not in ("internal", "root"):
            continue
        meta = junction_meta.get(f"R_{u}", junction_meta.get(u, {}))
        regime = meta.get("regime", "sprout")
        theta = float(meta.get("theta_thick_deg", 180.0))
        baseK = 0.3 if regime == "sprout" else 0.6
        steer = max(0.0, (180.0 - theta) / 60.0)
        K = baseK + 0.8*(steer**2)
        for v in G_return.successors(u):
            e = (u, v)
            if e not in dp_minor_return:
                continue
            mdot = return_mdot[e]
            w = return_w[e]
            dp_minor_return[e] += minor_loss_pa(mdot, w, depth_mm, K)

    # Worst path over leaves
    worst_pa = 0.0
    leaves = [n for n in G_supply.nodes if G_supply.nodes[n].get("kind") == "leaf"]
    for leaf in leaves:
        # supply path edges root->leaf
        path_nodes = nx.shortest_path(G_supply, "root", leaf)
        edges_sup = list(zip(path_nodes[:-1], path_nodes[1:]))
        dp_sup = sum(dp_edge_supply[e] + dp_minor_supply.get(e, 0.0) for e in edges_sup)

        # return path edges root(outlet)->leaf, but flow is leaf->root, dp same
        path_nodes_r = nx.shortest_path(G_return, "root", leaf)
        edges_ret = list(zip(path_nodes_r[:-1], path_nodes_r[1:]))
        dp_ret = sum(dp_edge_return[e] + dp_minor_return.get(e, 0.0) for e in edges_ret)

        dp = dp_sup + dp_ret
        if dp > worst_pa:
            worst_pa = dp

    return float(worst_pa / 1000.0)  # kPa
