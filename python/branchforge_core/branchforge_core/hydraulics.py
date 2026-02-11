from __future__ import annotations

from typing import Dict, Tuple, List, Any
import math
import networkx as nx

from .sizing import WATER_RHO, WATER_MU, hydraulic_diameter_rect, flow_velocity_rect


# ---------------------------------------------------------------------------
# Fluid property database
# ---------------------------------------------------------------------------

# (rho [kg/m3], mu [Pa*s], cp [J/kg/K]) at ~25 C
_FLUID_PROPERTIES: Dict[str, Tuple[float, float, float]] = {
    "water":                (997.0,  0.00089, 4181.0),
    "ethylene_glycol_30":   (1040.0, 0.00200, 3580.0),
    "ethylene_glycol_50":   (1082.0, 0.00340, 3283.0),
    "ethylene_glycol_70":   (1110.0, 0.00800, 2900.0),
    "propylene_glycol_50":  (1035.0, 0.00500, 3500.0),
    "novec_7100":           (1510.0, 0.00058, 1183.0),
    "pao_2":                (798.0,  0.00520, 2090.0),
}


def get_fluid_properties(coolant: str, **custom) -> Tuple[float, float, float]:
    """Return (rho_kg_m3, mu_pa_s, cp_J_kgK) for a named coolant.

    Pass coolant="custom" with custom_rho_kg_m3, custom_mu_pa_s,
    custom_cp_J_kgK keyword args for user-defined fluids.
    """
    key = coolant.strip().lower()
    if key == "custom":
        return (
            custom.get("custom_rho_kg_m3", WATER_RHO),
            custom.get("custom_mu_pa_s", WATER_MU),
            custom.get("custom_cp_J_kgK", 4181.0),
        )
    if key not in _FLUID_PROPERTIES:
        raise ValueError(
            f"Unknown coolant '{coolant}'. "
            f"Supported: {sorted(_FLUID_PROPERTIES.keys())}"
        )
    return _FLUID_PROPERTIES[key]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _polyline_length_mm(path: List[Tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    s = 0.0
    for a, b in zip(path[:-1], path[1:]):
        s += math.hypot(b[0] - a[0], b[1] - a[1])
    return s


def _cross_section_area_m2(width_mm: float, depth_mm: float) -> float:
    return (width_mm / 1000.0) * (depth_mm / 1000.0)


# ---------------------------------------------------------------------------
# Friction factor
# ---------------------------------------------------------------------------

def friction_factor(Re: float) -> float:
    """Darcy friction factor.

    - Laminar  (Re < 2300): f = 64 / Re
    - Turbulent (Re >= 2300): Blasius smooth-pipe correlation f = 0.3164 * Re^-0.25
    """
    Re = max(1e-6, float(Re))
    if Re < 2300.0:
        return 64.0 / Re
    # Blasius smooth turbulent
    return 0.3164 * (Re ** -0.25)


# ---------------------------------------------------------------------------
# Reynolds number
# ---------------------------------------------------------------------------

def reynolds_number(
    mdot_kg_s: float,
    width_mm: float,
    depth_mm: float,
    rho_kg_m3: float = WATER_RHO,
    mu_pa_s: float = WATER_MU,
) -> float:
    """Compute Reynolds number for flow in a rectangular channel."""
    Dh = hydraulic_diameter_rect(width_mm, depth_mm)
    v = flow_velocity_rect(mdot_kg_s, width_mm, depth_mm, rho_kg_m3=rho_kg_m3)
    return rho_kg_m3 * v * Dh / mu_pa_s


# ---------------------------------------------------------------------------
# Straight-channel friction loss
# ---------------------------------------------------------------------------

def pressure_drop_edge_pa(
    mdot_kg_s: float,
    width_mm: float,
    depth_mm: float,
    length_mm: float,
    rho_kg_m3: float = WATER_RHO,
    mu_pa_s: float = WATER_MU,
) -> float:
    """Darcy-Weisbach friction pressure drop [Pa] for a rectangular channel."""
    L = length_mm / 1000.0
    Dh = hydraulic_diameter_rect(width_mm, depth_mm)
    v = flow_velocity_rect(mdot_kg_s, width_mm, depth_mm, rho_kg_m3=rho_kg_m3)
    Re = rho_kg_m3 * v * Dh / mu_pa_s
    f = friction_factor(Re)
    dp = f * (L / max(1e-9, Dh)) * (rho_kg_m3 * v * v / 2.0)
    return float(dp)


# ---------------------------------------------------------------------------
# Generic minor loss
# ---------------------------------------------------------------------------

def minor_loss_pa(
    mdot_kg_s: float,
    width_mm: float,
    depth_mm: float,
    K: float,
    rho_kg_m3: float = WATER_RHO,
) -> float:
    """K-factor minor loss: dP = K * (rho * v^2 / 2)."""
    v = flow_velocity_rect(mdot_kg_s, width_mm, depth_mm, rho_kg_m3=rho_kg_m3)
    return float(K * (rho_kg_m3 * v * v / 2.0))


# ---------------------------------------------------------------------------
# Contraction / expansion losses
# ---------------------------------------------------------------------------

def contraction_K(area_ratio: float) -> float:
    """Sudden contraction loss coefficient.

    area_ratio = A_small / A_large  (0 < area_ratio <= 1).
    Uses the classical Borda-Carnot relation for a sharp-edged contraction:
        K_c = 0.5 * (1 - area_ratio)
    Applied to the velocity in the *smaller* (downstream) section.
    """
    ar = max(1e-6, min(1.0, float(area_ratio)))
    return 0.5 * (1.0 - ar)


def expansion_K(area_ratio: float) -> float:
    """Sudden expansion loss coefficient.

    area_ratio = A_small / A_large  (0 < area_ratio <= 1).
    Borda-Carnot expansion loss (referenced to upstream/smaller velocity):
        K_e = (1 - area_ratio)^2
    """
    ar = max(1e-6, min(1.0, float(area_ratio)))
    return (1.0 - ar) ** 2


# ---------------------------------------------------------------------------
# Port entry / exit losses
# ---------------------------------------------------------------------------

# Sharp-edged inlet (reservoir -> channel): K = 0.5
PORT_ENTRY_K = 0.5

# Sharp-edged outlet (channel -> reservoir): K = 1.0
PORT_EXIT_K = 1.0


def port_entry_loss_pa(
    mdot_kg_s: float,
    width_mm: float,
    depth_mm: float,
    rho_kg_m3: float = WATER_RHO,
) -> float:
    """Pressure loss at a sharp-edged inlet port (reservoir -> channel)."""
    return minor_loss_pa(mdot_kg_s, width_mm, depth_mm, PORT_ENTRY_K, rho_kg_m3)


def port_exit_loss_pa(
    mdot_kg_s: float,
    width_mm: float,
    depth_mm: float,
    rho_kg_m3: float = WATER_RHO,
) -> float:
    """Pressure loss at a sharp-edged exit port (channel -> reservoir)."""
    return minor_loss_pa(mdot_kg_s, width_mm, depth_mm, PORT_EXIT_K, rho_kg_m3)


# ---------------------------------------------------------------------------
# Idelchik-style junction K-factor model
# ---------------------------------------------------------------------------

def junction_K_idelchik(
    theta_deg: float,
    area_ratio: float,
    flow_split_ratio: float,
    regime: str = "sprout",
) -> float:
    """Compute junction loss coefficient using an Idelchik-inspired correlation.

    Parameters
    ----------
    theta_deg : float
        Branch angle in degrees (0 = straight run, 90 = right-angle tee,
        180 = hairpin reversal).  In BranchForge, ``theta_thick_deg``
        typically describes the *included* angle of the junction.
    area_ratio : float
        A_branch / A_main  (cross-section area ratio, 0 < ar <= 1).
        When the child channel is narrower than the parent, ar < 1.
    flow_split_ratio : float
        q_branch / q_main  (mass-flow fraction going into this branch,
        0 < fsr <= 1).
    regime : str
        'sprout' (gentle bifurcation) or 'branch' (sharper split).
        Sprout junctions get a lower base K because the geometry is
        designed with smooth transitions.

    Returns
    -------
    float
        Loss coefficient K referenced to the branch velocity.

    Notes
    -----
    Idelchik's handbook (Diagram 7-20 ff.) expresses the branch loss
    coefficient of a dividing tee as a function of the branch angle,
    area ratio, and flow-split ratio.  The full tables are complex; here
    we use a simplified analytical fit that captures the dominant trends:

        K = C_base
            + C_angle * (1 - cos(theta))
            + C_area  * (1 - ar)^2
            + C_flow  * fsr^2

    The coefficients are chosen to reproduce typical Idelchik values for
    rectangular micro-channel junctions.
    """
    theta_rad = math.radians(max(0.0, min(180.0, float(theta_deg))))
    ar = max(1e-6, min(1.0, float(area_ratio)))
    fsr = max(0.0, min(1.0, float(flow_split_ratio)))

    # Base loss: sprout junctions are optimised, branch junctions are sharper
    if regime == "sprout":
        C_base = 0.15
    else:
        C_base = 0.35

    # Angle term: straight-through (0 deg) has zero extra loss; 90 deg tee
    # adds significant loss; 180 deg is worst.
    C_angle = 1.2
    angle_term = C_angle * (1.0 - math.cos(theta_rad))

    # Area-ratio term: mismatch between parent and branch cross-sections
    # penalises the flow (Idelchik shows this as roughly quadratic).
    C_area = 0.5
    area_term = C_area * (1.0 - ar) ** 2

    # Flow-split term: when a large fraction of the flow turns into the
    # branch, momentum-change losses increase roughly as fsr^2.
    C_flow = 0.8
    flow_term = C_flow * fsr ** 2

    K = C_base + angle_term + area_term + flow_term
    return float(K)


# ---------------------------------------------------------------------------
# Width-change (contraction / expansion) loss between parent and child edges
# ---------------------------------------------------------------------------

def _width_change_loss_pa(
    mdot_child_kg_s: float,
    w_parent_mm: float,
    w_child_mm: float,
    depth_mm: float,
    rho_kg_m3: float = WATER_RHO,
) -> float:
    """Compute pressure loss [Pa] due to a sudden width change at a junction.

    If the child channel is narrower than the parent it is a contraction;
    if wider it is an expansion.  Loss is referenced to the velocity in the
    *smaller* cross-section.
    """
    if abs(w_parent_mm - w_child_mm) < 1e-6:
        return 0.0

    A_parent = _cross_section_area_m2(w_parent_mm, depth_mm)
    A_child = _cross_section_area_m2(w_child_mm, depth_mm)

    if A_child < A_parent:
        # Contraction: child is narrower
        ar = A_child / A_parent
        K = contraction_K(ar)
        # Reference velocity is in the smaller (child) section
        v = flow_velocity_rect(mdot_child_kg_s, w_child_mm, depth_mm,
                               rho_kg_m3=rho_kg_m3)
    else:
        # Expansion: child is wider
        ar = A_parent / A_child
        K = expansion_K(ar)
        # Reference velocity is in the smaller (parent) section
        # but we only know child mdot; for mass continuity it equals parent mdot
        v = flow_velocity_rect(mdot_child_kg_s, w_parent_mm, depth_mm,
                               rho_kg_m3=rho_kg_m3)

    return float(K * (rho_kg_m3 * v * v / 2.0))


# ---------------------------------------------------------------------------
# Network pressure-drop estimation
# ---------------------------------------------------------------------------

def estimate_network_dp_kpa(
    G_supply: nx.DiGraph,
    G_return: nx.DiGraph,
    supply_paths: Dict[Tuple[str, str], List[Tuple[float, float]]],
    return_paths: Dict[Tuple[str, str], List[Tuple[float, float]]],
    supply_mdot: Dict[Tuple[str, str], float],
    return_mdot: Dict[Tuple[str, str], float],
    supply_w: Dict[Tuple[str, str], float],
    return_w: Dict[Tuple[str, str], float],
    depth_mm: float,
    junction_meta: Dict[str, Dict[str, Any]] | None = None,
    rho_kg_m3: float = WATER_RHO,
    mu_pa_s: float = WATER_MU,
) -> Tuple[float, float]:
    """Compute worst-case inlet->leaf->outlet pressure drop (kPa).

    Returns
    -------
    (dp_kpa, max_reynolds) : Tuple[float, float]
        dp_kpa      - worst-case total pressure drop in kPa
        max_reynolds - maximum Reynolds number observed across all edges

    Parameters
    ----------
    junction_meta : dict, optional
        Keyed by node_id with fields:
          - regime: 'sprout' | 'branch'
          - theta_thick_deg: junction included angle
    rho_kg_m3 : float
        Fluid density (default: water at 25 C).
    mu_pa_s : float
        Fluid dynamic viscosity (default: water at 25 C).
    """
    if junction_meta is None:
        junction_meta = {}

    max_Re = 0.0

    # ------------------------------------------------------------------
    # Helper: compute edge friction dp and track Reynolds number
    # ------------------------------------------------------------------
    def _edge_dp_and_re(
        paths: Dict[Tuple[str, str], List[Tuple[float, float]]],
        mdots: Dict[Tuple[str, str], float],
        widths: Dict[Tuple[str, str], float],
    ) -> Dict[Tuple[str, str], float]:
        nonlocal max_Re
        dp_map: Dict[Tuple[str, str], float] = {}
        for e, path in paths.items():
            mdot = mdots[e]
            w = widths[e]
            L = _polyline_length_mm(path)
            dp = pressure_drop_edge_pa(mdot, w, depth_mm, L,
                                       rho_kg_m3=rho_kg_m3,
                                       mu_pa_s=mu_pa_s)
            dp_map[e] = dp
            Re = reynolds_number(mdot, w, depth_mm,
                                 rho_kg_m3=rho_kg_m3,
                                 mu_pa_s=mu_pa_s)
            if Re > max_Re:
                max_Re = Re
        return dp_map

    dp_edge_supply = _edge_dp_and_re(supply_paths, supply_mdot, supply_w)
    dp_edge_return = _edge_dp_and_re(return_paths, return_mdot, return_w)

    # ------------------------------------------------------------------
    # Junction minor losses (supply tree)
    # ------------------------------------------------------------------
    dp_minor_supply: Dict[Tuple[str, str], float] = {e: 0.0 for e in dp_edge_supply}

    for u in G_supply.nodes:
        if u == "root":
            continue
        node_kind = G_supply.nodes[u].get("kind")
        if node_kind not in ("internal", "root"):
            continue

        meta = junction_meta.get(u, {})
        regime = meta.get("regime", "sprout")
        theta = float(meta.get("theta_thick_deg", 180.0))

        # Determine parent edge width (for area-ratio and contraction/expansion)
        parent_edges = list(G_supply.in_edges(u))
        if parent_edges:
            parent_edge = parent_edges[0]
            w_parent = supply_w.get(parent_edge, 0.0)
            mdot_parent = supply_mdot.get(parent_edge, 0.0)
        else:
            w_parent = 0.0
            mdot_parent = 0.0

        # Total outflow from this node (for flow-split ratio)
        child_edges = [(u, v) for v in G_supply.successors(u)
                       if (u, v) in dp_minor_supply]
        total_child_mdot = sum(supply_mdot.get(ce, 0.0) for ce in child_edges)

        for ce in child_edges:
            mdot_child = supply_mdot[ce]
            w_child = supply_w[ce]

            # Flow split ratio
            if total_child_mdot > 1e-12:
                fsr = mdot_child / total_child_mdot
            else:
                fsr = 0.5

            # Area ratio (branch / parent)
            if w_parent > 1e-6:
                ar = _cross_section_area_m2(w_child, depth_mm) / \
                     _cross_section_area_m2(w_parent, depth_mm)
                ar = max(1e-6, min(1.0, ar))
            else:
                ar = 1.0

            # Idelchik junction K
            K_junc = junction_K_idelchik(theta, ar, fsr, regime)
            dp_minor_supply[ce] += minor_loss_pa(mdot_child, w_child, depth_mm,
                                                  K_junc, rho_kg_m3)

            # Contraction / expansion loss at width change
            if w_parent > 1e-6:
                dp_minor_supply[ce] += _width_change_loss_pa(
                    mdot_child, w_parent, w_child, depth_mm, rho_kg_m3)

    # ------------------------------------------------------------------
    # Junction minor losses (return tree)
    # ------------------------------------------------------------------
    dp_minor_return: Dict[Tuple[str, str], float] = {e: 0.0 for e in dp_edge_return}

    for u in G_return.nodes:
        if u == "root":
            continue
        node_kind = G_return.nodes[u].get("kind")
        if node_kind not in ("internal", "root"):
            continue

        meta = junction_meta.get(f"R_{u}", junction_meta.get(u, {}))
        regime = meta.get("regime", "sprout")
        theta = float(meta.get("theta_thick_deg", 180.0))

        parent_edges = list(G_return.in_edges(u))
        if parent_edges:
            parent_edge = parent_edges[0]
            w_parent = return_w.get(parent_edge, 0.0)
            mdot_parent = return_mdot.get(parent_edge, 0.0)
        else:
            w_parent = 0.0
            mdot_parent = 0.0

        child_edges = [(u, v) for v in G_return.successors(u)
                       if (u, v) in dp_minor_return]
        total_child_mdot = sum(return_mdot.get(ce, 0.0) for ce in child_edges)

        for ce in child_edges:
            mdot_child = return_mdot[ce]
            w_child = return_w[ce]

            if total_child_mdot > 1e-12:
                fsr = mdot_child / total_child_mdot
            else:
                fsr = 0.5

            if w_parent > 1e-6:
                ar = _cross_section_area_m2(w_child, depth_mm) / \
                     _cross_section_area_m2(w_parent, depth_mm)
                ar = max(1e-6, min(1.0, ar))
            else:
                ar = 1.0

            K_junc = junction_K_idelchik(theta, ar, fsr, regime)
            dp_minor_return[ce] += minor_loss_pa(mdot_child, w_child, depth_mm,
                                                  K_junc, rho_kg_m3)

            if w_parent > 1e-6:
                dp_minor_return[ce] += _width_change_loss_pa(
                    mdot_child, w_parent, w_child, depth_mm, rho_kg_m3)

    # ------------------------------------------------------------------
    # Port entry/exit losses (applied once per path at inlet and outlet)
    # ------------------------------------------------------------------
    # Find the root edges for supply and return to compute port losses
    supply_root_edges = [(u, v) for (u, v) in dp_edge_supply if u == "root"]
    return_root_edges = [(u, v) for (u, v) in dp_edge_return if u == "root"]

    # Port entry loss: flow enters the supply network at the root
    dp_port_entry = 0.0
    if supply_root_edges:
        # Use the root edge (there is typically one) for port entry K
        re = supply_root_edges[0]
        dp_port_entry = port_entry_loss_pa(
            supply_mdot[re], supply_w[re], depth_mm, rho_kg_m3)

    # Port exit loss: flow leaves the return network at the return root
    dp_port_exit = 0.0
    if return_root_edges:
        re = return_root_edges[0]
        dp_port_exit = port_exit_loss_pa(
            return_mdot[re], return_w[re], depth_mm, rho_kg_m3)

    # ------------------------------------------------------------------
    # Worst-case path: inlet -> leaf -> outlet
    # ------------------------------------------------------------------
    worst_pa = 0.0
    leaves = [n for n in G_supply.nodes
              if G_supply.nodes[n].get("kind") == "leaf"]

    for leaf in leaves:
        # Supply path: root -> leaf
        try:
            path_nodes = nx.shortest_path(G_supply, "root", leaf)
        except nx.NetworkXNoPath:
            continue
        edges_sup = list(zip(path_nodes[:-1], path_nodes[1:]))
        dp_sup = sum(dp_edge_supply.get(e, 0.0) + dp_minor_supply.get(e, 0.0)
                     for e in edges_sup)

        # Return path: root(outlet) -> leaf (graph direction), but flow is
        # leaf -> root; pressure drop magnitude is the same.
        try:
            path_nodes_r = nx.shortest_path(G_return, "root", leaf)
        except nx.NetworkXNoPath:
            continue
        edges_ret = list(zip(path_nodes_r[:-1], path_nodes_r[1:]))
        dp_ret = sum(dp_edge_return.get(e, 0.0) + dp_minor_return.get(e, 0.0)
                     for e in edges_ret)

        dp_total = dp_sup + dp_ret + dp_port_entry + dp_port_exit
        if dp_total > worst_pa:
            worst_pa = dp_total

    dp_kpa = float(worst_pa / 1000.0)
    return (dp_kpa, float(max_Re))
