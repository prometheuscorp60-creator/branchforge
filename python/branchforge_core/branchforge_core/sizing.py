from __future__ import annotations

from typing import Dict, Tuple
import math

WATER_RHO = 997.0  # kg/m3
WATER_MU = 0.00089  # Pa*s
WATER_CP = 4181.0  # J/kg/K

# (rho [kg/m3], mu [Pa*s], cp [J/kg/K]) at ~25 C
_FLUID_PROPS: Dict[str, Tuple[float, float, float]] = {
    "water":                (997.0,  0.00089, 4181.0),
    "ethylene_glycol_30":   (1040.0, 0.00200, 3580.0),
    "ethylene_glycol_50":   (1082.0, 0.00340, 3283.0),
    "ethylene_glycol_70":   (1110.0, 0.00800, 2900.0),
    "propylene_glycol_50":  (1035.0, 0.00500, 3500.0),
    "novec_7100":           (1510.0, 0.00058, 1183.0),
    "pao_2":                (798.0,  0.00520, 2090.0),
}


def get_fluid_properties(coolant: str, **custom) -> Tuple[float, float, float]:
    if coolant == "custom":
        return (
            custom.get("custom_rho_kg_m3", WATER_RHO),
            custom.get("custom_mu_pa_s", WATER_MU),
            custom.get("custom_cp_J_kgK", WATER_CP),
        )
    return _FLUID_PROPS.get(coolant, (WATER_RHO, WATER_MU, WATER_CP))


def get_fluid_cp(coolant: str) -> float:
    return get_fluid_properties(coolant)[2]


def width_from_flow(
    mdot_kg_s: float,
    depth_mm: float,
    v_max_m_s: float,
    min_w_mm: float,
    max_w_mm: float,
    rho_kg_m3: float = WATER_RHO,
) -> float:
    Q = mdot_kg_s / rho_kg_m3
    depth_m = depth_mm / 1000.0
    if depth_m <= 0:
        raise ValueError("channel depth must be > 0")
    A = Q / max(1e-6, v_max_m_s)
    w_m = A / depth_m
    w_mm = w_m * 1000.0
    w_mm = max(min_w_mm, min(max_w_mm, w_mm))
    return float(w_mm)


def murray_law_width(
    parent_width_mm: float,
    depth_mm: float,
    flow_fraction: float,
    min_w_mm: float = 0.3,
    max_w_mm: float = 12.0,
    exponent: float = 3.0,
) -> float:
    """Compute child channel width using Murray's law generalization.

    Murray's law: d_parent^n = d_child1^n + d_child2^n
    For a binary split with flow fraction f going to this child:
        d_child = d_parent * f^(1/n)

    We use hydraulic diameter as the effective 'd', then solve back for width
    given the fixed channel depth.

    Parameters
    ----------
    parent_width_mm : float
        Width of the parent channel.
    depth_mm : float
        Channel depth (fixed for all channels).
    flow_fraction : float
        Fraction of parent mass flow going to this child (0 < f <= 1).
    exponent : float
        Murray's law exponent. Classical = 3.0. Surface-minimization
        networks (Meng et al. 2026) may use values between 2.5-3.0.
    """
    if depth_mm <= 0 or parent_width_mm <= 0:
        return min_w_mm

    f = max(1e-6, min(1.0, float(flow_fraction)))

    # Parent hydraulic diameter
    w_p = parent_width_mm / 1000.0
    d = depth_mm / 1000.0
    Dh_parent = 2.0 * w_p * d / (w_p + d)

    # Murray's law: Dh_child = Dh_parent * f^(1/n)
    Dh_child = Dh_parent * (f ** (1.0 / exponent))

    # Solve for width from Dh = 2*w*d / (w + d)
    # Dh*(w + d) = 2*w*d  =>  Dh*w + Dh*d = 2*w*d  =>  w*(2*d - Dh) = Dh*d
    denom = 2.0 * d - Dh_child
    if denom <= 1e-12:
        return max_w_mm

    w_child = (Dh_child * d) / denom
    w_child_mm = w_child * 1000.0

    return float(max(min_w_mm, min(max_w_mm, w_child_mm)))


def hydraulic_diameter_rect(width_mm: float, depth_mm: float) -> float:
    w = width_mm / 1000.0
    d = depth_mm / 1000.0
    return float(2 * w * d / (w + d))


def flow_velocity_rect(
    mdot_kg_s: float, width_mm: float, depth_mm: float, rho_kg_m3: float = WATER_RHO
) -> float:
    Q = mdot_kg_s / rho_kg_m3
    A = (width_mm / 1000.0) * (depth_mm / 1000.0)
    if A <= 0:
        return 0.0
    return float(Q / A)
