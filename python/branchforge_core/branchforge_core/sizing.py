from __future__ import annotations

from typing import Dict, Tuple
import math

WATER_RHO = 997.0  # kg/m3
WATER_MU = 0.00089  # Pa*s
WATER_CP = 4181.0  # J/kg/K


def width_from_flow(
    mdot_kg_s: float,
    depth_mm: float,
    v_max_m_s: float,
    min_w_mm: float,
    max_w_mm: float,
    rho_kg_m3: float = WATER_RHO,
) -> float:
    # area = Q / v_max
    Q = mdot_kg_s / rho_kg_m3  # m3/s
    depth_m = depth_mm / 1000.0
    if depth_m <= 0:
        raise ValueError("channel depth must be > 0")
    A = Q / max(1e-6, v_max_m_s)  # m2
    w_m = A / depth_m
    w_mm = w_m * 1000.0
    w_mm = max(min_w_mm, min(max_w_mm, w_mm))
    return float(w_mm)


def hydraulic_diameter_rect(width_mm: float, depth_mm: float) -> float:
    w = width_mm / 1000.0
    d = depth_mm / 1000.0
    return float(2*w*d/(w+d))  # meters


def flow_velocity_rect(mdot_kg_s: float, width_mm: float, depth_mm: float, rho_kg_m3: float = WATER_RHO) -> float:
    Q = mdot_kg_s / rho_kg_m3
    A = (width_mm/1000.0) * (depth_mm/1000.0)
    if A <= 0:
        return 0.0
    return float(Q / A)
