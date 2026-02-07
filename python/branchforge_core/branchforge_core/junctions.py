from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional
import math


# Nature/PMC reported thresholds:
RHO_THRESHOLD = 0.6   # sprouting -> branching transition around rho≈0.6
CHI_TRIFURCATION = 0.83  # trifurcation transition around chi≈0.83


def _wrap_angle(a: float) -> float:
    # wrap to (-pi, pi]
    while a <= -math.pi:
        a += 2*math.pi
    while a > math.pi:
        a -= 2*math.pi
    return a


def angle_of(v: Tuple[float, float]) -> float:
    return math.atan2(v[1], v[0])


def unit(v: Tuple[float, float]) -> Tuple[float, float]:
    x, y = v
    n = math.hypot(x, y)
    if n <= 1e-12:
        return (1.0, 0.0)
    return (x/n, y/n)


def rotate(v: Tuple[float,float], ang: float) -> Tuple[float,float]:
    x, y = v
    c = math.cos(ang)
    s = math.sin(ang)
    return (c*x - s*y, s*x + c*y)


def angular_distance(a: float, b: float) -> float:
    return abs(_wrap_angle(a - b))


def omega_from_theta(theta: float) -> float:
    """Ω(θ) relation reported in the paper (steradians).

    Ω = 4π sin^2((π − θ)/4)
    """
    return 4*math.pi * (math.sin((math.pi - theta)/4.0) ** 2)


def theta_from_omega(omega: float) -> float:
    # invert Ω = 4π sin^2((π − θ)/4)
    omega = max(0.0, min(4*math.pi, omega))
    s2 = omega / (4*math.pi)
    s = math.sqrt(max(0.0, min(1.0, s2)))
    a = math.asin(s)
    theta = math.pi - 4.0*a
    return theta


OMEGA_STEINER = omega_from_theta(2*math.pi/3)  # corresponds to θ=120°


def thick_angle_from_rho(rho: float) -> float:
    """Angle between the two thicker links as a function of thickness ratio ρ.

    Implemented as:
    - ρ <= ρ_th: sprouting => thick links straight => θ = π (180°)
    - ρ > ρ_th: linear Ω(ρ) from 0 to Ω_steiner at ρ=1, then invert to θ.

    This matches the qualitative finding that Ω is ~0 in sprouting regime
    and grows roughly linearly with ρ in branching regime, approaching symmetric
    branching near ρ≈1.
    """
    if rho <= RHO_THRESHOLD:
        return math.pi
    # clamp rho
    rho_c = max(RHO_THRESHOLD, min(1.0, rho))
    t = (rho_c - RHO_THRESHOLD) / (1.0 - RHO_THRESHOLD)
    omega = OMEGA_STEINER * t
    theta = theta_from_omega(omega)
    # ensure within [120°, 180°]
    theta = max(2*math.pi/3, min(math.pi, theta))
    return theta


def slerp_angle(a0: float, a1: float, t: float) -> float:
    # shortest path interpolation
    da = _wrap_angle(a1 - a0)
    return _wrap_angle(a0 + t*da)


@dataclass
class BifurcationDirections:
    parent: Tuple[float, float]
    thick: Tuple[float, float]
    thin: Tuple[float, float]
    regime: str
    rho: float
    theta_thick_deg: float


def compute_bifurcation_directions(
    parent_vec: Tuple[float,float],
    thick_child_vec: Tuple[float,float],
    thin_child_vec: Tuple[float,float],
    rho: float,
) -> BifurcationDirections:
    """Return desired direction unit vectors for a (parent, thick, thin) bifurcation.

    parent_vec: from node -> parent
    thick_child_vec: from node -> thick child
    thin_child_vec: from node -> thin child
    rho: thickness ratio (thin/thick), using perimeters or equivalent.

    Output directions are unit vectors in XY.
    """
    p = unit(parent_vec)
    thick0 = unit(thick_child_vec)
    thin0 = unit(thin_child_vec)

    theta = thick_angle_from_rho(rho)
    regime = "sprout" if rho < RHO_THRESHOLD else "branch"

    # Parent angle
    ap = angle_of(p)
    a_thick0 = angle_of(thick0)

    # candidate thick directions relative to parent: ap ± theta
    cand1 = _wrap_angle(ap + theta)
    cand2 = _wrap_angle(ap - theta)
    # choose closer to original thick direction
    if angular_distance(cand1, a_thick0) <= angular_distance(cand2, a_thick0):
        a_thick = cand1
        a_other = cand2
    else:
        a_thick = cand2
        a_other = cand1

    thick_dir = (math.cos(a_thick), math.sin(a_thick))

    # Thin direction:
    # - In sprout: perpendicular to thick axis (which is roughly opposite of parent direction)
    # - In branch: smoothly morph from perpendicular (at rho_th) to the symmetric other-side branch (at rho=1)
    axis = unit((-p[0], -p[1]))  # straight continuation direction
    a_axis = angle_of(axis)
    # perpendicular candidates
    a_perp1 = _wrap_angle(a_axis + math.pi/2)
    a_perp2 = _wrap_angle(a_axis - math.pi/2)
    a_thin0 = angle_of(thin0)
    a_perp = a_perp1 if angular_distance(a_perp1, a_thin0) <= angular_distance(a_perp2, a_thin0) else a_perp2

    if rho <= RHO_THRESHOLD:
        a_thin = a_perp
    else:
        t = (min(1.0, rho) - RHO_THRESHOLD) / (1.0 - RHO_THRESHOLD)
        # morph toward the symmetric other-side direction
        a_thin = slerp_angle(a_perp, a_other, t)

    thin_dir = (math.cos(a_thin), math.sin(a_thin))

    return BifurcationDirections(
        parent=p,
        thick=unit(thick_dir),
        thin=unit(thin_dir),
        regime=regime,
        rho=float(rho),
        theta_thick_deg=float(theta * 180.0 / math.pi),
    )


def should_merge_to_trifurcation(link_length_mm: float, circumference_mm: float, chi_th: float = CHI_TRIFURCATION) -> bool:
    """Heuristic merge rule inspired by χ transition.

    In the paper, χ controls thickness relative to terminal separation, and a trifurcation
    becomes optimal around χ≈0.83.

    For our engineered networks we use a local proxy:
      χ_local = circumference / link_length
    and merge two nearby bifurcations if χ_local >= χ_th.
    """
    l = max(1e-9, float(link_length_mm))
    w = max(0.0, float(circumference_mm))
    chi_local = w / l
    return chi_local >= chi_th
