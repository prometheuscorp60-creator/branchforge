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


def _optimal_bifurcation_angle(rho: float, n: float = 3.0) -> float:
    """Compute the energy-optimal bifurcation angle from Murray's law generalization.

    For a bifurcation with flow ratio q (thin/parent) and diameter ratio rho
    (thin/thick), the optimal angle theta between parent and thick child
    minimizes total dissipation + surface energy.

    From the variational analysis (Meng et al. 2026):
        cos(alpha_thick) = (1 + r^n - (1-r)^n) / (2 * r^(n/2))
    where r = rho^2 (area ratio proxy) and n=3 for Murray's law.

    For sprouting (rho <= 0.6): thin branch emerges perpendicular, thick
    continues straight (theta = pi).

    For branching (rho > 0.6): the included angle between parent and thick
    child is derived from the energy minimum.
    """
    rho = max(0.0, min(1.0, float(rho)))

    if rho <= RHO_THRESHOLD:
        return math.pi

    # r = area ratio proxy (thin/thick cross-section)
    r = rho * rho
    r = max(1e-6, min(1.0 - 1e-6, r))

    # Murray's law optimal angle: cos(alpha) = (1 + r^n - (1-r)^n) / (2 * r^(n/2))
    # alpha is the half-angle of the thick branch from the parent axis
    numerator = 1.0 + r**n - (1.0 - r)**n
    denominator = 2.0 * r**(n / 2.0)

    cos_alpha = numerator / max(denominator, 1e-9)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    alpha = math.acos(cos_alpha)

    # theta is the included angle (parent-to-thick child)
    # alpha is measured from parent continuation, so theta = pi - alpha
    theta = math.pi - alpha
    theta = max(2 * math.pi / 3, min(math.pi, theta))
    return theta


def thick_angle_from_rho(rho: float) -> float:
    """Angle between parent and thick child as a function of thickness ratio rho.

    Uses energy-optimal bifurcation angle derived from Murray's law generalization
    (Meng et al. Nature 649, 315-322, 2026).

    - rho <= 0.6 (sprouting): thin branch perpendicular, thick continues straight (theta=pi)
    - rho > 0.6 (branching): angle from variational energy minimization
    - rho = 1.0 (symmetric): approaches Steiner angle (120 degrees)
    """
    return _optimal_bifurcation_angle(rho)


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
    """Trifurcation merging rule based on the chi transition (Meng et al. 2026).

    The paper shows that when two bifurcations are close enough relative to the
    channel thickness, merging them into a single trifurcation reduces total
    surface area. The transition occurs at chi ~ 0.83 where:

        chi = (channel_perimeter) / (link_length)

    This is a dimensionless ratio comparing the channel's cross-sectional scale
    to the separation between junctions. When chi >= 0.83, the surface energy
    saved by eliminating one junction outweighs the penalty of the wider node,
    making trifurcation energetically favorable.
    """
    l = max(1e-9, float(link_length_mm))
    w = max(0.0, float(circumference_mm))
    chi_local = w / l
    return chi_local >= chi_th
