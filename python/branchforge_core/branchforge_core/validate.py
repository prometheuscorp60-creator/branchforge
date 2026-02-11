from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import math

from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union

from .schemas import ConstraintsSpec


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    location: Optional[str] = None


@dataclass
class ValidationResult:
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    channels_clipped: Optional[Polygon | MultiPolygon] = None

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        e = len(self.errors)
        w = len(self.warnings)
        return f"{'PASS' if self.valid else 'FAIL'}: {e} error(s), {w} warning(s)"


def validate_candidate(
    outline: Polygon,
    channels_fp: Polygon | MultiPolygon,
    constraints: ConstraintsSpec,
    edge_paths: Dict[Tuple[str, str], List[Tuple[float, float]]],
    edge_widths: Dict[Tuple[str, str], float],
) -> ValidationResult:
    issues: List[ValidationIssue] = []
    clipped = channels_fp

    # 1. Channel footprint within outline
    if not channels_fp.is_empty:
        if not outline.contains(channels_fp):
            overflow = channels_fp.difference(outline)
            if not overflow.is_empty and overflow.area > 0.01:
                issues.append(ValidationIssue(
                    severity="warning",
                    code="CHANNELS_OUTSIDE_OUTLINE",
                    message=f"Channel footprint extends {overflow.area:.2f} mm² outside plate outline. Clipping.",
                ))
                clipped = channels_fp.intersection(outline)
                if not clipped.is_valid:
                    clipped = clipped.buffer(0)

    # 2. Shapely validity
    if not channels_fp.is_valid:
        issues.append(ValidationIssue(
            severity="warning",
            code="INVALID_POLYGON",
            message="Channel footprint polygon is invalid (self-intersections). Attempting buffer(0) fix.",
        ))

    # 3. Minimum wall thickness check
    min_wall = float(constraints.min_wall_mm)
    if min_wall > 0 and not channels_fp.is_empty:
        shrunk_outline = outline.buffer(-min_wall)
        if not shrunk_outline.is_empty:
            if not shrunk_outline.contains(channels_fp):
                issues.append(ValidationIssue(
                    severity="warning",
                    code="WALL_THICKNESS_VIOLATION",
                    message=f"Some channels are within {min_wall:.1f} mm of plate boundary.",
                ))

    # 4. Bend radius check (proxy: angle changes in routed paths)
    min_bend_r = float(constraints.min_bend_radius_mm)
    if min_bend_r > 0:
        sharp_bends = 0
        for e, path in edge_paths.items():
            if len(path) < 3:
                continue
            w = float(edge_widths.get(e, 1.0))
            for i in range(1, len(path) - 1):
                a, b, c = path[i - 1], path[i], path[i + 1]
                v1 = (b[0] - a[0], b[1] - a[1])
                v2 = (c[0] - b[0], c[1] - b[1])
                n1 = math.hypot(v1[0], v1[1])
                n2 = math.hypot(v2[0], v2[1])
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                cos_ang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
                cos_ang = max(-1.0, min(1.0, cos_ang))
                ang = math.acos(cos_ang)
                if ang > 0.1:  # non-trivial bend
                    seg_len = min(n1, n2)
                    local_r = seg_len / (2.0 * math.sin(ang / 2.0)) if ang > 1e-6 else float("inf")
                    if local_r < min_bend_r:
                        sharp_bends += 1
        if sharp_bends > 0:
            issues.append(ValidationIssue(
                severity="warning",
                code="SHARP_BENDS",
                message=f"{sharp_bends} bend(s) violate min bend radius ({min_bend_r:.1f} mm).",
            ))

    # 5. Channel width bounds
    for e, w in edge_widths.items():
        if w < constraints.min_channel_width_mm - 0.01:
            issues.append(ValidationIssue(
                severity="warning",
                code="CHANNEL_TOO_NARROW",
                message=f"Edge {e}: width {w:.2f} mm < min {constraints.min_channel_width_mm:.2f} mm",
                location=str(e),
            ))
        if w > constraints.max_channel_width_mm + 0.01:
            issues.append(ValidationIssue(
                severity="warning",
                code="CHANNEL_TOO_WIDE",
                message=f"Edge {e}: width {w:.2f} mm > max {constraints.max_channel_width_mm:.2f} mm",
                location=str(e),
            ))

    # 6. Keepout clearance
    from .geometry import keepouts_geometry
    ko = keepouts_geometry(constraints)
    if ko is not None and not channels_fp.is_empty:
        ko_margin = ko.buffer(min_wall)
        if channels_fp.intersects(ko_margin):
            overlap = channels_fp.intersection(ko_margin)
            if overlap.area > 0.01:
                issues.append(ValidationIssue(
                    severity="error",
                    code="KEEPOUT_VIOLATION",
                    message=f"Channels overlap keepout zone by {overlap.area:.2f} mm².",
                ))

    has_errors = any(i.severity == "error" for i in issues)
    return ValidationResult(
        valid=not has_errors,
        issues=issues,
        channels_clipped=clipped,
    )
