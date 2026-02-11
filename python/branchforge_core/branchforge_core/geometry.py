from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math

from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union, linemerge, polygonize

from .schemas import PlateSpec, ConstraintsSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DXF unit codes ($INSUNITS) -> scale factor to millimetres
# Reference: AutoCAD DXF docs, header variable $INSUNITS
# ---------------------------------------------------------------------------
_INSUNITS_TO_MM: Dict[int, float] = {
    0: 1.0,       # Unitless -- assume mm
    1: 25.4,      # Inches
    2: 304.8,     # Feet
    3: 1609344.0, # Miles
    4: 1.0,       # Millimetres
    5: 10.0,      # Centimetres
    6: 1000.0,    # Metres
    7: 1e6,       # Kilometres
    8: 0.0000254, # Microinches
    9: 0.001,     # Mils (thousandths of an inch)
    10: 914.4,    # Yards
    11: 1e-7,     # Angstroms
    12: 1e-6,     # Nanometres
    13: 0.001,    # Microns (micrometres)
    14: 100.0,    # Decimetres
    15: 10000.0,  # Decametres
    16: 100000.0, # Hectometres
    17: 1e9,      # Gigametres
    18: 1.496e14, # Astronomical units
    19: 9.461e18, # Light years
    20: 3.086e22, # Parsecs
}


def detect_dxf_units(path: str) -> float:
    """Read the $INSUNITS header variable from a DXF file and return a scale
    factor that converts drawing units to millimetres.

    If the header variable is missing or unrecognised the function returns 1.0
    (i.e. assume the file is already in mm).
    """
    import ezdxf

    try:
        doc = ezdxf.readfile(path)
    except Exception as exc:
        logger.warning("detect_dxf_units: failed to read '%s': %s", path, exc)
        return 1.0

    try:
        insunits = doc.header.get("$INSUNITS", 0)
    except Exception:
        insunits = 0

    scale = _INSUNITS_TO_MM.get(int(insunits), 1.0)
    if insunits not in _INSUNITS_TO_MM:
        logger.warning(
            "detect_dxf_units: unknown $INSUNITS value %s in '%s'; assuming mm",
            insunits,
            path,
        )
    else:
        logger.debug(
            "detect_dxf_units: $INSUNITS=%s -> scale=%.6g mm for '%s'",
            insunits,
            scale,
            path,
        )
    return scale


# ---------------------------------------------------------------------------
# Internal helpers for building polylines from LINE entities
# ---------------------------------------------------------------------------

def _endpoint_key(x: float, y: float, tol: float = 1e-6) -> Tuple[int, int]:
    """Quantise a 2-D point so that nearby endpoints hash to the same key."""
    return (round(x / tol), round(y / tol))


def _lines_to_closed_polygons(
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    tol: float = 1e-6,
) -> List[Polygon]:
    """Given a list of (start, end) line segments try to merge them into
    closed rings and return valid Shapely Polygons.
    """
    if not lines:
        return []

    shapely_lines = [LineString([s, e]) for s, e in lines]
    merged = linemerge(shapely_lines)
    polygons: List[Polygon] = []

    # polygonize will produce polygons from any closed rings it finds
    for poly in polygonize(merged):
        if poly.is_valid and not poly.is_empty and poly.area > 0:
            polygons.append(poly)

    return polygons


# ---------------------------------------------------------------------------
# Main DXF -> Polygon function
# ---------------------------------------------------------------------------

def polygon_from_dxf(path: str, scale_factor: float = 1.0) -> Polygon:
    """Import the largest closed polyline from a DXF file.

    Improvements over the original implementation:
    * Collects **all** LWPOLYLINE / POLYLINE entities (not just the first).
    * Also collects LINE entities and attempts to merge them into closed
      polygons.
    * Picks the candidate with the largest area.
    * Applies an optional *scale_factor* (default 1.0) to every coordinate.
      Use :func:`detect_dxf_units` to obtain a suitable factor.

    Parameters
    ----------
    path : str
        Filesystem path to a ``.dxf`` file.
    scale_factor : float, optional
        Multiplicative factor applied to every (x, y) coordinate.  A value of
        ``25.4`` would convert inches to millimetres, for example.

    Returns
    -------
    Polygon
        A valid Shapely Polygon representing the largest closed outline found.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If no usable closed polyline could be extracted.
    """
    import ezdxf

    try:
        doc = ezdxf.readfile(path)
    except FileNotFoundError:
        raise
    except ezdxf.DXFError as exc:
        raise ValueError(f"Failed to parse DXF file '{path}': {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Unexpected error reading DXF file '{path}': {exc}") from exc

    msp = doc.modelspace()

    candidates: List[Polygon] = []
    line_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    for entity in msp:
        dxft = entity.dxftype()

        # --- LWPOLYLINE / POLYLINE -------------------------------------------
        if dxft in ("LWPOLYLINE", "POLYLINE"):
            try:
                pts: List[Tuple[float, float]] = []
                if dxft == "LWPOLYLINE":
                    pts = [
                        (float(p[0]) * scale_factor, float(p[1]) * scale_factor)
                        for p in entity.get_points()
                    ]
                    closed = bool(entity.closed)
                else:
                    pts = [
                        (
                            float(v.dxf.location.x) * scale_factor,
                            float(v.dxf.location.y) * scale_factor,
                        )
                        for v in entity.vertices
                    ]
                    closed = bool(entity.is_closed)

                if len(pts) < 3:
                    logger.debug("Skipping polyline with < 3 vertices in '%s'", path)
                    continue

                if not closed:
                    # Close the ring so Shapely treats it as a polygon
                    pts = pts + [pts[0]]

                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty and poly.area > 0:
                    candidates.append(poly)
                else:
                    logger.debug(
                        "Discarding degenerate polyline (area=%.4g) in '%s'",
                        poly.area,
                        path,
                    )
            except Exception as exc:
                logger.warning(
                    "Error processing %s entity in '%s': %s", dxft, path, exc
                )
                continue

        # --- LINE entities ----------------------------------------------------
        elif dxft == "LINE":
            try:
                start = (
                    float(entity.dxf.start.x) * scale_factor,
                    float(entity.dxf.start.y) * scale_factor,
                )
                end = (
                    float(entity.dxf.end.x) * scale_factor,
                    float(entity.dxf.end.y) * scale_factor,
                )
                line_segments.append((start, end))
            except Exception as exc:
                logger.warning("Error processing LINE entity in '%s': %s", path, exc)
                continue

    # --- Attempt to form polygons from collected LINE segments ----------------
    if line_segments:
        try:
            line_polys = _lines_to_closed_polygons(line_segments)
            for lp in line_polys:
                if lp.is_valid and not lp.is_empty and lp.area > 0:
                    candidates.append(lp)
                    logger.debug(
                        "Formed polygon (area=%.4g) from LINE segments in '%s'",
                        lp.area,
                        path,
                    )
        except Exception as exc:
            logger.warning(
                "Failed to merge LINE segments into polygons in '%s': %s", path, exc
            )

    # --- Pick the largest candidate by area ----------------------------------
    if not candidates:
        raise ValueError(
            f"No closed polyline or mergeable LINE loop found in DXF file '{path}'."
        )

    best = max(candidates, key=lambda p: p.area)
    logger.info(
        "polygon_from_dxf: selected polygon with area=%.4g mm^2 out of %d candidates "
        "from '%s'",
        best.area,
        len(candidates),
        path,
    )

    if not best.is_valid:
        best = best.buffer(0)

    return best


# ---------------------------------------------------------------------------
# Plate / keepout / region helpers  (unchanged API, enhanced DXF path)
# ---------------------------------------------------------------------------

def plate_polygon(plate: PlateSpec) -> Polygon:
    """Return a Shapely Polygon for the plate outline described by *plate*.

    When ``plate.kind == "polygon"`` and ``plate.dxf_path`` is provided the
    function now also attempts automatic unit detection so that DXF files saved
    in inches, metres, etc. are correctly converted to millimetres.
    """
    if plate.kind == "rectangle":
        w = float(plate.width_mm)
        h = float(plate.height_mm)
        return box(0.0, 0.0, w, h)

    if plate.kind == "polygon":
        if plate.dxf_path:
            # Auto-detect units and derive a scale factor
            scale = detect_dxf_units(plate.dxf_path)
            poly = polygon_from_dxf(plate.dxf_path, scale_factor=scale)
        else:
            if not plate.polygon:
                raise ValueError(
                    "plate.polygon is required when plate.kind='polygon' "
                    "and no dxf_path is provided"
                )
            pts = [(p.x_mm, p.y_mm) for p in plate.polygon]
            poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly

    raise ValueError(f"Unsupported plate.kind={plate.kind}")


def keepouts_geometry(constraints: ConstraintsSpec):
    """Build a single Shapely geometry representing all keepout zones."""
    geoms = []
    for r in constraints.keepout_rects:
        if r.centered:
            x0 = r.x_mm - r.w_mm / 2
            y0 = r.y_mm - r.h_mm / 2
        else:
            x0 = r.x_mm
            y0 = r.y_mm
        geom = box(x0, y0, x0 + r.w_mm, y0 + r.h_mm)
        margin = float(r.margin_mm)
        if margin:
            geom = geom.buffer(margin)
        geoms.append(geom)

    for c in constraints.keepout_circles:
        geom = Point(float(c.x_mm), float(c.y_mm)).buffer(
            float(c.r_mm) + float(c.margin_mm)
        )
        geoms.append(geom)

    if not geoms:
        return None
    return unary_union(geoms)


def allowed_region(
    poly: Polygon, constraints: ConstraintsSpec, clearance_mm: float
) -> Polygon:
    """Shrink *poly* by *clearance_mm* and subtract keepout zones."""
    shrunk = poly.buffer(-clearance_mm)
    if shrunk.is_empty:
        # Fall back: no shrinking (avoid failure); caller should score low.
        shrunk = poly
    ko = keepouts_geometry(constraints)
    if ko is not None:
        shrunk = shrunk.difference(ko.buffer(clearance_mm))
    if not shrunk.is_valid:
        shrunk = shrunk.buffer(0)
    return shrunk


def clamp_point_to_allowed(
    pt: Tuple[float, float],
    allowed: Polygon,
    jitter_mm: float = 2.0,
    tries: int = 50,
) -> Tuple[float, float]:
    """Return a point inside *allowed*, starting from *pt* and jittering if
    necessary.  Falls back to the nearest boundary point as a last resort.
    """
    x, y = pt
    if allowed.contains(Point(x, y)):
        return (x, y)

    # Jitter search in an expanding spiral
    for i in range(tries):
        ang = (2 * math.pi) * (i / max(1, tries - 1))
        rr = jitter_mm * (0.2 + 0.8 * (i / tries))
        cand = (x + rr * math.cos(ang), y + rr * math.sin(ang))
        if allowed.contains(Point(*cand)):
            return cand

    # Last resort: nearest point on the allowed boundary
    try:
        nearest = allowed.exterior.interpolate(
            allowed.exterior.project(Point(x, y))
        )
        return (nearest.x, nearest.y)
    except Exception:
        return (x, y)
