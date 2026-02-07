from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

from .schemas import PlateSpec, ConstraintsSpec



def polygon_from_dxf(path: str) -> Polygon:
    import ezdxf
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    # find first closed polyline
    for e in msp:
        dxft = e.dxftype()
        if dxft in ("LWPOLYLINE", "POLYLINE"):
            try:
                pts = []
                if dxft == "LWPOLYLINE":
                    pts = [(float(p[0]), float(p[1])) for p in e.get_points()]
                    closed = bool(e.closed)
                else:
                    pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in e.vertices]
                    closed = bool(e.is_closed)
                if len(pts) >= 3:
                    if not closed:
                        # if not explicitly closed, close it
                        pts = pts + [pts[0]]
                    poly = Polygon(pts)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    return poly
            except Exception:
                continue
    raise ValueError("No closed polyline found in DXF.")


def plate_polygon(plate: PlateSpec) -> Polygon:
    if plate.kind == "rectangle":
        w = float(plate.width_mm)
        h = float(plate.height_mm)
        return box(0.0, 0.0, w, h)
    if plate.kind == "polygon":
        if plate.dxf_path:
            poly = polygon_from_dxf(plate.dxf_path)
        else:
            if not plate.polygon:
                raise ValueError("plate.polygon is required when plate.kind='polygon' and no dxf_path is provided")
            pts = [(p.x_mm, p.y_mm) for p in plate.polygon]
            poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    raise ValueError(f"Unsupported plate.kind={plate.kind}")


def keepouts_geometry(constraints: ConstraintsSpec):
    geoms = []
    for r in constraints.keepout_rects:
        if r.centered:
            x0 = r.x_mm - r.w_mm/2
            y0 = r.y_mm - r.h_mm/2
        else:
            x0 = r.x_mm
            y0 = r.y_mm
        geom = box(x0, y0, x0 + r.w_mm, y0 + r.h_mm)
        margin = float(r.margin_mm)
        if margin:
            geom = geom.buffer(margin)
        geoms.append(geom)

    for c in constraints.keepout_circles:
        geom = Point(float(c.x_mm), float(c.y_mm)).buffer(float(c.r_mm) + float(c.margin_mm))
        geoms.append(geom)

    if not geoms:
        return None
    return unary_union(geoms)


def allowed_region(poly: Polygon, constraints: ConstraintsSpec, clearance_mm: float) -> Polygon:
    # clearance_mm should include min_wall + half-width etc.
    # Shrink boundary by clearance, then subtract keepouts expanded by clearance.
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


def clamp_point_to_allowed(pt: Tuple[float, float], allowed: Polygon, jitter_mm: float = 2.0, tries: int = 50) -> Tuple[float, float]:
    x, y = pt
    if allowed.contains(Point(x, y)):
        return (x, y)
    # Jitter search
    for i in range(tries):
        ang = (2*math.pi) * (i / max(1, tries-1))
        rr = jitter_mm * (0.2 + 0.8*(i/tries))
        cand = (x + rr*math.cos(ang), y + rr*math.sin(ang))
        if allowed.contains(Point(*cand)):
            return cand
    # As a last resort: use nearest point on allowed boundary
    try:
        nearest = allowed.exterior.interpolate(allowed.exterior.project(Point(x, y)))
        return (nearest.x, nearest.y)
    except Exception:
        return (x, y)
