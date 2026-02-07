from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import math

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

from .schemas import ConstraintsSpec


def channel_footprint_polygon(
    edge_paths: Dict[Tuple[str,str], List[Tuple[float,float]]],
    edge_width_mm: Dict[Tuple[str,str], float],
    cap_style: int = 1,
    join_style: int = 1,
) -> Polygon | MultiPolygon:
    """Create a 2D channel footprint as union(LineString(path).buffer(width/2))."""
    geoms = []
    for e, pts in edge_paths.items():
        if len(pts) < 2:
            continue
        w = float(edge_width_mm.get(e, 1.0))
        r = max(0.1, w/2.0)
        ls = LineString(pts)
        geoms.append(ls.buffer(r, cap_style=cap_style, join_style=join_style))
    if not geoms:
        return Polygon()
    u = unary_union(geoms)
    return u


def export_dxf_centerlines(
    out_path: str,
    outline: Polygon,
    edge_paths: Dict[Tuple[str,str], List[Tuple[float,float]]],
    edge_width_mm: Dict[Tuple[str,str], float],
):
    import ezdxf
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.units = ezdxf.units.MM

    # Outline layer
    doc.layers.new("OUTLINE", dxfattribs={"color": 7})
    doc.layers.new("CENTERLINES", dxfattribs={"color": 3})

    ext = list(outline.exterior.coords)
    msp.add_lwpolyline(ext, dxfattribs={"layer": "OUTLINE", "closed": True})

    for e, pts in edge_paths.items():
        if len(pts) < 2:
            continue
        w = float(edge_width_mm.get(e, 1.0))
        msp.add_lwpolyline(pts, dxfattribs={"layer": "CENTERLINES", "closed": False, "constant_width": w})

    doc.saveas(out_path)


def export_cad_solids(
    plate_outline: Polygon,
    channels_fp: Polygon | MultiPolygon,
    constraints: ConstraintsSpec,
    out_plate_step: str,
    out_channels_step: str,
    out_plate_stl: str,
):
    """Export CAD solids using CadQuery.

    Plate is extruded +Z from z=0.
    Channel void is extruded -Z from the *top surface* (z=plate_thickness), depth=channel_depth.
    """
    try:
        import cadquery as cq
    except Exception as e:
        raise RuntimeError(
            "CadQuery not installed. Use the Docker setup or install branchforge-core[cad]."
        ) from e

    plate_th = float(constraints.plate_thickness_mm)
    ch_depth = float(constraints.channel_depth_mm)

    # Plate solid
    outline_pts = list(plate_outline.exterior.coords)
    plate = cq.Workplane("XY").polyline(outline_pts).close().extrude(plate_th)

    # Channels void solid(s)
    def extrude_poly(poly: Polygon):
        ext = list(poly.exterior.coords)
        wp = cq.Workplane("XY").workplane(offset=plate_th).polyline(ext).close()
        solid = wp.extrude(-ch_depth)  # cut down from top
        # holes
        for hole in poly.interiors:
            hp = list(hole.coords)
            hsolid = cq.Workplane("XY").workplane(offset=plate_th).polyline(hp).close().extrude(-ch_depth)
            solid = solid.cut(hsolid)
        return solid

    if channels_fp.is_empty:
        channels_solid = cq.Workplane("XY").box(0.1, 0.1, 0.1).translate((0,0,-1000))  # far away dummy
    else:
        if isinstance(channels_fp, Polygon):
            channels_solid = extrude_poly(channels_fp)
        else:
            solids = [extrude_poly(p) for p in channels_fp.geoms]
            channels_solid = solids[0]
            for s in solids[1:]:
                channels_solid = channels_solid.union(s)

    # Cut plate
    plate_cut = plate.cut(channels_solid)

    # Export
    cq.exporters.export(plate_cut, out_plate_step)
    cq.exporters.export(channels_solid, out_channels_step)
    cq.exporters.export(plate_cut, out_plate_stl)
