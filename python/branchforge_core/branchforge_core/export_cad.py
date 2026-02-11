from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import math

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

from .schemas import ConstraintsSpec, PortSpec


def channel_footprint_polygon(
    edge_paths: Dict[Tuple[str, str], List[Tuple[float, float]]],
    edge_width_mm: Dict[Tuple[str, str], float],
    cap_style: int = 1,
    join_style: int = 1,
) -> Polygon | MultiPolygon:
    geoms = []
    for e, pts in edge_paths.items():
        if len(pts) < 2:
            continue
        w = float(edge_width_mm.get(e, 1.0))
        r = max(0.1, w / 2.0)
        ls = LineString(pts)
        geoms.append(ls.buffer(r, cap_style=cap_style, join_style=join_style))
    if not geoms:
        return Polygon()
    u = unary_union(geoms)
    return u


def total_channel_length_mm(
    edge_paths: Dict[Tuple[str, str], List[Tuple[float, float]]],
) -> float:
    total = 0.0
    for pts in edge_paths.values():
        if len(pts) < 2:
            continue
        for a, b in zip(pts[:-1], pts[1:]):
            total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def channel_area_mm2(fp: Polygon | MultiPolygon) -> float:
    if fp.is_empty:
        return 0.0
    return float(fp.area)


def export_dxf_centerlines(
    out_path: str,
    outline: Polygon,
    edge_paths: Dict[Tuple[str, str], List[Tuple[float, float]]],
    edge_width_mm: Dict[Tuple[str, str], float],
    ports: Optional[PortSpec] = None,
):
    import ezdxf
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.units = ezdxf.units.MM

    doc.layers.new("OUTLINE", dxfattribs={"color": 7})
    doc.layers.new("CENTERLINES", dxfattribs={"color": 3})
    doc.layers.new("PORTS", dxfattribs={"color": 1})
    doc.layers.new("CHANNEL_BOUNDS", dxfattribs={"color": 5})

    ext = list(outline.exterior.coords)
    msp.add_lwpolyline(ext, dxfattribs={"layer": "OUTLINE", "closed": True})

    for e, pts in edge_paths.items():
        if len(pts) < 2:
            continue
        w = float(edge_width_mm.get(e, 1.0))
        msp.add_lwpolyline(
            pts,
            dxfattribs={"layer": "CENTERLINES", "closed": False, "constant_width": w},
        )

    # Channel boundary outlines
    for e, pts in edge_paths.items():
        if len(pts) < 2:
            continue
        w = float(edge_width_mm.get(e, 1.0))
        ls = LineString(pts)
        buffered = ls.buffer(w / 2.0, cap_style=1, join_style=1)
        if isinstance(buffered, Polygon):
            polys = [buffered]
        elif isinstance(buffered, MultiPolygon):
            polys = list(buffered.geoms)
        else:
            continue
        for poly in polys:
            coords = list(poly.exterior.coords)
            msp.add_lwpolyline(
                coords,
                dxfattribs={"layer": "CHANNEL_BOUNDS", "closed": True},
            )

    if ports is not None:
        inlet_xy = (ports.inlet.x_mm, ports.inlet.y_mm)
        outlet_xy = (ports.outlet.x_mm, ports.outlet.y_mm)
        r_in = ports.inlet_diameter_mm / 2.0
        r_out = ports.outlet_diameter_mm / 2.0
        msp.add_circle(inlet_xy, r_in, dxfattribs={"layer": "PORTS"})
        msp.add_circle(outlet_xy, r_out, dxfattribs={"layer": "PORTS"})
        msp.add_text(
            "INLET",
            dxfattribs={"layer": "PORTS", "height": 2.0},
        ).set_placement(
            (inlet_xy[0] + r_in + 1, inlet_xy[1]),
        )
        msp.add_text(
            "OUTLET",
            dxfattribs={"layer": "PORTS", "height": 2.0},
        ).set_placement(
            (outlet_xy[0] + r_out + 1, outlet_xy[1]),
        )

    doc.saveas(out_path)


def _port_bore_solid(cq_module, xy: Tuple[float, float], diameter_mm: float, plate_thickness_mm: float):
    r = diameter_mm / 2.0
    bore = (
        cq_module.Workplane("XY")
        .workplane(offset=0)
        .center(xy[0], xy[1])
        .circle(r)
        .extrude(plate_thickness_mm)
    )
    return bore


def _manifold_pocket(
    cq_module,
    port_xy: Tuple[float, float],
    port_diameter_mm: float,
    first_channel_width_mm: float,
    plate_thickness_mm: float,
    channel_depth_mm: float,
):
    pocket_w = max(port_diameter_mm, first_channel_width_mm * 2.0)
    pocket_l = pocket_w * 0.8
    pocket_depth = channel_depth_mm

    pocket = (
        cq_module.Workplane("XY")
        .workplane(offset=plate_thickness_mm)
        .center(port_xy[0], port_xy[1])
        .rect(pocket_w, pocket_l)
        .extrude(-pocket_depth)
    )
    return pocket


def export_cad_solids(
    plate_outline: Polygon,
    channels_fp: Polygon | MultiPolygon,
    constraints: ConstraintsSpec,
    out_plate_step: str,
    out_channels_step: str,
    out_plate_stl: str,
    ports: Optional[PortSpec] = None,
):
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
        solid = wp.extrude(-ch_depth)
        for hole in poly.interiors:
            hp = list(hole.coords)
            hsolid = (
                cq.Workplane("XY")
                .workplane(offset=plate_th)
                .polyline(hp)
                .close()
                .extrude(-ch_depth)
            )
            solid = solid.cut(hsolid)
        return solid

    if channels_fp.is_empty:
        channels_solid = (
            cq.Workplane("XY").box(0.1, 0.1, 0.1).translate((0, 0, -1000))
        )
    else:
        if isinstance(channels_fp, Polygon):
            channels_solid = extrude_poly(channels_fp)
        else:
            solids = [extrude_poly(p) for p in channels_fp.geoms]
            channels_solid = solids[0]
            for s in solids[1:]:
                channels_solid = channels_solid.union(s)

    # Cut plate by channels
    plate_cut = plate.cut(channels_solid)

    # Port bores (through-holes for fittings)
    if ports is not None:
        inlet_xy = (ports.inlet.x_mm, ports.inlet.y_mm)
        outlet_xy = (ports.outlet.x_mm, ports.outlet.y_mm)

        inlet_bore = _port_bore_solid(cq, inlet_xy, ports.inlet_diameter_mm, plate_th)
        outlet_bore = _port_bore_solid(cq, outlet_xy, ports.outlet_diameter_mm, plate_th)
        plate_cut = plate_cut.cut(inlet_bore)
        plate_cut = plate_cut.cut(outlet_bore)

        # Manifold pockets
        inlet_pocket = _manifold_pocket(
            cq,
            inlet_xy,
            ports.inlet_diameter_mm,
            constraints.max_channel_width_mm,
            plate_th,
            ch_depth,
        )
        outlet_pocket = _manifold_pocket(
            cq,
            outlet_xy,
            ports.outlet_diameter_mm,
            constraints.max_channel_width_mm,
            plate_th,
            ch_depth,
        )
        plate_cut = plate_cut.cut(inlet_pocket)
        plate_cut = plate_cut.cut(outlet_pocket)

    # Keepout bore holes (bolt holes)
    for ko_circle in constraints.keepout_circles:
        bore = _port_bore_solid(
            cq,
            (ko_circle.x_mm, ko_circle.y_mm),
            ko_circle.r_mm * 2.0,
            plate_th,
        )
        plate_cut = plate_cut.cut(bore)

    # Export
    cq.exporters.export(plate_cut, out_plate_step)
    cq.exporters.export(channels_solid, out_channels_step)
    cq.exporters.export(plate_cut, out_plate_stl)
