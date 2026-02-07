from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import os
import json
import math
import random

import numpy as np
import networkx as nx
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

from .schemas import JobSpec, CandidateSummary, CandidateMetrics, CandidateArtifacts
from .geometry import plate_polygon, allowed_region, clamp_point_to_allowed
from .heatmap import load_csv_grid, load_image_grid, normalize_to_total, grid_cell_centers
from .clustering import weighted_kmeans, sample_weighted
from .tree import build_supply_tree, build_return_tree, compute_edge_mass_flows, node_leaf_mass_flow
from .sizing import width_from_flow, WATER_CP
from .routing import GridRouter, rdp_simplify
from .junctions import compute_bifurcation_directions, should_merge_to_trifurcation
from .hydraulics import estimate_network_dp_kpa
from .scoring import assign_heat_to_leaves, leaf_deltaT, manufacturing_score
from .export_cad import channel_footprint_polygon, export_cad_solids, export_dxf_centerlines
from .reporting import render_preview_png, render_report_pdf


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _plate_dims_mm_from_outline(outline: Polygon, plate_spec_kind: str, plate_w: float, plate_h: float) -> Tuple[float,float]:
    if plate_spec_kind == "rectangle":
        return (float(plate_w), float(plate_h))
    minx, miny, maxx, maxy = outline.bounds
    return (float(maxx-minx), float(maxy-miny))


def _leaf_points_from_heatmap(
    outline: Polygon,
    allowed: Polygon,
    plate_w_mm: float,
    plate_h_mm: float,
    heat_grid: np.ndarray,
    n_leaves: int,
    seed: int,
) -> List[Tuple[float,float]]:
    pts = grid_cell_centers(plate_w_mm, plate_h_mm, heat_grid).reshape(-1, 2)
    wts = heat_grid.reshape(-1).clip(min=0.0)

    # sample to keep kmeans fast
    pts_s, wts_s = sample_weighted(pts, wts, max_samples=2500, seed=seed)

    centers = weighted_kmeans(pts_s, wts_s, k=n_leaves, n_iter=40, seed=seed)
    out = []
    for c in centers:
        xy = (float(c[0]), float(c[1]))
        xy = clamp_point_to_allowed(xy, allowed)
        out.append(xy)
    return out


def _merge_trifurcations_in_tree(
    G: nx.DiGraph,
    edge_width_mm: Dict[Tuple[str,str], float],
    depth_mm: float,
):
    # One pass: merge internal child v into internal parent u if edge is tiny relative to perimeter
    to_merge = []
    for u, v in list(G.edges):
        if G.nodes[u].get("kind") not in ("internal", "root"):
            continue
        if G.nodes[v].get("kind") != "internal":
            continue
        # length
        xu, yu = G.nodes[u]["xy"]
        xv, yv = G.nodes[v]["xy"]
        l = math.hypot(xv-xu, yv-yu)
        w = float(edge_width_mm.get((u,v), 1.0))
        # perimeter proxy for circumference constraint:
        circ = 2.0 * (w + depth_mm)
        if should_merge_to_trifurcation(l, circ):
            to_merge.append((u, v))

    # Perform merges (careful to not double-merge)
    merged = set()
    for u, v in to_merge:
        if v in merged or u in merged:
            continue
        if not G.has_edge(u, v):
            continue
        # move u toward v
        xu, yu = G.nodes[u]["xy"]
        xv, yv = G.nodes[v]["xy"]
        G.nodes[u]["xy"] = ((xu+xv)/2.0, (yu+yv)/2.0)

        # rewire v's children to u
        children = list(G.successors(v))
        for c in children:
            if c == u:
                continue
            G.add_edge(u, c)
        # remove edge u->v
        if G.has_edge(u, v):
            G.remove_edge(u, v)
        # remove v if isolated
        try:
            if G.in_degree(v) == 0 and G.out_degree(v) == 0:
                G.remove_node(v)
            else:
                # if still has incoming edges, do not remove (shouldn't happen in tree)
                pass
        except Exception:
            pass
        merged.add(v)


def _route_tree_edges(
    G: nx.DiGraph,
    router: GridRouter,
    edge_width_mm: Dict[Tuple[str,str], float],
    min_wall_mm: float,
    simplify_eps_mm: float,
) -> Tuple[Dict[Tuple[str,str], List[Tuple[float,float]]], Dict[Tuple[str,str], bool]]:
    paths: Dict[Tuple[str,str], List[Tuple[float,float]]] = {}
    ok: Dict[Tuple[str,str], bool] = {}

    for u in nx.topological_sort(G):
        for v in G.successors(u):
            e = (u, v)
            w = float(edge_width_mm.get(e, 1.0))
            clearance = min_wall_mm + w/2.0
            start = G.nodes[u]["xy"]
            goal = G.nodes[v]["xy"]
            path = router.route(start, goal, turn_penalty=0.15)
            success = True
            if path is None or len(path) < 2:
                path = [start, goal]
                success = False
            path = rdp_simplify(path, epsilon=simplify_eps_mm)
            paths[e] = path
            ok[e] = success
            # reserve space
            router.add_path_obstacle(path, radius_mm=clearance)
    return paths, ok


def _apply_junction_rules(
    G: nx.DiGraph,
    edge_paths: Dict[Tuple[str,str], List[Tuple[float,float]]],
    edge_width_mm: Dict[Tuple[str,str], float],
    depth_mm: float,
    stub_len_mm: float,
) -> Dict[str, Dict[str, Any]]:
    """Adjust first segment of child paths near each internal node to satisfy sprout/branch rules.

    Returns metadata keyed by node id for later scoring/ΔP penalties.
    """
    meta: Dict[str, Dict[str, Any]] = {}

    # precompute parent of each node (tree assumption)
    parent = {}
    for u, v in G.edges:
        parent[v] = u

    for n in G.nodes:
        if n == "root":
            continue
        if G.nodes[n].get("kind") != "internal":
            continue
        ch = list(G.successors(n))
        if len(ch) != 2:
            continue  # only handle bifurcation in this MVP
        p = parent.get(n)
        if p is None:
            continue

        # vectors based on existing routed paths
        # parent vec: from n -> p, use first step along path (p->n) reversed if exists
        # We have edge path p->n stored at (p,n).
        e_pn = (p, n)
        if e_pn in edge_paths and len(edge_paths[e_pn]) >= 2:
            # path points from p->n; last segment is into n
            pn_path = edge_paths[e_pn]
            v_parent = (pn_path[-2][0] - pn_path[-1][0], pn_path[-2][1] - pn_path[-1][1])  # n->toward parent
        else:
            # fallback: direct
            xp, yp = G.nodes[p]["xy"]
            xn, yn = G.nodes[n]["xy"]
            v_parent = (xp-xn, yp-yn)

        # child vecs: from n -> child, use first segment of (n,child) path
        child_vecs = []
        for c in ch:
            e_nc = (n, c)
            if e_nc in edge_paths and len(edge_paths[e_nc]) >= 2:
                nc_path = edge_paths[e_nc]
                v = (nc_path[1][0] - nc_path[0][0], nc_path[1][1] - nc_path[0][1])
            else:
                xc, yc = G.nodes[c]["xy"]
                xn, yn = G.nodes[n]["xy"]
                v = (xc-xn, yc-yn)
            child_vecs.append((c, v))

        # identify thick vs thin child by perimeter proxy
        perimeters = []
        for c, _ in child_vecs:
            w = float(edge_width_mm.get((n, c), 1.0))
            perim = 2.0 * (w + depth_mm)  # rectangle perimeter proxy
            perimeters.append((c, perim))
        perimeters_sorted = sorted(perimeters, key=lambda t: t[1], reverse=True)
        thick_child = perimeters_sorted[0][0]
        thin_child = perimeters_sorted[1][0]
        perim_thick = perimeters_sorted[0][1]
        perim_thin = perimeters_sorted[1][1]

        rho = perim_thin / max(1e-9, perim_thick)

        thick_vec = dict(child_vecs)[thick_child]
        thin_vec = dict(child_vecs)[thin_child]

        dirs = compute_bifurcation_directions(v_parent, thick_vec, thin_vec, rho)
        meta[n] = {
            "regime": dirs.regime,
            "rho": dirs.rho,
            "theta_thick_deg": dirs.theta_thick_deg,
            "thick_child": thick_child,
            "thin_child": thin_child,
        }

        # modify child paths with stubs
        xn, yn = G.nodes[n]["xy"]

        def apply_stub(child_id: str, dir_vec: Tuple[float,float]):
            e = (n, child_id)
            if e not in edge_paths:
                return
            pts = edge_paths[e]
            if len(pts) < 2:
                return
            stub = (xn + dir_vec[0]*stub_len_mm, yn + dir_vec[1]*stub_len_mm)
            # insert as second point
            new_pts = [pts[0], stub] + pts[1:]
            # optional: remove near-duplicate
            cleaned = [new_pts[0]]
            for pnt in new_pts[1:]:
                if (pnt[0]-cleaned[-1][0])**2 + (pnt[1]-cleaned[-1][1])**2 > 1e-6:
                    cleaned.append(pnt)
            edge_paths[e] = cleaned

        apply_stub(thick_child, dirs.thick)
        apply_stub(thin_child, dirs.thin)

    return meta


def generate_job(job_spec: JobSpec, job_dir: str) -> List[CandidateSummary]:
    _ensure_dir(job_dir)
    inputs_dir = os.path.join(job_dir, "inputs")
    outputs_dir = os.path.join(job_dir, "candidates")
    _ensure_dir(inputs_dir)
    _ensure_dir(outputs_dir)

    outline = plate_polygon(job_spec.plate)
    plate_w_mm, plate_h_mm = _plate_dims_mm_from_outline(outline, job_spec.plate.kind, job_spec.plate.width_mm, job_spec.plate.height_mm)

    # Load heatmap
    if job_spec.heatmap.kind == "csv":
        grid = load_csv_grid(job_spec.heatmap.path)
    else:
        grid = load_image_grid(job_spec.heatmap.path, flip_y=job_spec.heatmap.flip_y)
    grid = normalize_to_total(grid, job_spec.heatmap.total_watts)

    # Derived leaf target ΔT
    dT = float(job_spec.fluid.target_deltaT_C)
    if dT <= 0:
        dT = 10.0

    cand_summaries: List[CandidateSummary] = []

    n_total = int(job_spec.generation.n_candidates)
    leaf_counts = list(job_spec.generation.leaf_counts) or [12]

    # choose a conservative clearance for allowed region
    clearance_mm = float(job_spec.constraints.min_wall_mm + job_spec.constraints.max_channel_width_mm/2.0)
    allowed = allowed_region(outline, job_spec.constraints, clearance_mm=clearance_mm)

    for idx in range(n_total):
        n_leaves = int(leaf_counts[idx % len(leaf_counts)])
        seed = int(job_spec.generation.seed + idx*1337)

        cand_dir = os.path.join(outputs_dir, f"cand_{idx:02d}")
        _ensure_dir(cand_dir)

        inlet = (job_spec.ports.inlet.x_mm, job_spec.ports.inlet.y_mm)
        outlet = (job_spec.ports.outlet.x_mm, job_spec.ports.outlet.y_mm)

        # leaf points from heatmap
        leaf_xy = _leaf_points_from_heatmap(outline, allowed, plate_w_mm, plate_h_mm, grid, n_leaves, seed=seed)

        # assign watts to leaves and compute mass flows
        leaf_watts = assign_heat_to_leaves(plate_w_mm, plate_h_mm, grid, leaf_xy)
        leaf_mdot = (leaf_watts / (WATER_CP * dT)).astype(float)  # kg/s
        total_mdot = float(leaf_mdot.sum())

        # build trees
        Gs = build_supply_tree(inlet, leaf_xy, leaf_mdot.tolist(), allowed, seed=seed)
        Gr = build_return_tree(outlet, leaf_xy, leaf_mdot.tolist(), allowed, seed=seed+999)

        # edge flows
        mdot_s = compute_edge_mass_flows(Gs)
        mdot_r = compute_edge_mass_flows(Gr)

        # channel sizing
        depth_mm = float(job_spec.constraints.channel_depth_mm)
        v_max = float(job_spec.generation.v_max_m_per_s)
        w_s = {e: width_from_flow(mdot_s[e], depth_mm, v_max, job_spec.constraints.min_channel_width_mm, job_spec.constraints.max_channel_width_mm) for e in mdot_s}
        w_r = {e: width_from_flow(mdot_r[e], depth_mm, v_max, job_spec.constraints.min_channel_width_mm, job_spec.constraints.max_channel_width_mm) for e in mdot_r}

        # trifurcation merge pass (optional; rare)
        _merge_trifurcations_in_tree(Gs, w_s, depth_mm)
        _merge_trifurcations_in_tree(Gr, w_r, depth_mm)

        # Routing
        router = GridRouter(allowed, resolution_mm=float(job_spec.constraints.grid_resolution_mm))
        simplify_eps = float(job_spec.constraints.grid_resolution_mm) * 0.75

        supply_paths, ok_s = _route_tree_edges(Gs, router, w_s, job_spec.constraints.min_wall_mm, simplify_eps)
        return_paths, ok_r = _route_tree_edges(Gr, router, w_r, job_spec.constraints.min_wall_mm, simplify_eps)

        # Junction rules + stubs
        stub_len = max(2.5*job_spec.constraints.min_bend_radius_mm, 3.0*job_spec.constraints.min_channel_width_mm)
        junction_meta_s = _apply_junction_rules(Gs, supply_paths, w_s, depth_mm=depth_mm, stub_len_mm=stub_len)
        junction_meta_r = _apply_junction_rules(Gr, return_paths, w_r, depth_mm=depth_mm, stub_len_mm=stub_len)

        # Channel footprint
        fp_s = channel_footprint_polygon(supply_paths, w_s)
        fp_r = channel_footprint_polygon(return_paths, w_r)
        channels_fp = unary_union([fp_s, fp_r])
        if not channels_fp.is_valid:
            channels_fp = channels_fp.buffer(0)

        # Exports
        plate_step = os.path.join(cand_dir, "plate.step")
        channels_step = os.path.join(cand_dir, "channels.step")
        plate_stl = os.path.join(cand_dir, "plate.stl")
        channels_dxf = os.path.join(cand_dir, "channels.dxf")
        preview_png = os.path.join(cand_dir, "preview.png")
        report_pdf = os.path.join(cand_dir, "report.pdf")
        paths_json = os.path.join(cand_dir, "paths.json")

        # Save debug geometry JSON
        overlay = {
            "outline": list(outline.exterior.coords),
            "inlet": inlet,
            "outlet": outlet,
            "supply_paths": {f"{u}->{v}": supply_paths[(u,v)] for (u,v) in supply_paths},
            "return_paths": {f"{u}->{v}": return_paths[(u,v)] for (u,v) in return_paths},
            "supply_widths": {f"{u}->{v}": w_s[(u,v)] for (u,v) in w_s},
            "return_widths": {f"{u}->{v}": w_r[(u,v)] for (u,v) in w_r},
            "junction_meta_supply": junction_meta_s,
            "junction_meta_return": junction_meta_r,
        }
        with open(paths_json, "w", encoding="utf-8") as f:
            json.dump(overlay, f, indent=2)

        # DXF
        try:
            export_dxf_centerlines(channels_dxf, outline, {**supply_paths, **return_paths}, {**w_s, **w_r})
        except Exception:
            # don't hard fail
            pass

        # CAD solids
        export_cad_solids(outline, channels_fp, job_spec.constraints, plate_step, channels_step, plate_stl)

        # Preview + report
        render_preview_png(preview_png, outline, channels_fp, inlet, outlet, title=f"Candidate {idx}")
        # Metrics
        dp_kpa = estimate_network_dp_kpa(
            Gs, Gr,
            supply_paths, return_paths,
            mdot_s, mdot_r,
            w_s, w_r,
            depth_mm=depth_mm,
            junction_meta=junction_meta_s,  # supply meta used; return is similar (good enough)
        )

        leaf_dt = leaf_deltaT(leaf_watts, leaf_mdot)
        uni_std = float(np.std(leaf_dt))
        uni_max = float(np.max(leaf_dt))
        manuf = manufacturing_score({**supply_paths, **return_paths}, {**ok_s, **ok_r})

        metrics = CandidateMetrics(
            delta_p_kpa=float(dp_kpa),
            uniformity_deltaT_C_std=uni_std,
            uniformity_deltaT_C_max=uni_max,
            manufacturing_score=float(manuf),
            n_leaves=int(n_leaves),
            total_mass_flow_g_per_s=float(total_mdot*1000.0),
            total_watts=float(job_spec.heatmap.total_watts),
        )

        spec_summary = {
            "Plate": f"{job_spec.plate.kind} ({plate_w_mm:.1f}×{plate_h_mm:.1f} mm)",
            "Process": job_spec.constraints.process_preset,
            "Plate thickness (mm)": job_spec.constraints.plate_thickness_mm,
            "Channel depth (mm)": job_spec.constraints.channel_depth_mm,
            "Min channel width (mm)": job_spec.constraints.min_channel_width_mm,
            "Min wall (mm)": job_spec.constraints.min_wall_mm,
            "Grid resolution (mm)": job_spec.constraints.grid_resolution_mm,
            "Total watts (W)": job_spec.heatmap.total_watts,
            "Target ΔT (°C)": job_spec.fluid.target_deltaT_C,
            "Inlet (mm)": f"({inlet[0]:.1f}, {inlet[1]:.1f})",
            "Outlet (mm)": f"({outlet[0]:.1f}, {outlet[1]:.1f})",
        }

        render_report_pdf(report_pdf, spec_summary=spec_summary, metrics=metrics.model_dump(), preview_png_path=preview_png)

        cand = CandidateSummary(
            index=idx,
            label=_label_for_candidate(idx, metrics),
            metrics=metrics,
            artifacts=CandidateArtifacts(
                preview_png=os.path.relpath(preview_png, job_dir),
                report_pdf=os.path.relpath(report_pdf, job_dir),
                plate_step=os.path.relpath(plate_step, job_dir),
                channels_step=os.path.relpath(channels_step, job_dir),
                plate_stl=os.path.relpath(plate_stl, job_dir),
                channels_dxf=os.path.relpath(channels_dxf, job_dir),
                json_paths=os.path.relpath(paths_json, job_dir),
            ),
            overlay=None,  # we keep overlay as JSON file to keep API light
        )
        cand_summaries.append(cand)

    # Rank labels post-hoc
    cand_summaries = _rank_and_relabel(cand_summaries, job_spec)

    return cand_summaries


def _label_for_candidate(idx: int, metrics: CandidateMetrics) -> str:
    return f"Candidate {idx}"


def _rank_and_relabel(cands: List[CandidateSummary], spec: JobSpec) -> List[CandidateSummary]:
    # Compute overall score (lower is better)
    wP = float(spec.generation.weight_pressure)
    wU = float(spec.generation.weight_uniformity)
    wM = float(spec.generation.weight_manufacturing)

    # normalize
    dp = np.array([c.metrics.delta_p_kpa for c in cands], dtype=float)
    uni = np.array([c.metrics.uniformity_deltaT_C_std for c in cands], dtype=float)
    man = np.array([c.metrics.manufacturing_score for c in cands], dtype=float)

    def norm(x):
        x = np.asarray(x, dtype=float)
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx - mn < 1e-9:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    dp_n = norm(dp)
    uni_n = norm(uni)
    man_n = norm(1.0 - man)  # higher manufacturing score is better => invert

    score = wP*dp_n + wU*uni_n + wM*man_n

    order = np.argsort(score).tolist()
    # relabel
    out = []
    for rank, i in enumerate(order):
        c = cands[i]
        label = "Recommended" if rank == 0 else f"Alt {rank}"
        c.label = label
        out.append(c)
    return out
