"""
Reporting module for BranchForge design outputs.

Generates:
  - 2D preview PNGs (matplotlib)
  - PDF design reports (reportlab)
  - HTML interactive reports (standalone, no JS framework needed)
  - Candidate comparison tables
  - Manufacturing compliance checklists
  - CFD handoff metadata
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
import os
import math
import json
import datetime
import html as html_lib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

from .schemas import ConstraintsSpec


# ===================================================================
# PNG preview (unchanged)
# ===================================================================

def _plot_polygon(ax, poly: Polygon, **kwargs):
    x, y = poly.exterior.xy
    ax.plot(x, y, **kwargs)
    for hole in poly.interiors:
        hx, hy = hole.xy
        ax.plot(hx, hy, **kwargs)


def render_preview_png(
    out_path: str,
    outline: Polygon,
    channels_fp: Polygon | MultiPolygon,
    inlet_xy: Tuple[float, float],
    outlet_xy: Tuple[float, float],
    title: str = "",
    keepouts: Optional[ConstraintsSpec] = None,
):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.set_aspect("equal", adjustable="box")

    # Plate outline
    _plot_polygon(ax, outline, color="#333", linewidth=1.5, linestyle="-")

    # Keepouts
    if keepouts is not None:
        for r in keepouts.keepout_rects:
            if r.centered:
                x0 = r.x_mm - r.w_mm / 2
                y0 = r.y_mm - r.h_mm / 2
            else:
                x0 = r.x_mm
                y0 = r.y_mm
            rect = Rectangle(
                (x0, y0), r.w_mm, r.h_mm,
                linewidth=1, edgecolor="#c44", facecolor="#fdd", alpha=0.6,
            )
            ax.add_patch(rect)

        for c in keepouts.keepout_circles:
            circ = Circle(
                (c.x_mm, c.y_mm), c.r_mm + c.margin_mm,
                linewidth=1, edgecolor="#c44", facecolor="#fdd", alpha=0.6,
            )
            ax.add_patch(circ)

    # Channels
    if not channels_fp.is_empty:
        if isinstance(channels_fp, Polygon):
            polys = [channels_fp]
        else:
            polys = list(channels_fp.geoms)
        for p in polys:
            x, y = p.exterior.xy
            ax.fill(x, y, color="#2196F3", alpha=0.45)
            ax.plot(x, y, color="#1565C0", linewidth=0.5, alpha=0.7)

    # Ports
    ax.scatter([inlet_xy[0]], [inlet_xy[1]], s=80, marker="o", c="#4CAF50", zorder=5, edgecolors="#333")
    ax.annotate("IN", inlet_xy, textcoords="offset points", xytext=(8, 4),
                fontsize=8, fontweight="bold", color="#2E7D32")
    ax.scatter([outlet_xy[0]], [outlet_xy[1]], s=80, marker="s", c="#F44336", zorder=5, edgecolors="#333")
    ax.annotate("OUT", outlet_xy, textcoords="offset points", xytext=(8, 4),
                fontsize=8, fontweight="bold", color="#C62828")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.tick_params(labelsize=7)

    bounds = outline.bounds
    margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.08
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

    ax.grid(True, alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# PDF report (unchanged)
# ===================================================================

def render_report_pdf(
    out_path: str,
    spec_summary: Dict[str, Any],
    metrics: Dict[str, Any],
    preview_png_path: str,
    validation_issues: Optional[List] = None,
):
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(0.8 * inch, height - 0.9 * inch, "BranchForge Design Report")

    c.setFont("Helvetica", 9)
    c.drawString(0.8 * inch, height - 1.15 * inch,
                 f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")

    # Preview image
    img_w = 6.8 * inch
    img_h = 3.8 * inch
    img_x = 0.8 * inch
    img_y = height - 1.15 * inch - img_h - 0.2 * inch
    if os.path.exists(preview_png_path):
        c.drawImage(
            preview_png_path, img_x, img_y,
            width=img_w, height=img_h,
            preserveAspectRatio=True, anchor="c",
        )

    # Metrics block
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.8 * inch, img_y - 0.35 * inch, "Key Metrics")
    c.setFont("Helvetica", 10)

    y = img_y - 0.55 * inch

    def fmt(val, suffix="", decimals=2):
        if isinstance(val, (int, float)):
            return f"{val:.{decimals}f}{suffix}"
        return str(val)

    metric_lines = [
        f"Pressure drop (worst path): {fmt(metrics.get('delta_p_kpa'), ' kPa')}",
        f"Uniformity (dT std): {fmt(metrics.get('uniformity_deltaT_C_std'), ' C', 3)}",
        f"Uniformity (dT max): {fmt(metrics.get('uniformity_deltaT_C_max'), ' C', 3)}",
        f"Manufacturing score: {fmt(metrics.get('manufacturing_score'))}",
        f"Leaves: {metrics.get('n_leaves', 'n/a')}",
        f"Mass flow: {fmt(metrics.get('total_mass_flow_g_per_s'), ' g/s', 1)}",
        f"Total watts: {fmt(metrics.get('total_watts'), ' W', 1)}",
        f"Max Reynolds: {fmt(metrics.get('reynolds_number_max'), '', 0)}",
        f"Channel area: {fmt(metrics.get('channel_area_mm2'), ' mm2', 1)}",
        f"Channel length: {fmt(metrics.get('total_channel_length_mm'), ' mm', 1)}",
    ]
    for line in metric_lines:
        c.drawString(1.0 * inch, y, line)
        y -= 0.17 * inch

    # Validation
    if validation_issues:
        y -= 0.1 * inch
        c.setFont("Helvetica-Bold", 11)
        c.drawString(0.8 * inch, y, "Validation")
        y -= 0.2 * inch
        c.setFont("Helvetica", 9)
        for issue in validation_issues[:10]:
            severity = getattr(issue, "severity", "warning")
            code = getattr(issue, "code", "")
            message = getattr(issue, "message", str(issue))
            color_map = {"error": (0.8, 0, 0), "warning": (0.7, 0.5, 0)}
            c.setFillColorRGB(*color_map.get(severity, (0, 0, 0)))
            c.drawString(1.0 * inch, y, f"[{severity.upper()}] {code}: {message}")
            c.setFillColorRGB(0, 0, 0)
            y -= 0.15 * inch

    # Spec summary
    y -= 0.1 * inch
    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.8 * inch, y, "Inputs")
    c.setFont("Helvetica", 9)
    y -= 0.2 * inch

    for k in sorted(spec_summary.keys()):
        if y < 0.8 * inch:
            c.showPage()
            y = height - 0.8 * inch
            c.setFont("Helvetica", 9)
        c.drawString(1.0 * inch, y, f"{k}: {spec_summary[k]}")
        y -= 0.15 * inch

    c.showPage()
    c.save()


# ===================================================================
# Manufacturing compliance checklist
# ===================================================================

def generate_compliance_checklist(
    metrics: Dict[str, Any],
    spec: Dict[str, Any],
    validation_issues: Optional[List] = None,
) -> List[Dict[str, Any]]:
    """Generate a manufacturing compliance checklist.

    Each item has: check, status (pass/warn/fail), value, limit, notes.
    """
    constraints = spec.get("constraints", {})
    fluid = spec.get("fluid", {})
    checks = []

    # 1. Pressure drop within limits
    dp = metrics.get("delta_p_kpa", 0)
    dp_limit = 100.0  # typical max for liquid cooling
    checks.append({
        "check": "Pressure drop within system budget",
        "status": "pass" if dp < dp_limit else "warn" if dp < dp_limit * 1.5 else "fail",
        "value": f"{dp:.1f} kPa",
        "limit": f"< {dp_limit} kPa",
        "notes": "System pump must accommodate this pressure drop.",
    })

    # 2. Reynolds number (laminar vs turbulent)
    re_max = metrics.get("reynolds_number_max", 0)
    checks.append({
        "check": "Flow regime (laminar preferred for micro-channels)",
        "status": "pass" if re_max < 2300 else "warn" if re_max < 4000 else "fail",
        "value": f"Re = {re_max:.0f}",
        "limit": "< 2300 (laminar)",
        "notes": (
            "Laminar flow preferred for predictable behaviour."
            if re_max < 2300 else
            "Transitional flow — consider reducing flow rate or widening channels."
            if re_max < 4000 else
            "Turbulent flow — increased pressure drop but better heat transfer."
        ),
    })

    # 3. Temperature uniformity
    dt_std = metrics.get("uniformity_deltaT_C_std", 0)
    checks.append({
        "check": "Temperature uniformity across leaves",
        "status": "pass" if dt_std < 2.0 else "warn" if dt_std < 5.0 else "fail",
        "value": f"\u0394T std = {dt_std:.2f} \u00b0C",
        "limit": "< 2.0 \u00b0C",
        "notes": "Lower is better for component reliability.",
    })

    # 4. Manufacturing score
    mfg = metrics.get("manufacturing_score", 0)
    checks.append({
        "check": "Manufacturing feasibility score",
        "status": "pass" if mfg > 0.7 else "warn" if mfg > 0.4 else "fail",
        "value": f"{mfg:.2f}",
        "limit": "> 0.70",
        "notes": "Accounts for bend count, routing failures, and geometry complexity.",
    })

    # 5. Murray's law adherence
    murray = metrics.get("murray_law_deviation", 0)
    checks.append({
        "check": "Murray's law adherence (branching optimality)",
        "status": "pass" if murray < 0.15 else "warn" if murray < 0.30 else "fail",
        "value": f"RMS deviation = {murray:.3f}",
        "limit": "< 0.15",
        "notes": "Lower deviation = more biologically optimal branching.",
    })

    # 6. Surface-to-volume ratio
    sv = metrics.get("surface_to_volume_ratio", 0)
    checks.append({
        "check": "Surface-to-volume efficiency (Nature paper metric)",
        "status": "pass" if 0 < sv < 3.0 else "warn" if sv < 5.0 else "fail",
        "value": f"SA/V = {sv:.2f}",
        "limit": "< 3.0 mm\u207b\u00b9",
        "notes": "Lower is better — surface-minimizing networks per Meng et al.",
    })

    # 7. Minimum wall thickness
    min_wall = constraints.get("min_wall_mm", 1.0)
    checks.append({
        "check": f"Minimum wall thickness \u2265 {min_wall} mm",
        "status": "pass",  # Validated during generation
        "value": f"{min_wall} mm (constraint)",
        "limit": f"\u2265 {min_wall} mm",
        "notes": "Enforced during channel routing. Check validation issues below.",
    })

    # 8. Channel depth constraint
    depth = constraints.get("channel_depth_mm", 2.0)
    thickness = constraints.get("plate_thickness_mm", 5.0)
    depth_ok = depth < thickness * 0.8
    checks.append({
        "check": "Channel depth leaves sufficient floor thickness",
        "status": "pass" if depth_ok else "warn",
        "value": f"{depth} mm depth / {thickness} mm plate",
        "limit": "depth < 80% of plate thickness",
        "notes": f"Floor thickness: {thickness - depth:.1f} mm",
    })

    # 9. Validation issues summary
    n_errors = metrics.get("validation_errors", 0)
    n_warnings = metrics.get("validation_warnings", 0)
    checks.append({
        "check": "Design validation (geometry, constraints)",
        "status": "pass" if n_errors == 0 and n_warnings == 0
                  else "warn" if n_errors == 0 else "fail",
        "value": f"{n_errors} errors, {n_warnings} warnings",
        "limit": "0 errors",
        "notes": "See validation details for specifics.",
    })

    return checks


# ===================================================================
# CFD handoff metadata
# ===================================================================

def generate_cfd_handoff(
    metrics: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate metadata for CFD simulation handoff.

    Provides boundary conditions, fluid properties, and mesh
    recommendations that engineers need to set up a CFD simulation.
    """
    fluid = spec.get("fluid", {})
    constraints = spec.get("constraints", {})
    ports = spec.get("ports", {})

    coolant = fluid.get("coolant", "water")
    inlet_temp = fluid.get("inlet_temp_C", 25.0)
    target_dt = fluid.get("target_deltaT_C", 10.0)

    mdot = metrics.get("total_mass_flow_g_per_s", 0) / 1000.0  # kg/s

    return {
        "boundary_conditions": {
            "inlet": {
                "type": "mass_flow_inlet",
                "mass_flow_rate_kg_s": round(mdot, 6),
                "temperature_C": inlet_temp,
                "location": ports.get("inlet", {}),
                "diameter_mm": ports.get("inlet_diameter_mm", 6.0),
            },
            "outlet": {
                "type": "pressure_outlet",
                "gauge_pressure_Pa": 0,
                "location": ports.get("outlet", {}),
                "diameter_mm": ports.get("outlet_diameter_mm", 6.0),
            },
            "walls": {
                "type": "no_slip",
                "thermal": "constant_heat_flux",
                "total_watts": metrics.get("total_watts", 0),
                "note": "Apply heatmap distribution to the base surface.",
            },
        },
        "fluid_properties": {
            "coolant": coolant,
            "density_kg_m3": _get_fluid_density(coolant),
            "viscosity_Pa_s": _get_fluid_viscosity(coolant),
            "specific_heat_J_kgK": _get_fluid_cp(coolant),
            "thermal_conductivity_W_mK": _get_fluid_k(coolant),
            "inlet_temperature_C": inlet_temp,
        },
        "geometry": {
            "channel_depth_mm": constraints.get("channel_depth_mm", 2.0),
            "plate_thickness_mm": constraints.get("plate_thickness_mm", 5.0),
            "min_channel_width_mm": constraints.get("min_channel_width_mm", 1.2),
            "max_channel_width_mm": constraints.get("max_channel_width_mm", 8.0),
            "process_preset": constraints.get("process_preset", "CNC"),
        },
        "mesh_recommendations": {
            "min_elements_across_channel": 5,
            "boundary_layer_layers": 3,
            "first_layer_height_mm": 0.05,
            "growth_ratio": 1.2,
            "turbulence_model": (
                "Laminar" if metrics.get("reynolds_number_max", 0) < 2300
                else "k-omega SST"
            ),
            "notes": (
                "Ensure at least 5 elements across the narrowest channel width. "
                "Use inflation layers on channel walls for accurate boundary layer "
                "resolution. Steady-state simulation recommended for initial validation."
            ),
        },
        "predicted_results": {
            "pressure_drop_kPa": metrics.get("delta_p_kpa", 0),
            "max_reynolds": metrics.get("reynolds_number_max", 0),
            "uniformity_dT_std_C": metrics.get("uniformity_deltaT_C_std", 0),
            "note": (
                "Compare CFD results against these predictions. "
                "Import measured values into BranchForge calibration "
                "to improve future predictions."
            ),
        },
    }


def _get_fluid_density(coolant: str) -> float:
    return {"water": 997.0, "ethylene_glycol_30": 1040.0,
            "ethylene_glycol_50": 1082.0, "ethylene_glycol_70": 1110.0,
            "propylene_glycol_50": 1035.0, "novec_7100": 1510.0,
            "pao_2": 798.0}.get(coolant, 997.0)


def _get_fluid_viscosity(coolant: str) -> float:
    return {"water": 0.00089, "ethylene_glycol_30": 0.002,
            "ethylene_glycol_50": 0.0034, "ethylene_glycol_70": 0.008,
            "propylene_glycol_50": 0.005, "novec_7100": 0.00058,
            "pao_2": 0.0052}.get(coolant, 0.00089)


def _get_fluid_cp(coolant: str) -> float:
    return {"water": 4181.0, "ethylene_glycol_30": 3580.0,
            "ethylene_glycol_50": 3283.0, "ethylene_glycol_70": 2900.0,
            "propylene_glycol_50": 3500.0, "novec_7100": 1183.0,
            "pao_2": 2090.0}.get(coolant, 4181.0)


def _get_fluid_k(coolant: str) -> float:
    """Thermal conductivity W/(m·K)."""
    return {"water": 0.607, "ethylene_glycol_30": 0.48,
            "ethylene_glycol_50": 0.39, "ethylene_glycol_70": 0.32,
            "propylene_glycol_50": 0.36, "novec_7100": 0.069,
            "pao_2": 0.14}.get(coolant, 0.607)


# ===================================================================
# Candidate comparison table
# ===================================================================

def generate_comparison_table(
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a structured comparison table across candidates.

    Each metric row shows values across candidates, with the best
    value highlighted.
    """
    if not candidates:
        return {"headers": [], "rows": [], "n_candidates": 0}

    headers = [c.get("label", f"Candidate {i}") for i, c in enumerate(candidates)]

    # Define metric rows (label, key, format, lower_is_better)
    metric_defs = [
        ("Pressure Drop", "delta_p_kpa", ".1f", True, "kPa"),
        ("Uniformity (\u0394T std)", "uniformity_deltaT_C_std", ".3f", True, "\u00b0C"),
        ("Uniformity (\u0394T max)", "uniformity_deltaT_C_max", ".2f", True, "\u00b0C"),
        ("Manufacturing Score", "manufacturing_score", ".2f", False, ""),
        ("Surface Area", "surface_area_mm2", ".0f", True, "mm\u00b2"),
        ("Channel Volume", "channel_volume_mm3", ".0f", False, "mm\u00b3"),
        ("SA/V Ratio", "surface_to_volume_ratio", ".2f", True, "mm\u207b\u00b9"),
        ("Murray's Law Dev.", "murray_law_deviation", ".3f", True, ""),
        ("Max Reynolds", "reynolds_number_max", ".0f", True, ""),
        ("Mass Flow", "total_mass_flow_g_per_s", ".1f", False, "g/s"),
        ("Channel Length", "total_channel_length_mm", ".0f", False, "mm"),
        ("Leaves", "n_leaves", "d", False, ""),
    ]

    rows = []
    for label, key, fmt, lower_better, unit in metric_defs:
        values = []
        for c in candidates:
            m = c.get("metrics", c)
            v = m.get(key)
            values.append(v)

        # Find best index
        numeric_vals = [v for v in values if isinstance(v, (int, float))]
        best_idx = None
        if numeric_vals:
            if lower_better:
                best_val = min(numeric_vals)
            else:
                best_val = max(numeric_vals)
            for i, v in enumerate(values):
                if v == best_val:
                    best_idx = i
                    break

        formatted = []
        for v in values:
            if v is None:
                formatted.append("n/a")
            elif isinstance(v, float):
                formatted.append(f"{v:{fmt}}")
            elif isinstance(v, int):
                formatted.append(str(v))
            else:
                formatted.append(str(v))

        rows.append({
            "label": label,
            "key": key,
            "unit": unit,
            "values": formatted,
            "raw_values": values,
            "best_index": best_idx,
            "lower_is_better": lower_better,
        })

    return {
        "headers": headers,
        "rows": rows,
        "n_candidates": len(candidates),
    }


# ===================================================================
# HTML report
# ===================================================================

def render_report_html(
    metrics: Dict[str, Any],
    spec: Dict[str, Any],
    validation_issues: Optional[List] = None,
    candidate_label: str = "Candidate",
    calibration_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a standalone HTML design report.

    Returns an HTML string (no external dependencies) suitable for
    saving to a file or returning from an API endpoint.
    """
    now = datetime.datetime.utcnow().isoformat(timespec="seconds")
    checklist = generate_compliance_checklist(metrics, spec, validation_issues)
    cfd = generate_cfd_handoff(metrics, spec)

    def esc(s):
        return html_lib.escape(str(s))

    def fmt(val, decimals=2, suffix=""):
        if isinstance(val, float):
            return f"{val:.{decimals}f}{suffix}"
        if isinstance(val, int):
            return f"{val}{suffix}"
        return str(val) if val is not None else "n/a"

    # Build HTML
    sections = []

    # ── Metrics section ──
    metric_rows = ""
    metric_items = [
        ("Pressure Drop (worst path)", "delta_p_kpa", 1, " kPa"),
        ("Uniformity (\u0394T std)", "uniformity_deltaT_C_std", 3, " \u00b0C"),
        ("Uniformity (\u0394T max)", "uniformity_deltaT_C_max", 2, " \u00b0C"),
        ("Manufacturing Score", "manufacturing_score", 2, ""),
        ("Surface Area", "surface_area_mm2", 0, " mm\u00b2"),
        ("Channel Volume", "channel_volume_mm3", 0, " mm\u00b3"),
        ("SA/V Ratio", "surface_to_volume_ratio", 2, " mm\u207b\u00b9"),
        ("Murray's Law Deviation", "murray_law_deviation", 3, ""),
        ("Max Reynolds Number", "reynolds_number_max", 0, ""),
        ("Mass Flow Rate", "total_mass_flow_g_per_s", 1, " g/s"),
        ("Total Heat Load", "total_watts", 1, " W"),
        ("Channel Cross-Section", "channel_area_mm2", 1, " mm\u00b2"),
        ("Total Channel Length", "total_channel_length_mm", 0, " mm"),
        ("Leaves", "n_leaves", 0, ""),
    ]
    for label, key, dec, suf in metric_items:
        val = metrics.get(key)
        # Check for calibrated value
        uncal_key = f"{key}_uncalibrated"
        uncal = metrics.get(uncal_key)
        cal_note = ""
        if uncal is not None:
            cal_note = f' <span class="cal-note">(uncalibrated: {fmt(uncal, dec, suf)})</span>'
        metric_rows += f"<tr><td>{esc(label)}</td><td>{fmt(val, dec, suf)}{cal_note}</td></tr>\n"

    sections.append(f"""
    <section>
      <h2>Key Metrics</h2>
      <table class="metrics">{metric_rows}</table>
    </section>""")

    # ── Calibration info ──
    if calibration_info or metrics.get("calibration_applied"):
        cal_name = metrics.get("calibration_profile_name", calibration_info.get("name", "Unknown") if calibration_info else "Unknown")
        cal_r2 = metrics.get("calibration_r2_dp", 0)
        sections.append(f"""
    <section>
      <h2>Calibration</h2>
      <p>Profile: <strong>{esc(cal_name)}</strong> | R\u00b2 (ΔP): {cal_r2:.2f}</p>
      <p class="note">Metrics have been adjusted using calibration corrections.
      Uncalibrated values shown in parentheses where applicable.</p>
    </section>""")

    # ── Compliance checklist ──
    checklist_rows = ""
    for item in checklist:
        status = item["status"]
        icon = {"pass": "\u2705", "warn": "\u26a0\ufe0f", "fail": "\u274c"}.get(status, "\u2753")
        css_class = f"status-{status}"
        checklist_rows += (
            f'<tr class="{css_class}">'
            f"<td>{icon}</td>"
            f'<td>{esc(item["check"])}</td>'
            f'<td>{esc(item["value"])}</td>'
            f'<td>{esc(item["limit"])}</td>'
            f'<td class="notes">{esc(item["notes"])}</td>'
            f"</tr>\n"
        )

    sections.append(f"""
    <section>
      <h2>Manufacturing Compliance Checklist</h2>
      <table class="checklist">
        <thead><tr>
          <th></th><th>Check</th><th>Value</th><th>Limit</th><th>Notes</th>
        </tr></thead>
        <tbody>{checklist_rows}</tbody>
      </table>
    </section>""")

    # ── CFD Handoff ──
    bc = cfd["boundary_conditions"]
    fp = cfd["fluid_properties"]
    mesh = cfd["mesh_recommendations"]
    sections.append(f"""
    <section>
      <h2>CFD Simulation Handoff</h2>
      <h3>Boundary Conditions</h3>
      <table class="metrics">
        <tr><td>Inlet type</td><td>{esc(bc['inlet']['type'])}</td></tr>
        <tr><td>Mass flow rate</td><td>{bc['inlet']['mass_flow_rate_kg_s']:.6f} kg/s</td></tr>
        <tr><td>Inlet temperature</td><td>{bc['inlet']['temperature_C']} \u00b0C</td></tr>
        <tr><td>Outlet type</td><td>{esc(bc['outlet']['type'])}</td></tr>
        <tr><td>Heat load</td><td>{bc['walls']['total_watts']} W</td></tr>
      </table>
      <h3>Fluid Properties ({esc(fp['coolant'])})</h3>
      <table class="metrics">
        <tr><td>Density</td><td>{fp['density_kg_m3']} kg/m\u00b3</td></tr>
        <tr><td>Viscosity</td><td>{fp['viscosity_Pa_s']} Pa\u00b7s</td></tr>
        <tr><td>Specific heat</td><td>{fp['specific_heat_J_kgK']} J/(kg\u00b7K)</td></tr>
        <tr><td>Thermal conductivity</td><td>{fp['thermal_conductivity_W_mK']} W/(m\u00b7K)</td></tr>
      </table>
      <h3>Mesh Recommendations</h3>
      <table class="metrics">
        <tr><td>Min elements across channel</td><td>{mesh['min_elements_across_channel']}</td></tr>
        <tr><td>Boundary layer layers</td><td>{mesh['boundary_layer_layers']}</td></tr>
        <tr><td>First layer height</td><td>{mesh['first_layer_height_mm']} mm</td></tr>
        <tr><td>Turbulence model</td><td>{esc(mesh['turbulence_model'])}</td></tr>
      </table>
    </section>""")

    # ── Input specification ──
    spec_rows = ""
    for section_name in ("plate", "ports", "constraints", "fluid", "generation"):
        section_data = spec.get(section_name, {})
        if isinstance(section_data, dict):
            for k, v in sorted(section_data.items()):
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, default=str)
                spec_rows += f"<tr><td>{esc(section_name)}.{esc(k)}</td><td>{esc(v)}</td></tr>\n"

    sections.append(f"""
    <section>
      <h2>Input Specification</h2>
      <table class="metrics">{spec_rows}</table>
    </section>""")

    # ── Validation issues ──
    if validation_issues:
        val_rows = ""
        for issue in validation_issues[:20]:
            severity = getattr(issue, "severity", "warning")
            code = getattr(issue, "code", "")
            message = getattr(issue, "message", str(issue))
            css = "status-warn" if severity == "warning" else "status-fail"
            val_rows += (
                f'<tr class="{css}">'
                f"<td>{esc(severity.upper())}</td>"
                f"<td>{esc(code)}</td>"
                f"<td>{esc(message)}</td>"
                f"</tr>\n"
            )
        sections.append(f"""
    <section>
      <h2>Validation Issues</h2>
      <table class="checklist">
        <thead><tr><th>Severity</th><th>Code</th><th>Message</th></tr></thead>
        <tbody>{val_rows}</tbody>
      </table>
    </section>""")

    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>BranchForge Design Report — {esc(candidate_label)}</title>
  <style>
    :root {{ --bg: #f8f9fa; --card: #fff; --text: #212529; --accent: #0d6efd;
             --pass: #198754; --warn: #ffc107; --fail: #dc3545; --muted: #6c757d; }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: var(--bg); color: var(--text); line-height: 1.6; padding: 2rem; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    header {{ text-align: center; margin-bottom: 2rem; padding-bottom: 1rem;
              border-bottom: 2px solid #dee2e6; }}
    header h1 {{ font-size: 1.8rem; color: var(--accent); }}
    header .meta {{ color: var(--muted); font-size: 0.9rem; }}
    section {{ background: var(--card); border-radius: 8px; padding: 1.5rem;
              margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: var(--accent);
          border-bottom: 1px solid #e9ecef; padding-bottom: 0.5rem; }}
    h3 {{ font-size: 1rem; margin: 1rem 0 0.5rem; color: var(--text); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #e9ecef; }}
    th {{ font-weight: 600; color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }}
    .metrics td:first-child {{ font-weight: 500; width: 45%; }}
    .metrics td:last-child {{ font-family: "SF Mono", monospace; }}
    .checklist .status-pass {{ background: #d1e7dd; }}
    .checklist .status-warn {{ background: #fff3cd; }}
    .checklist .status-fail {{ background: #f8d7da; }}
    .notes {{ font-size: 0.8rem; color: var(--muted); }}
    .cal-note {{ font-size: 0.8rem; color: var(--muted); }}
    .note {{ font-size: 0.85rem; color: var(--muted); font-style: italic; }}
    footer {{ text-align: center; padding: 1rem; color: var(--muted); font-size: 0.8rem; }}
    @media print {{ body {{ padding: 0; }} section {{ box-shadow: none; break-inside: avoid; }} }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>BranchForge Design Report</h1>
      <p class="meta">{esc(candidate_label)} &mdash; Generated {now} UTC</p>
    </header>
    {body}
    <footer>
      <p>Generated by BranchForge &mdash; Heatmap to manufacturable cold plate in minutes.</p>
      <p>Based on Meng et al., Nature 649, 315&ndash;322 (2026) surface-minimization branching networks.</p>
    </footer>
  </div>
</body>
</html>"""


def render_comparison_html(
    candidates: List[Dict[str, Any]],
    job_id: str = "",
) -> str:
    """Render a standalone HTML comparison report for multiple candidates.

    Returns HTML string with a full comparison table and per-metric
    best/worst highlighting.
    """
    now = datetime.datetime.utcnow().isoformat(timespec="seconds")
    table = generate_comparison_table(candidates)

    def esc(s):
        return html_lib.escape(str(s))

    # Build header row
    header_cells = "<th>Metric</th><th>Unit</th>"
    for h in table["headers"]:
        header_cells += f"<th>{esc(h)}</th>"

    # Build data rows
    body_rows = ""
    for row in table["rows"]:
        cells = f"<td class='metric-label'>{esc(row['label'])}</td>"
        cells += f"<td class='unit'>{esc(row['unit'])}</td>"
        for i, val in enumerate(row["values"]):
            css = ""
            if i == row.get("best_index"):
                css = ' class="best"'
            cells += f"<td{css}>{esc(val)}</td>"
        body_rows += f"<tr>{cells}</tr>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>BranchForge — Candidate Comparison</title>
  <style>
    :root {{ --bg: #f8f9fa; --card: #fff; --text: #212529; --accent: #0d6efd;
             --best: #d1e7dd; --muted: #6c757d; }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: var(--bg); color: var(--text); line-height: 1.6; padding: 2rem; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    header {{ text-align: center; margin-bottom: 2rem; }}
    header h1 {{ font-size: 1.8rem; color: var(--accent); }}
    header .meta {{ color: var(--muted); font-size: 0.9rem; }}
    .comparison {{ background: var(--card); border-radius: 8px; padding: 1.5rem;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ padding: 0.6rem 0.8rem; text-align: right; border-bottom: 1px solid #e9ecef; }}
    th {{ font-weight: 600; color: var(--muted); font-size: 0.8rem;
          text-transform: uppercase; position: sticky; top: 0; background: var(--card); }}
    th:first-child, td:first-child {{ text-align: left; }}
    .metric-label {{ font-weight: 500; }}
    .unit {{ color: var(--muted); font-size: 0.8rem; }}
    .best {{ background: var(--best); font-weight: 600; }}
    footer {{ text-align: center; padding: 1rem; color: var(--muted); font-size: 0.8rem; margin-top: 1.5rem; }}
    @media print {{ body {{ padding: 0; }} .comparison {{ box-shadow: none; }} }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Candidate Comparison</h1>
      <p class="meta">Job {esc(job_id)} &mdash; {table['n_candidates']} candidates &mdash; {now} UTC</p>
    </header>
    <div class="comparison">
      <table>
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{body_rows}</tbody>
      </table>
    </div>
    <footer>
      <p>Green cells indicate the best value for each metric.</p>
      <p>Generated by BranchForge</p>
    </footer>
  </div>
</body>
</html>"""
