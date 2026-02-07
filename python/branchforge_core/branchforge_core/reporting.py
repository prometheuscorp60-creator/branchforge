from __future__ import annotations

from typing import Dict, Any, Tuple
import os
import io
import math
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


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
    inlet_xy: Tuple[float,float],
    outlet_xy: Tuple[float,float],
    title: str = "",
):
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)
    ax.set_aspect("equal", adjustable="box")
    _plot_polygon(ax, outline, linewidth=1.5)

    if not channels_fp.is_empty:
        if isinstance(channels_fp, Polygon):
            polys = [channels_fp]
        else:
            polys = list(channels_fp.geoms)
        for p in polys:
            x, y = p.exterior.xy
            ax.fill(x, y, alpha=0.5)

    ax.scatter([inlet_xy[0]], [inlet_xy[1]], marker="o")
    ax.text(inlet_xy[0], inlet_xy[1], "IN", fontsize=8)
    ax.scatter([outlet_xy[0]], [outlet_xy[1]], marker="o")
    ax.text(outlet_xy[0], outlet_xy[1], "OUT", fontsize=8)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(outline.bounds[0]-5, outline.bounds[2]+5)
    ax.set_ylim(outline.bounds[1]-5, outline.bounds[3]+5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_report_pdf(
    out_path: str,
    spec_summary: Dict[str, Any],
    metrics: Dict[str, Any],
    preview_png_path: str,
):
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.8*inch, height - 0.9*inch, "BranchForge Design Report")

    c.setFont("Helvetica", 10)
    c.drawString(0.8*inch, height - 1.1*inch, f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")

    # Preview image
    img_w = 6.8*inch
    img_h = 3.6*inch
    img_x = 0.8*inch
    img_y = height - 1.1*inch - img_h - 0.2*inch
    if os.path.exists(preview_png_path):
        c.drawImage(preview_png_path, img_x, img_y, width=img_w, height=img_h, preserveAspectRatio=True, anchor='c')

    # Metrics block
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.8*inch, img_y - 0.35*inch, "Key metrics")
    c.setFont("Helvetica", 10)

    y = img_y - 0.55*inch
    lines = [
        f"ΔP (worst path): {metrics.get('delta_p_kpa', 'n/a'):.2f} kPa" if isinstance(metrics.get('delta_p_kpa'), (int,float)) else f"ΔP (worst path): {metrics.get('delta_p_kpa')}",
        f"Uniformity (ΔT std): {metrics.get('uniformity_deltaT_C_std', 'n/a'):.3f} °C" if isinstance(metrics.get('uniformity_deltaT_C_std'), (int,float)) else f"Uniformity (ΔT std): {metrics.get('uniformity_deltaT_C_std')}",
        f"Uniformity (ΔT max): {metrics.get('uniformity_deltaT_C_max', 'n/a'):.3f} °C" if isinstance(metrics.get('uniformity_deltaT_C_max'), (int,float)) else f"Uniformity (ΔT max): {metrics.get('uniformity_deltaT_C_max')}",
        f"Manufacturing score: {metrics.get('manufacturing_score', 'n/a'):.2f}" if isinstance(metrics.get('manufacturing_score'), (int,float)) else f"Manufacturing score: {metrics.get('manufacturing_score')}",
        f"Leaves: {metrics.get('n_leaves', 'n/a')}",
        f"Total mass flow: {metrics.get('total_mass_flow_g_per_s', 'n/a'):.1f} g/s" if isinstance(metrics.get('total_mass_flow_g_per_s'), (int,float)) else f"Total mass flow: {metrics.get('total_mass_flow_g_per_s')}",
        f"Total watts: {metrics.get('total_watts', 'n/a'):.1f} W" if isinstance(metrics.get('total_watts'), (int,float)) else f"Total watts: {metrics.get('total_watts')}",
    ]
    for line in lines:
        c.drawString(1.0*inch, y, line)
        y -= 0.18*inch

    # Spec summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.8*inch, y - 0.1*inch, "Inputs")
    c.setFont("Helvetica", 9)
    y -= 0.35*inch

    def draw_kv(k, v):
        nonlocal y
        c.drawString(1.0*inch, y, f"{k}: {v}")
        y -= 0.16*inch

    for k in sorted(spec_summary.keys()):
        draw_kv(k, spec_summary[k])

    c.showPage()
    c.save()
