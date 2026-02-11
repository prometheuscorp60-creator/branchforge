"""
Pre-configured templates for common cold plate use cases.

Each template provides realistic engineering defaults for plate geometry,
port placement, keepout zones, fluid parameters, and generation settings.
All dimensions are in millimetres unless noted otherwise.
"""

from __future__ import annotations

from typing import List

from .schemas import (
    TemplateSpec,
    PlateSpec,
    PortSpec,
    ConstraintsSpec,
    FluidSpec,
    GenerationSpec,
    KeepoutCircle,
    Point2D,
)

# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

_TEMPLATES: List[TemplateSpec] = [
    # -----------------------------------------------------------------------
    # 1. gpu_single  --  Single GPU cold plate
    # -----------------------------------------------------------------------
    TemplateSpec(
        id="gpu_single",
        name="Single GPU Cold Plate",
        description=(
            "100 x 60 mm cold plate sized for a single GPU die with a central "
            "hotspot at 350 W.  Four M3 bolt holes near the corners serve as "
            "keepout zones.  Inlet on the left edge, outlet on the right."
        ),
        category="gpu",
        tags=["gpu", "single-die", "cnc", "350W"],
        plate=PlateSpec(
            kind="rectangle",
            width_mm=100.0,
            height_mm=60.0,
        ),
        ports=PortSpec(
            inlet=Point2D(x_mm=5.0, y_mm=30.0),
            outlet=Point2D(x_mm=95.0, y_mm=30.0),
            inlet_diameter_mm=6.0,
            outlet_diameter_mm=6.0,
        ),
        constraints=ConstraintsSpec(
            plate_thickness_mm=5.0,
            channel_depth_mm=2.0,
            min_channel_width_mm=1.2,
            max_channel_width_mm=8.0,
            min_wall_mm=1.0,
            min_bend_radius_mm=3.0,
            grid_resolution_mm=2.0,
            process_preset="CNC",
            keepout_circles=[
                # M3 bolt holes (1.5 mm radius + 1.5 mm margin = 3.0 mm clear)
                KeepoutCircle(x_mm=5.0,  y_mm=5.0,  r_mm=1.5, margin_mm=1.5, label="M3-BL"),
                KeepoutCircle(x_mm=95.0, y_mm=5.0,  r_mm=1.5, margin_mm=1.5, label="M3-BR"),
                KeepoutCircle(x_mm=5.0,  y_mm=55.0, r_mm=1.5, margin_mm=1.5, label="M3-TL"),
                KeepoutCircle(x_mm=95.0, y_mm=55.0, r_mm=1.5, margin_mm=1.5, label="M3-TR"),
            ],
        ),
        fluid=FluidSpec(
            coolant="water",
            inlet_temp_C=25.0,
            target_deltaT_C=10.0,
        ),
        generation=GenerationSpec(
            n_candidates=10,
            leaf_counts=[8, 12, 16],
            seed=42,
            weight_pressure=1.0,
            weight_uniformity=1.0,
            weight_manufacturing=0.5,
            v_max_m_per_s=1.5,
        ),
        heatmap_description=(
            "Centre hotspot (~25 x 25 mm) at plate midpoint delivering 350 W.  "
            "Gaussian profile with peak flux ~560 kW/m^2."
        ),
    ),

    # -----------------------------------------------------------------------
    # 2. gpu_dual  --  Dual GPU module
    # -----------------------------------------------------------------------
    TemplateSpec(
        id="gpu_dual",
        name="Dual GPU Module Cold Plate",
        description=(
            "180 x 80 mm cold plate for a dual-GPU module dissipating 700 W "
            "total (350 W per die).  Two hotspot zones sit side-by-side along "
            "the plate centre line.  Six M4 bolt holes ring the perimeter.  "
            "Inlet at bottom-left, outlet at bottom-right."
        ),
        category="gpu",
        tags=["gpu", "dual-die", "cnc", "700W"],
        plate=PlateSpec(
            kind="rectangle",
            width_mm=180.0,
            height_mm=80.0,
        ),
        ports=PortSpec(
            inlet=Point2D(x_mm=5.0, y_mm=10.0),
            outlet=Point2D(x_mm=175.0, y_mm=10.0),
            inlet_diameter_mm=8.0,
            outlet_diameter_mm=8.0,
        ),
        constraints=ConstraintsSpec(
            plate_thickness_mm=6.0,
            channel_depth_mm=2.5,
            min_channel_width_mm=1.2,
            max_channel_width_mm=8.0,
            min_wall_mm=1.0,
            min_bend_radius_mm=3.0,
            grid_resolution_mm=2.0,
            process_preset="CNC",
            keepout_circles=[
                # 6x M4 bolt holes (2.0 mm radius + 2.0 mm margin)
                # Bottom edge
                KeepoutCircle(x_mm=6.0,   y_mm=6.0,  r_mm=2.0, margin_mm=2.0, label="M4-BL"),
                KeepoutCircle(x_mm=90.0,  y_mm=6.0,  r_mm=2.0, margin_mm=2.0, label="M4-BC"),
                KeepoutCircle(x_mm=174.0, y_mm=6.0,  r_mm=2.0, margin_mm=2.0, label="M4-BR"),
                # Top edge
                KeepoutCircle(x_mm=6.0,   y_mm=74.0, r_mm=2.0, margin_mm=2.0, label="M4-TL"),
                KeepoutCircle(x_mm=90.0,  y_mm=74.0, r_mm=2.0, margin_mm=2.0, label="M4-TC"),
                KeepoutCircle(x_mm=174.0, y_mm=74.0, r_mm=2.0, margin_mm=2.0, label="M4-TR"),
            ],
        ),
        fluid=FluidSpec(
            coolant="water",
            inlet_temp_C=25.0,
            target_deltaT_C=10.0,
        ),
        generation=GenerationSpec(
            n_candidates=12,
            leaf_counts=[8, 12, 16],
            seed=42,
            weight_pressure=1.0,
            weight_uniformity=1.2,
            weight_manufacturing=0.5,
            v_max_m_per_s=1.5,
        ),
        heatmap_description=(
            "Two 25 x 25 mm hotspots centred at (55, 40) and (125, 40), each "
            "delivering 350 W.  Gaussian profiles, combined peak ~560 kW/m^2."
        ),
    ),

    # -----------------------------------------------------------------------
    # 3. multi_chip_4  --  4-chip module
    # -----------------------------------------------------------------------
    TemplateSpec(
        id="multi_chip_4",
        name="4-Chip Module Cold Plate",
        description=(
            "150 x 150 mm cold plate for a quad-chip module at 1200 W total "
            "(300 W per die).  Four equal hotspots arranged in a 2x2 grid.  "
            "Eight M3 bolt holes around the perimeter.  Inlet at centre-top, "
            "outlet at bottom-right corner."
        ),
        category="multi_chip",
        tags=["multi-chip", "quad", "cnc", "1200W"],
        plate=PlateSpec(
            kind="rectangle",
            width_mm=150.0,
            height_mm=150.0,
        ),
        ports=PortSpec(
            inlet=Point2D(x_mm=75.0, y_mm=145.0),
            outlet=Point2D(x_mm=145.0, y_mm=5.0),
            inlet_diameter_mm=8.0,
            outlet_diameter_mm=8.0,
        ),
        constraints=ConstraintsSpec(
            plate_thickness_mm=6.0,
            channel_depth_mm=2.5,
            min_channel_width_mm=1.0,
            max_channel_width_mm=8.0,
            min_wall_mm=1.0,
            min_bend_radius_mm=3.0,
            grid_resolution_mm=2.0,
            process_preset="CNC",
            keepout_circles=[
                # 8x M3 bolt holes around the perimeter
                # Corners
                KeepoutCircle(x_mm=5.0,   y_mm=5.0,   r_mm=1.5, margin_mm=1.5, label="M3-BL"),
                KeepoutCircle(x_mm=145.0, y_mm=5.0,   r_mm=1.5, margin_mm=1.5, label="M3-BR"),
                KeepoutCircle(x_mm=5.0,   y_mm=145.0, r_mm=1.5, margin_mm=1.5, label="M3-TL"),
                KeepoutCircle(x_mm=145.0, y_mm=145.0, r_mm=1.5, margin_mm=1.5, label="M3-TR"),
                # Mid-edges
                KeepoutCircle(x_mm=75.0,  y_mm=5.0,   r_mm=1.5, margin_mm=1.5, label="M3-BC"),
                KeepoutCircle(x_mm=75.0,  y_mm=145.0, r_mm=1.5, margin_mm=1.5, label="M3-TC"),
                KeepoutCircle(x_mm=5.0,   y_mm=75.0,  r_mm=1.5, margin_mm=1.5, label="M3-ML"),
                KeepoutCircle(x_mm=145.0, y_mm=75.0,  r_mm=1.5, margin_mm=1.5, label="M3-MR"),
            ],
        ),
        fluid=FluidSpec(
            coolant="water",
            inlet_temp_C=25.0,
            target_deltaT_C=12.0,
        ),
        generation=GenerationSpec(
            n_candidates=15,
            leaf_counts=[12, 16, 20],
            seed=42,
            weight_pressure=1.0,
            weight_uniformity=1.5,
            weight_manufacturing=0.5,
            v_max_m_per_s=1.8,
        ),
        heatmap_description=(
            "Four 20 x 20 mm hotspots in a 2x2 grid centred at (50, 50), "
            "(100, 50), (50, 100), (100, 100).  Each die delivers 300 W "
            "(peak flux ~750 kW/m^2)."
        ),
    ),

    # -----------------------------------------------------------------------
    # 4. vrm_strip  --  VRM strip cooler
    # -----------------------------------------------------------------------
    TemplateSpec(
        id="vrm_strip",
        name="VRM Strip Cooler",
        description=(
            "200 x 30 mm narrow cold plate for a VRM strip at 120 W with "
            "near-uniform heat distribution.  Inlet at the left end, outlet "
            "at the right end.  Tight channel constraints suit the narrow "
            "aspect ratio."
        ),
        category="vrm",
        tags=["vrm", "strip", "cnc", "120W", "narrow"],
        plate=PlateSpec(
            kind="rectangle",
            width_mm=200.0,
            height_mm=30.0,
        ),
        ports=PortSpec(
            inlet=Point2D(x_mm=5.0, y_mm=15.0),
            outlet=Point2D(x_mm=195.0, y_mm=15.0),
            inlet_diameter_mm=4.0,
            outlet_diameter_mm=4.0,
        ),
        constraints=ConstraintsSpec(
            plate_thickness_mm=4.0,
            channel_depth_mm=1.5,
            min_channel_width_mm=0.8,
            max_channel_width_mm=4.0,
            min_wall_mm=0.8,
            min_bend_radius_mm=2.0,
            grid_resolution_mm=1.5,
            process_preset="CNC",
            keepout_circles=[],
        ),
        fluid=FluidSpec(
            coolant="water",
            inlet_temp_C=25.0,
            target_deltaT_C=8.0,
        ),
        generation=GenerationSpec(
            n_candidates=10,
            leaf_counts=[6, 8, 10],
            seed=42,
            weight_pressure=0.8,
            weight_uniformity=1.0,
            weight_manufacturing=0.6,
            v_max_m_per_s=1.2,
        ),
        heatmap_description=(
            "Near-uniform 120 W spread across the full 200 x 30 mm footprint "
            "(~20 kW/m^2).  Slight hot-spots at individual MOSFET locations."
        ),
    ),

    # -----------------------------------------------------------------------
    # 5. power_module  --  Power electronics module
    # -----------------------------------------------------------------------
    TemplateSpec(
        id="power_module",
        name="Power Electronics Module Cold Plate",
        description=(
            "120 x 80 mm cold plate for a high-power electronics module "
            "dissipating 800 W.  A single large hotspot is offset toward the "
            "upper-left quadrant.  Four M5 bolt holes at the corners.  "
            "AM (additive manufacturing) preset enables complex internal "
            "channel geometries."
        ),
        category="power",
        tags=["power", "igbt", "am", "800W", "additive"],
        plate=PlateSpec(
            kind="rectangle",
            width_mm=120.0,
            height_mm=80.0,
        ),
        ports=PortSpec(
            inlet=Point2D(x_mm=5.0, y_mm=40.0),
            outlet=Point2D(x_mm=115.0, y_mm=40.0),
            inlet_diameter_mm=8.0,
            outlet_diameter_mm=8.0,
        ),
        constraints=ConstraintsSpec(
            plate_thickness_mm=8.0,
            channel_depth_mm=3.0,
            min_channel_width_mm=0.6,
            max_channel_width_mm=6.0,
            min_wall_mm=0.5,
            min_bend_radius_mm=1.5,
            grid_resolution_mm=1.5,
            process_preset="AM",
            keepout_circles=[
                # 4x M5 bolt holes (2.5 mm radius + 2.5 mm margin)
                KeepoutCircle(x_mm=8.0,   y_mm=8.0,  r_mm=2.5, margin_mm=2.5, label="M5-BL"),
                KeepoutCircle(x_mm=112.0, y_mm=8.0,  r_mm=2.5, margin_mm=2.5, label="M5-BR"),
                KeepoutCircle(x_mm=8.0,   y_mm=72.0, r_mm=2.5, margin_mm=2.5, label="M5-TL"),
                KeepoutCircle(x_mm=112.0, y_mm=72.0, r_mm=2.5, margin_mm=2.5, label="M5-TR"),
            ],
        ),
        fluid=FluidSpec(
            coolant="ethylene_glycol_50",
            inlet_temp_C=30.0,
            target_deltaT_C=15.0,
        ),
        generation=GenerationSpec(
            n_candidates=12,
            leaf_counts=[10, 14, 18],
            seed=42,
            weight_pressure=0.8,
            weight_uniformity=1.0,
            weight_manufacturing=0.3,
            v_max_m_per_s=2.0,
        ),
        heatmap_description=(
            "Single 40 x 40 mm hotspot centred at (40, 50) delivering 800 W "
            "(peak flux ~500 kW/m^2).  Offset toward the upper-left quadrant."
        ),
    ),

    # -----------------------------------------------------------------------
    # 6. nvidia_gb200  --  High-power AI accelerator
    # -----------------------------------------------------------------------
    TemplateSpec(
        id="nvidia_gb200",
        name="High-Power AI Accelerator Cold Plate",
        description=(
            "200 x 120 mm cold plate for a high-power AI accelerator package "
            "at 1800 W.  Extreme heat density demands aggressive leaf counts "
            "and a fine grid resolution.  Eight M4 bolt holes surround the "
            "perimeter.  Designed for CNC with tight tolerances."
        ),
        category="ai_accelerator",
        tags=["gpu", "ai", "hpc", "cnc", "1800W", "high-density"],
        plate=PlateSpec(
            kind="rectangle",
            width_mm=200.0,
            height_mm=120.0,
        ),
        ports=PortSpec(
            inlet=Point2D(x_mm=5.0, y_mm=60.0),
            outlet=Point2D(x_mm=195.0, y_mm=60.0),
            inlet_diameter_mm=10.0,
            outlet_diameter_mm=10.0,
        ),
        constraints=ConstraintsSpec(
            plate_thickness_mm=8.0,
            channel_depth_mm=3.0,
            min_channel_width_mm=1.0,
            max_channel_width_mm=10.0,
            min_wall_mm=0.8,
            min_bend_radius_mm=2.5,
            grid_resolution_mm=1.5,
            process_preset="CNC",
            keepout_circles=[
                # 8x M4 bolt holes (2.0 mm radius + 2.0 mm margin)
                # Corners
                KeepoutCircle(x_mm=6.0,   y_mm=6.0,   r_mm=2.0, margin_mm=2.0, label="M4-BL"),
                KeepoutCircle(x_mm=194.0, y_mm=6.0,   r_mm=2.0, margin_mm=2.0, label="M4-BR"),
                KeepoutCircle(x_mm=6.0,   y_mm=114.0, r_mm=2.0, margin_mm=2.0, label="M4-TL"),
                KeepoutCircle(x_mm=194.0, y_mm=114.0, r_mm=2.0, margin_mm=2.0, label="M4-TR"),
                # Mid-edges
                KeepoutCircle(x_mm=100.0, y_mm=6.0,   r_mm=2.0, margin_mm=2.0, label="M4-BC"),
                KeepoutCircle(x_mm=100.0, y_mm=114.0, r_mm=2.0, margin_mm=2.0, label="M4-TC"),
                KeepoutCircle(x_mm=6.0,   y_mm=60.0,  r_mm=2.0, margin_mm=2.0, label="M4-ML"),
                KeepoutCircle(x_mm=194.0, y_mm=60.0,  r_mm=2.0, margin_mm=2.0, label="M4-MR"),
            ],
        ),
        fluid=FluidSpec(
            coolant="water",
            inlet_temp_C=20.0,
            target_deltaT_C=8.0,
        ),
        generation=GenerationSpec(
            n_candidates=20,
            leaf_counts=[16, 20, 24, 32],
            seed=42,
            weight_pressure=1.0,
            weight_uniformity=1.5,
            weight_manufacturing=0.4,
            v_max_m_per_s=2.5,
        ),
        heatmap_description=(
            "Large die footprint (~50 x 50 mm) centred on the plate delivering "
            "1800 W (peak flux ~720 kW/m^2).  Surrounding substrate dissipates "
            "an additional ~200 W as background."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Index for fast ID lookup
# ---------------------------------------------------------------------------
_TEMPLATE_INDEX = {t.id: t for t in _TEMPLATES}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_template(template_id: str) -> TemplateSpec:
    """Return a single template by its unique *template_id*.

    Raises ``KeyError`` if the ID is not found.
    """
    try:
        return _TEMPLATE_INDEX[template_id]
    except KeyError:
        available = ", ".join(sorted(_TEMPLATE_INDEX))
        raise KeyError(
            f"Unknown template '{template_id}'.  "
            f"Available templates: {available}"
        ) from None


def list_templates() -> List[TemplateSpec]:
    """Return every registered template, in definition order."""
    return list(_TEMPLATES)


def get_templates_by_category(category: str) -> List[TemplateSpec]:
    """Return all templates whose *category* matches (case-insensitive)."""
    cat_lower = category.lower()
    return [t for t in _TEMPLATES if t.category.lower() == cat_lower]
