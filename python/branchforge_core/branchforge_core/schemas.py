from __future__ import annotations

from typing import Literal, Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict


class Point2D(BaseModel):
    x_mm: float
    y_mm: float


class PlateSpec(BaseModel):
    kind: Literal["rectangle", "polygon"] = "rectangle"
    width_mm: float = 100.0
    height_mm: float = 60.0
    polygon: Optional[List[Point2D]] = None
    dxf_path: Optional[str] = None  # optional DXF outline import


class PortSpec(BaseModel):
    inlet: Point2D = Point2D(x_mm=5.0, y_mm=30.0)
    outlet: Point2D = Point2D(x_mm=95.0, y_mm=30.0)
    inlet_diameter_mm: float = 6.0
    outlet_diameter_mm: float = 6.0


class HeatmapSpec(BaseModel):
    kind: Literal["csv", "image"] = "csv"
    # Worker-local path to file
    path: str
    total_watts: float = 1000.0
    flip_y: bool = True  # images typically have origin top-left; we want bottom-left


class KeepoutRect(BaseModel):
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    # If True, x/y are center; else lower-left
    centered: bool = True
    margin_mm: float = 0.0


class KeepoutCircle(BaseModel):
    x_mm: float
    y_mm: float
    r_mm: float
    margin_mm: float = 0.0


class ConstraintsSpec(BaseModel):
    plate_thickness_mm: float = 5.0
    channel_depth_mm: float = 2.0
    min_channel_width_mm: float = 1.2
    max_channel_width_mm: float = 8.0
    min_wall_mm: float = 1.0
    min_bend_radius_mm: float = 3.0
    grid_resolution_mm: float = 2.0

    keepout_rects: List[KeepoutRect] = Field(default_factory=list)
    keepout_circles: List[KeepoutCircle] = Field(default_factory=list)

    process_preset: Literal["CNC", "AM", "ETCHED_LID"] = "CNC"


class FluidSpec(BaseModel):
    coolant: Literal["water"] = "water"
    inlet_temp_C: float = 25.0
    target_deltaT_C: float = 10.0


class GenerationSpec(BaseModel):
    n_candidates: int = 10
    leaf_counts: List[int] = Field(default_factory=lambda: [8, 12, 16])
    seed: int = 42

    # Higher means more penalty / preference
    weight_pressure: float = 1.0
    weight_uniformity: float = 1.0
    weight_manufacturing: float = 0.5

    # derived sizing knobs
    v_max_m_per_s: float = 1.5


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plate: PlateSpec
    ports: PortSpec
    heatmap: HeatmapSpec
    constraints: ConstraintsSpec = ConstraintsSpec()
    fluid: FluidSpec = FluidSpec()
    generation: GenerationSpec = GenerationSpec()


class CandidateMetrics(BaseModel):
    delta_p_kpa: float
    uniformity_deltaT_C_std: float
    uniformity_deltaT_C_max: float
    manufacturing_score: float

    n_leaves: int
    total_mass_flow_g_per_s: float
    total_watts: float


class CandidateArtifacts(BaseModel):
    preview_png: str
    report_pdf: str
    plate_step: str
    channels_step: str
    plate_stl: str
    channels_dxf: str

    # Debug
    json_paths: str


class CandidateSummary(BaseModel):
    index: int
    label: str
    metrics: CandidateMetrics
    artifacts: CandidateArtifacts

    # For front-end overlays (optional lightweight geometry)
    overlay: Optional[Dict[str, Any]] = None
