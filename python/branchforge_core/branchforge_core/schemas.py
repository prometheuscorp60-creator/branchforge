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
    dxf_path: Optional[str] = None


class PortSpec(BaseModel):
    inlet: Point2D = Point2D(x_mm=5.0, y_mm=30.0)
    outlet: Point2D = Point2D(x_mm=95.0, y_mm=30.0)
    inlet_diameter_mm: float = 6.0
    outlet_diameter_mm: float = 6.0


class HeatmapSpec(BaseModel):
    kind: Literal["csv", "image"] = "csv"
    path: str
    total_watts: float = 1000.0
    flip_y: bool = True


class KeepoutRect(BaseModel):
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    centered: bool = True
    margin_mm: float = 0.0
    label: str = ""


class KeepoutCircle(BaseModel):
    x_mm: float
    y_mm: float
    r_mm: float
    margin_mm: float = 0.0
    label: str = ""


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
    coolant: str = "water"
    inlet_temp_C: float = 25.0
    target_deltaT_C: float = 10.0

    # Custom fluid overrides (used when coolant == "custom")
    custom_rho_kg_m3: Optional[float] = None
    custom_mu_pa_s: Optional[float] = None
    custom_cp_J_kgK: Optional[float] = None


class GenerationSpec(BaseModel):
    n_candidates: int = 10
    leaf_counts: List[int] = Field(default_factory=lambda: [8, 12, 16])
    seed: int = 42

    weight_pressure: float = 1.0
    weight_uniformity: float = 1.0
    weight_manufacturing: float = 0.5
    weight_surface_area: float = 0.3

    v_max_m_per_s: float = 1.5


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plate: PlateSpec
    ports: PortSpec
    heatmap: HeatmapSpec
    constraints: ConstraintsSpec = ConstraintsSpec()
    fluid: FluidSpec = FluidSpec()
    generation: GenerationSpec = GenerationSpec()

    template_id: Optional[str] = None


class CandidateMetrics(BaseModel):
    delta_p_kpa: float
    uniformity_deltaT_C_std: float
    uniformity_deltaT_C_max: float
    manufacturing_score: float

    n_leaves: int
    total_mass_flow_g_per_s: float
    total_watts: float

    reynolds_number_max: float = 0.0
    channel_area_mm2: float = 0.0
    total_channel_length_mm: float = 0.0

    # Surface-minimization metrics (Meng et al. Nature 649, 315-322, 2026)
    surface_area_mm2: float = 0.0
    channel_volume_mm3: float = 0.0
    surface_to_volume_ratio: float = 0.0
    murray_law_deviation: float = 0.0

    validation_passed: bool = True
    validation_warnings: int = 0
    validation_errors: int = 0


class CandidateArtifacts(BaseModel):
    preview_png: str
    report_pdf: str
    plate_step: str
    channels_step: str
    plate_stl: str
    channels_dxf: str
    json_paths: str


class CandidateSummary(BaseModel):
    index: int
    label: str
    metrics: CandidateMetrics
    artifacts: CandidateArtifacts
    overlay: Optional[Dict[str, Any]] = None


class JobManifest(BaseModel):
    job_id: str = ""
    spec: Optional[Dict[str, Any]] = None
    git_version: str = ""
    seed: int = 0
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    n_candidates_requested: int = 0
    n_candidates_produced: int = 0
    n_candidates_failed: int = 0
    candidate_timings: List[Dict[str, Any]] = Field(default_factory=list)
    validation_summary: Dict[str, Any] = Field(default_factory=dict)


class TemplateSpec(BaseModel):
    id: str
    name: str
    description: str
    category: str
    plate: PlateSpec
    ports: PortSpec
    constraints: ConstraintsSpec
    fluid: FluidSpec = FluidSpec()
    generation: GenerationSpec = GenerationSpec()
    heatmap_description: str = ""
    tags: List[str] = Field(default_factory=list)
