"""
Auto-generate heatmap grids for built-in templates.

Each template defines a thermal profile programmatically so users
can run "one-click" template jobs without uploading a heatmap file.

Heatmaps are generated as numpy arrays (ny, nx) in units of watts.
The grid resolution is derived from the plate dimensions to produce
~50x50 cells for quick generation.
"""

from __future__ import annotations

import math
from typing import Tuple, List, Optional
import numpy as np

from .heatmap import normalize_to_total


# ---------------------------------------------------------------------------
# Gaussian hotspot builder
# ---------------------------------------------------------------------------

def _gaussian_hotspot(
    plate_w_mm: float,
    plate_h_mm: float,
    nx: int,
    ny: int,
    cx_mm: float,
    cy_mm: float,
    sigma_x_mm: float,
    sigma_y_mm: float,
    peak_watts: float,
) -> np.ndarray:
    """Create a 2D Gaussian hotspot centred at (cx, cy) mm.

    Returns array (ny, nx) with values in watts.
    """
    xs = (np.arange(nx) + 0.5) * (plate_w_mm / nx)
    ys = (np.arange(ny) + 0.5) * (plate_h_mm / ny)
    X, Y = np.meshgrid(xs, ys)

    exponent = -(
        ((X - cx_mm) ** 2) / (2 * sigma_x_mm ** 2) +
        ((Y - cy_mm) ** 2) / (2 * sigma_y_mm ** 2)
    )
    raw = np.exp(exponent)

    # Normalize so total watts = peak_watts
    raw_sum = raw.sum()
    if raw_sum > 0:
        raw = raw * (peak_watts / raw_sum)

    return raw


def _uniform_background(
    nx: int, ny: int,
    total_watts: float,
) -> np.ndarray:
    """Uniform heat spread across entire plate."""
    return np.full((ny, nx), total_watts / (nx * ny))


# ---------------------------------------------------------------------------
# Template-specific heatmap generators
# ---------------------------------------------------------------------------

def _heatmap_gpu_single(plate_w: float, plate_h: float, nx: int, ny: int) -> np.ndarray:
    """100x60mm single GPU: 350W centre hotspot."""
    return _gaussian_hotspot(
        plate_w, plate_h, nx, ny,
        cx_mm=50.0, cy_mm=30.0,
        sigma_x_mm=12.5, sigma_y_mm=12.5,
        peak_watts=350.0,
    )


def _heatmap_gpu_dual(plate_w: float, plate_h: float, nx: int, ny: int) -> np.ndarray:
    """180x80mm dual GPU: 2x 350W hotspots side-by-side."""
    h1 = _gaussian_hotspot(
        plate_w, plate_h, nx, ny,
        cx_mm=55.0, cy_mm=40.0,
        sigma_x_mm=12.5, sigma_y_mm=12.5,
        peak_watts=350.0,
    )
    h2 = _gaussian_hotspot(
        plate_w, plate_h, nx, ny,
        cx_mm=125.0, cy_mm=40.0,
        sigma_x_mm=12.5, sigma_y_mm=12.5,
        peak_watts=350.0,
    )
    return h1 + h2


def _heatmap_multi_chip_4(plate_w: float, plate_h: float, nx: int, ny: int) -> np.ndarray:
    """150x150mm quad-chip: 4x 300W in 2x2 grid."""
    grid = np.zeros((ny, nx))
    centers = [(50, 50), (100, 50), (50, 100), (100, 100)]
    for cx, cy in centers:
        grid += _gaussian_hotspot(
            plate_w, plate_h, nx, ny,
            cx_mm=cx, cy_mm=cy,
            sigma_x_mm=10.0, sigma_y_mm=10.0,
            peak_watts=300.0,
        )
    return grid


def _heatmap_vrm_strip(plate_w: float, plate_h: float, nx: int, ny: int) -> np.ndarray:
    """200x30mm VRM strip: near-uniform 120W with slight MOSFET hotspots."""
    base = _uniform_background(nx, ny, 80.0)

    # Add 8 small hotspots for individual MOSFETs
    n_mosfets = 8
    for i in range(n_mosfets):
        x = 12.5 + i * (175.0 / (n_mosfets - 1))
        base += _gaussian_hotspot(
            plate_w, plate_h, nx, ny,
            cx_mm=x, cy_mm=15.0,
            sigma_x_mm=5.0, sigma_y_mm=5.0,
            peak_watts=5.0,
        )
    return base


def _heatmap_power_module(plate_w: float, plate_h: float, nx: int, ny: int) -> np.ndarray:
    """120x80mm power module: 800W single hotspot offset upper-left."""
    return _gaussian_hotspot(
        plate_w, plate_h, nx, ny,
        cx_mm=40.0, cy_mm=50.0,
        sigma_x_mm=20.0, sigma_y_mm=20.0,
        peak_watts=800.0,
    )


def _heatmap_nvidia_gb200(plate_w: float, plate_h: float, nx: int, ny: int) -> np.ndarray:
    """200x120mm AI accelerator: 1800W die + 200W substrate background."""
    die = _gaussian_hotspot(
        plate_w, plate_h, nx, ny,
        cx_mm=100.0, cy_mm=60.0,
        sigma_x_mm=25.0, sigma_y_mm=25.0,
        peak_watts=1800.0,
    )
    substrate = _uniform_background(nx, ny, 200.0)
    return die + substrate


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_HEATMAP_GENERATORS = {
    "gpu_single": _heatmap_gpu_single,
    "gpu_dual": _heatmap_gpu_dual,
    "multi_chip_4": _heatmap_multi_chip_4,
    "vrm_strip": _heatmap_vrm_strip,
    "power_module": _heatmap_power_module,
    "nvidia_gb200": _heatmap_nvidia_gb200,
}


def generate_template_heatmap(
    template_id: str,
    plate_w_mm: float,
    plate_h_mm: float,
    resolution: int = 50,
) -> np.ndarray:
    """Generate a heatmap grid for a template.

    Parameters
    ----------
    template_id : str
        Must match a key in the template registry.
    plate_w_mm, plate_h_mm : float
        Plate dimensions in mm.
    resolution : int
        Approximate number of cells along the longer axis.

    Returns
    -------
    np.ndarray
        Heatmap in watts, shape (ny, nx).
    """
    gen = _HEATMAP_GENERATORS.get(template_id)
    if gen is None:
        raise KeyError(
            f"No heatmap generator for template '{template_id}'. "
            f"Available: {sorted(_HEATMAP_GENERATORS.keys())}"
        )

    aspect = plate_w_mm / max(plate_h_mm, 1e-6)
    if aspect >= 1.0:
        nx = resolution
        ny = max(10, int(round(resolution / aspect)))
    else:
        ny = resolution
        nx = max(10, int(round(resolution * aspect)))

    grid = gen(plate_w_mm, plate_h_mm, nx, ny)

    # Ensure non-negative
    grid = np.maximum(grid, 0.0)

    return grid


def save_heatmap_csv(grid: np.ndarray, path: str) -> None:
    """Save a heatmap grid as a CSV file (compatible with load_csv_grid)."""
    np.savetxt(path, grid, delimiter=",", fmt="%.4f")


def list_template_heatmaps() -> List[str]:
    """Return template IDs that have auto-generated heatmaps."""
    return sorted(_HEATMAP_GENERATORS.keys())
