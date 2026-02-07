from __future__ import annotations

from typing import Tuple
import csv
import numpy as np
from PIL import Image


def load_csv_grid(path: str) -> np.ndarray:
    # Accepts a plain numeric grid CSV (no headers).
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            vals = []
            for cell in row:
                cell = cell.strip()
                if not cell:
                    continue
                try:
                    vals.append(float(cell))
                except ValueError:
                    # ignore non-numeric (headers)
                    pass
            if vals:
                rows.append(vals)
    if not rows:
        raise ValueError("CSV heatmap appears empty or non-numeric.")
    # Pad ragged rows
    maxlen = max(len(r) for r in rows)
    rows2 = [r + [0.0]*(maxlen-len(r)) for r in rows]
    return np.array(rows2, dtype=float)


def load_image_grid(path: str, flip_y: bool = True) -> np.ndarray:
    img = Image.open(path).convert("L")  # grayscale
    arr = np.asarray(img, dtype=float) / 255.0
    if flip_y:
        arr = np.flipud(arr)
    return arr


def normalize_to_total(arr: np.ndarray, total_watts: float) -> np.ndarray:
    total = float(arr.sum())
    if total <= 0:
        # avoid divide by zero; make uniform
        return np.full_like(arr, fill_value=(total_watts / arr.size))
    return arr * (float(total_watts) / total)


def grid_cell_centers(plate_w_mm: float, plate_h_mm: float, grid: np.ndarray):
    ny, nx = grid.shape
    xs = (np.arange(nx) + 0.5) * (plate_w_mm / nx)
    ys = (np.arange(ny) + 0.5) * (plate_h_mm / ny)
    # produce arrays of shape (ny, nx, 2)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X, Y], axis=-1)
    return pts
