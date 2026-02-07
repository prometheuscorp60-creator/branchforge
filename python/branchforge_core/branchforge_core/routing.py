from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import heapq

import numpy as np
from shapely.geometry import Point, Polygon


@dataclass
class GridSpec:
    origin_x: float
    origin_y: float
    resolution: float
    nx: int
    ny: int


class GridRouter:
    def __init__(self, allowed: Polygon, resolution_mm: float = 2.0):
        self.allowed = allowed
        minx, miny, maxx, maxy = allowed.bounds
        pad = resolution_mm * 2
        minx -= pad
        miny -= pad
        maxx += pad
        maxy += pad

        nx = int(math.ceil((maxx - minx) / resolution_mm))
        ny = int(math.ceil((maxy - miny) / resolution_mm))
        self.spec = GridSpec(minx, miny, resolution_mm, nx, ny)

        self.blocked = np.zeros((ny, nx), dtype=bool)
        self.allowed_mask = np.zeros((ny, nx), dtype=bool)

        # build allowed mask
        for j in range(ny):
            y = miny + (j + 0.5) * resolution_mm
            for i in range(nx):
                x = minx + (i + 0.5) * resolution_mm
                if allowed.contains(Point(x, y)):
                    self.allowed_mask[j, i] = True

    def point_to_cell(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        x, y = xy
        i = int((x - self.spec.origin_x) / self.spec.resolution)
        j = int((y - self.spec.origin_y) / self.spec.resolution)
        i = max(0, min(self.spec.nx - 1, i))
        j = max(0, min(self.spec.ny - 1, j))
        return (i, j)

    def cell_to_point(self, ij: Tuple[int, int]) -> Tuple[float, float]:
        i, j = ij
        x = self.spec.origin_x + (i + 0.5) * self.spec.resolution
        y = self.spec.origin_y + (j + 0.5) * self.spec.resolution
        return (x, y)

    def is_free(self, ij: Tuple[int, int]) -> bool:
        i, j = ij
        return bool(self.allowed_mask[j, i] and (not self.blocked[j, i]))

    def nearest_free(self, ij: Tuple[int, int], max_r: int = 10) -> Tuple[int, int]:
        if self.is_free(ij):
            return ij
        ci, cj = ij
        for r in range(1, max_r + 1):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    ni, nj = ci + di, cj + dj
                    if ni < 0 or ni >= self.spec.nx or nj < 0 or nj >= self.spec.ny:
                        continue
                    if self.is_free((ni, nj)):
                        return (ni, nj)
        return ij

    def add_block_radius(self, xy: Tuple[float, float], radius_mm: float):
        # mark grid cells within radius_mm as blocked
        r = float(radius_mm)
        if r <= 0:
            return
        i0, j0 = self.point_to_cell(xy)
        rad_cells = int(math.ceil(r / self.spec.resolution))
        for dj in range(-rad_cells, rad_cells + 1):
            for di in range(-rad_cells, rad_cells + 1):
                ni, nj = i0 + di, j0 + dj
                if ni < 0 or ni >= self.spec.nx or nj < 0 or nj >= self.spec.ny:
                    continue
                x, y = self.cell_to_point((ni, nj))
                if (x - xy[0])**2 + (y - xy[1])**2 <= r**2:
                    self.blocked[nj, ni] = True

    def add_path_obstacle(self, path: List[Tuple[float, float]], radius_mm: float):
        for p in path:
            self.add_block_radius(p, radius_mm)

    def route(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float], turn_penalty: float = 0.1) -> Optional[List[Tuple[float, float]]]:
        start = self.nearest_free(self.point_to_cell(start_xy))
        goal = self.nearest_free(self.point_to_cell(goal_xy))
        if not self.is_free(start) or not self.is_free(goal):
            return None

        # A* with direction in state to penalize turns
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        move_cost = [1,1,1,1,math.sqrt(2),math.sqrt(2),math.sqrt(2),math.sqrt(2)]

        def heuristic(a: Tuple[int,int], b: Tuple[int,int]) -> float:
            dx = a[0]-b[0]
            dy = a[1]-b[1]
            return math.hypot(dx, dy)

        # state: (i,j,dir_index) where dir_index is last move direction (0..7) or 8 for none
        NONE_DIR = 8
        start_state = (start[0], start[1], NONE_DIR)
        goal_cell = goal

        open_heap = []
        heapq.heappush(open_heap, (0.0, start_state))
        gscore: Dict[Tuple[int,int,int], float] = {start_state: 0.0}
        came: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}

        while open_heap:
            _, state = heapq.heappop(open_heap)
            i, j, last_dir = state
            if (i, j) == goal_cell:
                # reconstruct (choose best dir at goal)
                best_state = state
                path_cells = []
                while True:
                    path_cells.append((best_state[0], best_state[1]))
                    if best_state == start_state:
                        break
                    best_state = came[best_state]
                path_cells.reverse()
                return [self.cell_to_point(c) for c in path_cells]

            for k, (di, dj) in enumerate(neighbors):
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= self.spec.nx or nj < 0 or nj >= self.spec.ny:
                    continue
                if not self.is_free((ni, nj)):
                    continue
                cost = move_cost[k]
                if last_dir != NONE_DIR and k != last_dir:
                    cost += turn_penalty
                nstate = (ni, nj, k)
                tentative = gscore[state] + cost
                if tentative < gscore.get(nstate, 1e18):
                    came[nstate] = state
                    gscore[nstate] = tentative
                    f = tentative + heuristic((ni, nj), goal_cell)
                    heapq.heappush(open_heap, (f, nstate))

        return None


def rdp_simplify(points: List[Tuple[float,float]], epsilon: float) -> List[Tuple[float,float]]:
    # Ramer-Douglas-Peucker simplification
    if len(points) < 3:
        return points

    def perp_dist(p, a, b):
        # distance point p to line segment ab
        ax, ay = a
        bx, by = b
        px, py = p
        dx = bx-ax
        dy = by-ay
        if dx == 0 and dy == 0:
            return math.hypot(px-ax, py-ay)
        t = ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        x = ax + t*dx
        y = ay + t*dy
        return math.hypot(px-x, py-y)

    def rec(pts):
        a = pts[0]
        b = pts[-1]
        maxd = -1.0
        idx = -1
        for i in range(1, len(pts)-1):
            d = perp_dist(pts[i], a, b)
            if d > maxd:
                maxd = d
                idx = i
        if maxd <= epsilon:
            return [a, b]
        left = rec(pts[:idx+1])
        right = rec(pts[idx:])
        return left[:-1] + right

    out = rec(points)
    # remove duplicate adjacent
    cleaned = [out[0]]
    for p in out[1:]:
        if (p[0]-cleaned[-1][0])**2 + (p[1]-cleaned[-1][1])**2 > 1e-9:
            cleaned.append(p)
    return cleaned
