#!/usr/bin/env python3
"""
occupancy_replan.py — take an occupancy grid + OmniVLA's action chunk and, if the chunk
would hit an obstacle, REPLAN a collision-free local trajectory toward the direction
Omni intended (A* on the grid to a look-ahead subgoal along Omni's bearing).

Architecture: Omni supplies the goal-directed *intent* (its chunk / bearing); this module
supplies *local safety* (obstacle-aware routing). An occupancy grid alone can only veto;
the A* planner on top is what actually reroutes.

Robot frame: x = forward (+), y = left (+). Grid is robot-centric.

This file has NO model / depth / GPU dependency — the grid is passed in. `occupancy_from_depth`
is a stub for wiring Depth Anything V2 (metric) later; run the __main__ demo to see the
planner work on a synthetic obstacle today.
"""
import heapq
from collections import deque

import numpy as np

# matplotlib is only used by render() (debug plots). It is imported lazily inside that function
# because the Isaac Sim container's Python does not ship it, and the grid/replanner must import
# cleanly there — that is the code path the live pipeline actually uses.


class Grid:
    """Robot-centric 2D occupancy grid. rows index x (forward), cols index y (left)."""

    def __init__(self, x_max=4.0, y_half=2.0, res=0.05):
        self.res = res
        self.x_max = x_max
        self.y_half = y_half
        self.y_min = -y_half
        self.nx = int(round(x_max / res))
        self.ny = int(round(2 * y_half / res))
        self.occ = np.zeros((self.nx, self.ny), dtype=bool)

    def xy_to_cell(self, x, y):
        return int(round(x / self.res)), int(round((y - self.y_min) / self.res))

    def cell_to_xy(self, r, c):
        return r * self.res, self.y_min + c * self.res

    def in_bounds(self, r, c):
        return 0 <= r < self.nx and 0 <= c < self.ny

    def add_box(self, x0, x1, y0, y1):
        r0, c0 = self.xy_to_cell(x0, y0)
        r1, c1 = self.xy_to_cell(x1, y1)
        self.occ[min(r0, r1):max(r0, r1) + 1, min(c0, c1):max(c0, c1) + 1] = True

    def inflate(self, radius_m):
        """Grow obstacles by the robot radius so the planned path keeps clearance."""
        rad = int(round(radius_m / self.res))
        if rad <= 0:
            return self.occ.copy()
        out = self.occ.copy()
        rr, cc = np.where(self.occ)
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if dr * dr + dc * dc > rad * rad:
                    continue
                r2, c2 = rr + dr, cc + dc
                m = (r2 >= 0) & (r2 < self.nx) & (c2 >= 0) & (c2 < self.ny)
                out[r2[m], c2[m]] = True
        return out


def _nearest_free(occ, cell):
    if not occ[cell]:
        return cell
    nx, ny = occ.shape
    seen = {cell}
    q = deque([cell])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < nx and 0 <= nc < ny and (nr, nc) not in seen:
                if not occ[nr, nc]:
                    return (nr, nc)
                seen.add((nr, nc))
                q.append((nr, nc))
    return None


def astar(occ, start, goal):
    """8-connected A* on a boolean occupancy grid. Returns list of cells or None."""
    nx, ny = occ.shape
    if not (0 <= goal[0] < nx and 0 <= goal[1] < ny):
        return None
    goal = _nearest_free(occ, goal)
    if goal is None:
        return None
    if occ[start]:
        start = _nearest_free(occ, start) or start

    def h(a):
        return ((a[0] - goal[0]) ** 2 + (a[1] - goal[1]) ** 2) ** 0.5

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    openq = [(h(start), 0.0, start)]
    came = {start: None}
    gscore = {start: 0.0}
    while openq:
        _f, g, cur = heapq.heappop(openq)
        if cur == goal:
            path = []
            n = cur
            while n is not None:
                path.append(n)
                n = came[n]
            return path[::-1]
        for dr, dc in nbrs:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < nx and 0 <= nc < ny) or occ[nr, nc]:
                continue
            ng = g + (dr * dr + dc * dc) ** 0.5
            if ng < gscore.get((nr, nc), 1e18):
                gscore[(nr, nc)] = ng
                came[(nr, nc)] = cur
                heapq.heappush(openq, (ng + h((nr, nc)), ng, (nr, nc)))
    return None


def chunk_is_free(grid, occ_inflated, wp):
    for x, y in wp:
        r, c = grid.xy_to_cell(x, y)
        if grid.in_bounds(r, c) and occ_inflated[r, c]:
            return False
    return True


def _subgoal_from_chunk(wp, lookahead):
    """Extend Omni's chunk bearing out to `lookahead` meters as the replanning subgoal,
    so the planner has room to route around obstacles toward Omni's intended direction."""
    end = np.asarray(wp[-1], float)
    n = np.hypot(*end)
    return np.array([lookahead, 0.0]) if n < 1e-6 else end / n * lookahead


def refine(grid, wp, robot_radius=0.20, lookahead=2.5, n_out=8):
    """Return (used_omni, refined_xy, occ_inflated, subgoal).
    If Omni's chunk is collision-free -> use it as-is (used_omni=True, refined_xy=wp).
    Else -> A* to the look-ahead subgoal, downsampled to n_out points (used_omni=False)."""
    occ_inf = grid.inflate(robot_radius)
    wp = np.asarray(wp, float).reshape(-1, 2)
    if chunk_is_free(grid, occ_inf, wp):
        return True, wp, occ_inf, None

    subgoal = _subgoal_from_chunk(wp, min(lookahead, grid.x_max - grid.res))
    path_cells = astar(occ_inf, grid.xy_to_cell(0.0, 0.0),
                       grid.xy_to_cell(subgoal[0], subgoal[1]))
    if not path_cells:
        return False, None, occ_inf, subgoal          # no safe path -> caller should stop
    path_xy = np.array([grid.cell_to_xy(r, c) for r, c in path_cells])
    if len(path_xy) > n_out:                            # downsample for the controller
        idx = np.linspace(0, len(path_xy) - 1, n_out).round().astype(int)
        path_xy = path_xy[idx]
    return False, path_xy, occ_inf, subgoal


def occupancy_from_depth(depth_m, K, cam_height, cam_pitch_deg=0.0,
                         grid=None, floor_margin=0.12, ceil_margin=0.20,
                         max_range=4.0, min_range=0.15):
    """Metric depth map -> robot-frame occupancy Grid.

    Back-projects each pixel with intrinsics K into the camera frame, rotates by the camera
    pitch, lifts by the camera height to get the point's height above the floor, keeps points
    that are neither floor nor ceiling (i.e. obstacles), and bins their (forward, left) into a
    top-down grid. In Habitat, K and cam_height are known exactly and depth is ground-truth, so
    this is metrically clean (unlike the monocular real-robot case).

    depth_m : (H,W) float depth along the optical axis, in metres.
    K       : 3x3 pinhole intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]].
    cam_height : camera height above the floor (m). cam_pitch_deg: down-tilt (+ = looking down).
    """
    depth_m = np.asarray(depth_m, float)
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_m
    valid = np.isfinite(z) & (z > min_range) & (z < max_range)
    z = z[valid]
    u = us[valid]
    v = vs[valid]
    # pinhole back-projection: camera frame X right, Y down, Z forward
    xc = (u - cx) * z / fx
    yc = (v - cy) * z / fy
    zc = z
    # apply camera pitch about the camera X axis (down-tilt +). Rotate (Y,Z).
    p = np.deg2rad(cam_pitch_deg)
    cp, sp = np.cos(p), np.sin(p)
    y_r = cp * yc - sp * zc
    z_r = sp * yc + cp * zc
    # to robot frame: forward = Z, left = -X, height above floor = cam_height - Y_down
    fwd = z_r
    left = -xc
    height = cam_height - y_r
    obstacle = (height > floor_margin) & (height < cam_height + ceil_margin) & (fwd > min_range)
    fwd, left = fwd[obstacle], left[obstacle]
    g = grid if grid is not None else Grid(x_max=max_range, y_half=2.0, res=0.05)
    rows = np.round(fwd / g.res).astype(int)
    cols = np.round((left - g.y_min) / g.res).astype(int)
    m = (rows >= 0) & (rows < g.nx) & (cols >= 0) & (cols < g.ny)
    g.occ[rows[m], cols[m]] = True
    return g


# ── visualization ───────────────────────────────────────────────────────────────
def render(grid, wp, used_omni, refined_xy, occ_inf, subgoal, out_path,
           title="occupancy replan"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 7))
    extent = [grid.y_min, grid.y_min + grid.ny * grid.res, 0, grid.nx * grid.res]
    ax.imshow(occ_inf.astype(float), origin="lower", extent=extent, cmap="Greys",
              alpha=0.30, aspect="equal")                        # inflated (clearance)
    ax.imshow(grid.occ.astype(float), origin="lower", extent=extent, cmap="Greys",
              alpha=0.75, aspect="equal")                        # true obstacle
    wp = np.asarray(wp, float).reshape(-1, 2)
    ax.plot(-wp[:, 1], wp[:, 0], color="#1f77b4", lw=2.5, marker="o", ms=4,
            label="OmniVLA chunk", zorder=4)
    if not used_omni and refined_xy is not None:
        ax.plot(-refined_xy[:, 1], refined_xy[:, 0], color="#2ca02c", lw=2.8,
                marker="o", ms=4, label="refined (safe)", zorder=5)
    if subgoal is not None:
        ax.scatter([-subgoal[1]], [subgoal[0]], marker="*", s=150, color="#d62728",
                   label="subgoal (Omni bearing)", zorder=6)
    ax.scatter([0], [0], marker="^", s=130, color="k", zorder=7)
    ax.set_xlim(grid.y_min, grid.y_min + grid.ny * grid.res)
    ax.set_ylim(0, grid.nx * grid.res)
    ax.set_xlabel("← right        left →   (m)")
    ax.set_ylabel("forward (m)")
    verdict = "chunk already safe" if used_omni else (
        "replanned around obstacle" if refined_xy is not None else "NO SAFE PATH")
    ax.set_title(f"{title} — {verdict}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import os
    out = "replan_demo"
    os.makedirs(out, exist_ok=True)

    # Scene A: obstacle dead ahead; Omni drives straight into it -> must reroute.
    gA = Grid(x_max=4.0, y_half=2.0, res=0.05)
    gA.add_box(0.6, 1.1, -0.45, 0.45)                # box blocking the center
    wp_straight = np.column_stack([np.linspace(0.1, 0.9, 8), np.zeros(8)])
    used, refined, occ_inf, sub = refine(gA, wp_straight)
    render(gA, wp_straight, used, refined, occ_inf, sub, os.path.join(out, "A_blocked.png"),
           "A: obstacle ahead")
    print(f"A: used_omni={used}  refined_pts={None if refined is None else len(refined)}")

    # Scene B: obstacle off to the right; Omni's straight chunk is already clear -> pass through.
    gB = Grid(x_max=4.0, y_half=2.0, res=0.05)
    gB.add_box(0.5, 1.0, -1.6, -0.9)
    used, refined, occ_inf, sub = refine(gB, wp_straight)
    render(gB, wp_straight, used, refined, occ_inf, sub, os.path.join(out, "B_clear.png"),
           "B: obstacle to the side")
    print(f"B: used_omni={used}  refined_pts={None if refined is None else len(refined)}")

    print(f"figures in {out}/")