#!/usr/bin/env python3
"""MSI-side camera-occupancy server for Trajectory_Generation_and_Control.

Same pipeline as camera_occupancy_node.py, but served over HTTP so the heavy
depth inference runs on an MSI GPU node instead of the Jetson (mirrors the
policy_server.py pattern: Jetson connects through an SSH tunnel).

    POST /calib      JSON {width,height,K,D,R,P}   -> caches rectify maps
    POST /occupancy  raw JPEG bytes                -> occupancy result
         ?fmt=png    debug image only (default; easy to eyeball with curl)
         ?fmt=npz    npz{grid,debug,meta} for the trajectory-refinement layer
    GET  /health     JSON status

All occupancy math is IMPORTED from camera_occupancy_node.py — do not copy
those functions here. One implementation, one test suite.

No ROS on MSI. numpy + opencv + torch + Depth Anything V2 only.
Runs on CPU with --allow-cpu (slow; offline/protocol testing only).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

# Pure math shared with the Jetson node (this file must sit next to it).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_occupancy_node import (
    backproject_depth,
    build_camera_occupancy,
    fit_floor_ransac,
    occupancy_to_debug_image,
)

MODEL_MAP = {
    "small": ("vits", 64, [48, 96, 192, 384]),
    "base": ("vitb", 128, [96, 192, 384, 768]),
    "large": ("vitl", 256, [256, 512, 1024, 1024]),
}


class NoCalibrationError(RuntimeError):
    """Raised when a frame arrives before /calib. Maps to HTTP 409 so the
    client knows to re-send calibration — no other failure should do that."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8890)
    p.add_argument("--model-size", choices=sorted(MODEL_MAP), default="small")
    p.add_argument("--depth-repo", type=str,
                   default="$SCRATCH/Depth-Anything-V2/metric_depth",
                   help="Path to Depth Anything V2 metric_depth dir ($SCRATCH expanded)")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Explicit .pth path; default = <depth-repo>/checkpoints/..._{vits|vitb|vitl}.pth")
    p.add_argument("--input-size", type=int, default=518)
    p.add_argument("--warmup-runs", type=int, default=3)
    p.add_argument("--allow-cpu", action="store_true",
                   help="Permit CPU inference (slow; offline/testing only)")
    p.add_argument("--cpu-threads", type=int, default=0,
                   help="torch CPU threads (0 = leave torch default)")
    # OGM/depth settings — identical names and defaults to the Jetson node.
    p.add_argument("--max-depth", type=float, default=20.0)
    p.add_argument("--sample-step", type=int, default=8)
    p.add_argument("--floor-candidate-y-fraction", type=float, default=0.42)
    p.add_argument("--ransac-threshold", type=float, default=0.08)
    p.add_argument("--ransac-iterations", type=int, default=100)
    p.add_argument("--floor-tolerance", type=float, default=0.10)
    p.add_argument("--obstacle-min-height", type=float, default=0.10)
    p.add_argument("--obstacle-max-height", type=float, default=1.50)
    p.add_argument("--grid-lateral-half-m", type=float, default=5.0)
    p.add_argument("--grid-forward-m", type=float, default=16.0)
    p.add_argument("--grid-resolution", type=float, default=0.10)
    p.add_argument("--angle-resolution-deg", type=float, default=0.25)
    p.add_argument("--min-floor-points-per-bin", type=int, default=3)
    p.add_argument("--max-obstacle-connection-m", type=float, default=0.30)
    p.add_argument("--debug-scale", type=int, default=4)
    p.add_argument("--latency-report-every", type=int, default=30)
    # --- diagnostics ---
    p.add_argument("--dump-dir", type=str, default="",
                   help="Write per-stage diagnostic images/stats here (debugging)")
    p.add_argument("--dump-every", type=int, default=0,
                   help="Dump every Nth frame (0=off). Use 10 with --dump-dir.")
    p.add_argument("--plane-min-inliers", type=float, default=0.55,
                   help="Never lock onto a floor plane whose inlier ratio is "
                        "below this — a bad plane must not persist via hysteresis")
    p.add_argument("--no-plane-hysteresis", action="store_true",
                   help="Disable cross-frame plane locking entirely")
    return p.parse_args()


class OccupancyEngine:
    """Loads the depth model once; turns one JPEG into one occupancy grid."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        self.args = args
        self.torch = torch

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.device = torch.device("cuda")
            print(f"[server] device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
        elif args.allow_cpu:
            self.device = torch.device("cpu")
            if args.cpu_threads > 0:
                torch.set_num_threads(args.cpu_threads)
            print(f"[server] WARNING: running on CPU ({torch.get_num_threads()} threads) — "
                  "expect seconds per frame. Fine for offline/protocol testing, "
                  "NOT for live runs.", flush=True)
        else:
            raise SystemExit(
                "No CUDA GPU visible. Request an allocation with --gres=gpu:..., "
                "or pass --allow-cpu for offline/protocol testing.")

        # State must exist before any request can arrive. Initialized here,
        # before the slow model load, so nothing observes a half-built engine.
        self._calib_lock = threading.Lock()
        self._map1 = self._map2 = None
        self._rect_k = None
        self._calib_size = None
        self._calib_signature = None
        self._infer_lock = threading.Lock()  # one model, one inference at a time
        self._rng = np.random.default_rng()
        self._last_plane = None              # floor-plane hysteresis across frames
        self._plane_ratio = deque(maxlen=100)
        self._latency = deque(maxlen=100)
        self._depth_latency = deque(maxlen=100)
        self._processed = 0

        # ---- latest-grid snapshot for /refine (Teja pipeline glue) ----
        self._snap_lock = threading.Lock()
        self._latest_grid = None
        self._latest_stamp = None
        self._latest_pose = None
        try:
            import signway_refinement as _ref
            self._refine = _ref
            print("[server] refinement pipeline loaded (Teja packages + glue)",
                  flush=True)
        except Exception as exc:
            self._refine = None
            print(f"[server] refinement unavailable ({exc}) — "
                  "occupancy endpoints still work", flush=True)

        depth_repo = Path(os.path.expandvars(args.depth_repo)).expanduser().resolve()
        if not depth_repo.exists():
            raise SystemExit(f"Depth Anything metric_depth dir not found: {depth_repo}")
        sys.path.insert(0, str(depth_repo))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise SystemExit(
                f"Could not import Depth Anything from {depth_repo}: {exc}\n"
                "Is the repo cloned and are torch/torchvision installed in this env?") from exc

        encoder, features, out_channels = MODEL_MAP[args.model_size]
        ckpt = Path(os.path.expandvars(args.checkpoint)).expanduser() if args.checkpoint else (
            depth_repo / "checkpoints" / f"depth_anything_v2_metric_hypersim_{encoder}.pth"
        )
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt}")

        print(f"[server] loading Depth Anything V2 {args.model_size} from {ckpt}", flush=True)
        model = DepthAnythingV2(encoder=encoder, features=features,
                                out_channels=out_channels, max_depth=args.max_depth)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
        self.model = model.to(self.device).eval()
        print(f"[server] model loaded on {self.device}", flush=True)

        if args.warmup_runs > 0:
            print(f"[server] warm-up: {args.warmup_runs} runs"
                  f"{' (slow on CPU — be patient)' if self.device.type == 'cpu' else ''}...",
                  flush=True)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        with torch.inference_mode():
            for _ in range(max(0, args.warmup_runs)):
                _ = self.model.infer_image(dummy, args.input_size)
                self._sync()
        print(f"[server] warm-up complete ({args.warmup_runs} runs)", flush=True)

    def _sync(self) -> None:
        """No-op on CPU; real barrier on CUDA (needed for honest timing)."""
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

    # ------------------------------------------------------------ diagnostics
    def _dump_frame(self, rectified, depth, xyz, uv, signed_height,
                    floor_mask, obst_mask, grid, debug, plane, plane_ratio):
        """Write every intermediate stage for ONE frame so the broken stage is
        visible instead of inferred. Cheap and only runs when --dump-dir is set."""
        a = self.args
        d = Path(os.path.expandvars(a.dump_dir)).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        n = self._processed
        h, w = depth.shape

        cv2.imwrite(str(d / f"{n:05d}_1_rectified.png"), rectified)

        # depth colormap: black = invalid/beyond max_depth
        vis = np.clip(depth, 0, a.max_depth) / max(a.max_depth, 1e-6)
        vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[~np.isfinite(depth) | (depth <= 0) | (depth > a.max_depth)] = 0
        cv2.imwrite(str(d / f"{n:05d}_2_depth.png"), vis)

        # classification overlay: GREEN floor, RED obstacle, BLUE neither
        ov = rectified.copy()
        colors = np.zeros((len(uv), 3), np.uint8)
        colors[:] = (255, 80, 0)            # BGR blue-ish = unclassified
        colors[floor_mask] = (0, 220, 0)    # green = floor
        colors[obst_mask] = (0, 0, 255)     # red = obstacle
        for (u, v), c in zip(uv, colors):
            cv2.circle(ov, (int(u), int(v)), 2, tuple(int(x) for x in c), -1)
        cv2.imwrite(str(d / f"{n:05d}_3_classify.png"), ov)
        cv2.imwrite(str(d / f"{n:05d}_4_grid.png"), debug)

        # numbers: is the asymmetry in the DEPTH or in the CLASSIFICATION?
        thirds = {}
        for name, lo, hi in (("left", 0, w // 3),
                             ("center", w // 3, 2 * w // 3),
                             ("right", 2 * w // 3, w)):
            sel = (uv[:, 0] >= lo) & (uv[:, 0] < hi)
            dsel = depth[:, lo:hi]
            valid = np.isfinite(dsel) & (dsel > 0) & (dsel <= a.max_depth)
            thirds[name] = {
                "points": int(sel.sum()),
                "floor": int((sel & floor_mask).sum()),
                "obstacle": int((sel & obst_mask).sum()),
                "depth_valid_frac": round(float(valid.mean()), 3),
                "depth_median_m": (round(float(np.median(dsel[valid])), 2)
                                   if valid.any() else None),
            }
        stats = {
            "frame": n,
            "plane_abc": [round(float(v), 5) for v in plane],
            "plane_inlier_ratio": round(float(plane_ratio), 3),
            "camera_height_est_m": round(float(plane[2]), 3),
            "points_total": int(len(xyz)),
            "floor_total": int(floor_mask.sum()),
            "obstacle_total": int(obst_mask.sum()),
            "signed_height_pct": {p: round(float(np.percentile(signed_height, p)), 3)
                                  for p in (1, 25, 50, 75, 99)},
            "grid_free_cells": int((grid == 0).sum()),
            "grid_occupied_cells": int((grid == 100).sum()),
            "grid_unknown_cells": int((grid == -1).sum()),
            "by_image_third": thirds,
        }
        (d / f"{n:05d}_5_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"[server] dumped frame {n} -> {d} "
              f"(plane_inliers={plane_ratio:.2f}, "
              f"floor L/C/R={thirds['left']['floor']}/"
              f"{thirds['center']['floor']}/{thirds['right']['floor']})", flush=True)

    # ---------------------------------------------------------------- calib
    def set_calib(self, payload: dict) -> dict:
        w, h = int(payload["width"]), int(payload["height"])
        if w <= 0 or h <= 0:
            raise ValueError("invalid width/height")
        K = np.asarray(payload["K"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(payload.get("D", []), dtype=np.float64)
        R = np.asarray(payload.get("R", np.eye(3).ravel().tolist()),
                       dtype=np.float64).reshape(3, 3)
        P = np.asarray(payload["P"], dtype=np.float64).reshape(3, 4)
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("K has invalid focal lengths")
        newK = P[:, :3].copy()
        if newK[0, 0] <= 0 or newK[1, 1] <= 0:
            newK = K.copy()

        size = (w, h)
        signature = (size,
                     tuple(np.round(K.ravel(), 10)),
                     tuple(np.round(D.ravel(), 10)),
                     tuple(np.round(R.ravel(), 10)),
                     tuple(np.round(newK.ravel(), 10)))
        with self._calib_lock:
            if self._calib_signature != signature:
                self._map1, self._map2 = cv2.initUndistortRectifyMap(
                    K, D, R, newK, size, cv2.CV_32FC1)
                self._rect_k = newK.astype(np.float64)
                self._calib_size = size
                self._calib_signature = signature
                print(f"[server] calib accepted: {w}x{h} "
                      f"fx={newK[0,0]:.2f} fy={newK[1,1]:.2f}", flush=True)
        return {"ok": True,
                "rect_fx": float(newK[0, 0]), "rect_fy": float(newK[1, 1]),
                "rect_cx": float(newK[0, 2]), "rect_cy": float(newK[1, 2])}

    # ------------------------------------------------------------ inference
    def refine(self, payload: dict) -> dict:
        """Run the Teja refinement pipeline against the newest grid."""
        if self._refine is None:
            return {"action": payload.get("action", []),
                    "reason": "error:refinement_not_loaded",
                    "safe": False, "modified": False, "blocked": False}
        with self._snap_lock:
            grid, stamp, pose = (self._latest_grid, self._latest_stamp,
                                 self._latest_pose)
        settings = self._refine.RefinementSettings(
            resolution_m=self.args.grid_resolution,
            lateral_half_m=self.args.grid_lateral_half_m)
        return self._refine.refine_action(
            payload.get("action"), grid,
            grid_stamp=stamp, grid_pose=pose,
            pose_now=tuple(payload["pose_now"]) if payload.get("pose_now") else None,
            now=payload.get("client_time"), settings=settings)

    def process_jpeg(self, jpeg: bytes, stamp: float = None, pose=None) -> dict:
        a = self.args
        with self._calib_lock:
            if self._calib_signature is None:
                raise NoCalibrationError("no calibration — POST /calib first")
            map1, map2 = self._map1, self._map2
            rect_k = self._rect_k.copy()
            calib_size = self._calib_size

        t0 = time.perf_counter()
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode JPEG body")
        h, w = bgr.shape[:2]
        if (w, h) != calib_size:
            raise ValueError(
                f"image is {w}x{h}, calib is {calib_size[0]}x{calib_size[1]}")
        rectified = cv2.remap(bgr, map1, map2, interpolation=cv2.INTER_LINEAR)

        with self._infer_lock:
            self._sync()
            td0 = time.perf_counter()
            with self.torch.inference_mode():
                depth = self.model.infer_image(rectified, a.input_size)
            self._sync()
            depth_ms = (time.perf_counter() - td0) * 1000.0

        xyz, uv = backproject_depth(
            depth, fx=float(rect_k[0, 0]), fy=float(rect_k[1, 1]),
            cx=float(rect_k[0, 2]), cy=float(rect_k[1, 2]),
            step=a.sample_step, max_depth=a.max_depth)
        if len(xyz) < 100:
            raise ValueError("too few valid depth points")

        candidates = xyz[uv[:, 1] >= int(h * a.floor_candidate_y_fraction)]
        if len(candidates) < 100:
            raise ValueError("too few lower-image points for floor RANSAC")
        coeff, inl = fit_floor_ransac(candidates, threshold=a.ransac_threshold,
                                      iterations=a.ransac_iterations, rng=self._rng)
        plane_ratio = float(np.asarray(inl).mean())
        # --- floor-plane hysteresis: the floor does not change between frames.
        # On glossy floors RANSAC can flip-flop between the true floor and a
        # reflection-built phantom plane, flickering the whole grid. Score last
        # frame's plane against THIS frame's points and keep it unless the new
        # fit is clearly (>5% inlier ratio) better. Self-corrects if the old
        # plane genuinely goes stale (ramp, big pitch change).
        if self._last_plane is not None and not a.no_plane_hysteresis:
            la, lb, lc = self._last_plane
            d_old = np.abs((la * candidates[:, 0] + lb * candidates[:, 2] + lc)
                           - candidates[:, 1])
            old_ratio = float((d_old <= a.ransac_threshold).mean())
            # Only defend an incumbent that is actually GOOD. Without this a
            # bad first-frame plane would persist forever (hysteresis turning
            # a one-frame accident into a permanent failure).
            if old_ratio >= a.plane_min_inliers and old_ratio >= plane_ratio - 0.05:
                coeff = np.asarray(self._last_plane, dtype=np.float64)
                plane_ratio = old_ratio
        # Never store a plane we would not be willing to lock onto.
        if plane_ratio >= a.plane_min_inliers:
            self._last_plane = tuple(float(v) for v in coeff)
        else:
            self._last_plane = None
            if self._processed % 10 == 0:
                print(f"[server] WARNING: weak floor fit "
                      f"(inliers={plane_ratio:.2f} < {a.plane_min_inliers}) — "
                      "depth or floor geometry is the problem, not the fit",
                      flush=True)
        self._plane_ratio.append(plane_ratio)
        ca, cb, cc = [float(v) for v in coeff]
        signed_height = (ca * xyz[:, 0] + cb * xyz[:, 2] + cc) - xyz[:, 1]

        floor_mask = np.abs(signed_height) <= a.floor_tolerance
        obst_mask = ((signed_height >= a.obstacle_min_height)
                     & (signed_height <= a.obstacle_max_height))
        floor_pts = xyz[floor_mask]
        obstacle_pts = xyz[obst_mask]
        grid = build_camera_occupancy(
            floor_pts=floor_pts, obstacle_pts=obstacle_pts,
            lateral_half_m=a.grid_lateral_half_m, forward_m=a.grid_forward_m,
            resolution=a.grid_resolution, angle_resolution_deg=a.angle_resolution_deg,
            min_floor_points_per_bin=a.min_floor_points_per_bin,
            max_obstacle_connection_m=a.max_obstacle_connection_m)
        debug = occupancy_to_debug_image(grid, scale=a.debug_scale)

        if a.dump_dir and a.dump_every > 0 and self._processed % a.dump_every == 0:
            try:
                self._dump_frame(rectified, depth, xyz, uv, signed_height,
                                 floor_mask, obst_mask, grid, debug,
                                 (ca, cb, cc), plane_ratio)
            except Exception as exc:      # diagnostics must never break serving
                print(f"[server] dump failed: {type(exc).__name__}: {exc}", flush=True)

        with self._snap_lock:
            self._latest_grid = grid
            self._latest_stamp = stamp if stamp is not None else time.time()
            self._latest_pose = pose

        total_ms = (time.perf_counter() - t0) * 1000.0
        self._depth_latency.append(depth_ms)
        self._latency.append(total_ms)
        self._processed += 1
        if a.latency_report_every > 0 and self._processed % a.latency_report_every == 0:
            md = float(np.mean(self._depth_latency))
            mt = float(np.mean(self._latency))
            fps = 1000.0 / mt if mt > 0 else float("inf")
            print(f"[server] processed={self._processed} depth={md:.1f} ms "
                  f"total={mt:.1f} ms (~{fps:.2f} FPS server-side), "
                  f"floor={len(floor_pts)} obst={len(obstacle_pts)} "
                  f"plane_inliers={float(np.mean(self._plane_ratio)):.2f}", flush=True)

        return {"grid": grid, "debug": debug,
                "depth_ms": depth_ms, "total_ms": total_ms,
                "meta": {"resolution": a.grid_resolution,
                         "lateral_half_m": a.grid_lateral_half_m,
                         "forward_m": a.grid_forward_m,
                         "values": "-1 unknown / 0 free / 100 occupied",
                         "frame": "camera: rows increase forward, cols increase right"}}

    def health(self) -> dict:
        return {"ok": True,
                "model_size": self.args.model_size,
                "device": str(self.device),
                "refinement_loaded": self._refine is not None,
                "grid_cached": self._latest_grid is not None,
                "processed": self._processed,
                "calib_ready": self._calib_signature is not None,
                "plane_inliers": (float(np.mean(self._plane_ratio))
                                  if self._plane_ratio else None),
                "mean_total_ms": float(np.mean(self._latency)) if self._latency else None}


def make_handler(engine: OccupancyEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: avoids per-frame TCP setup

        def log_message(self, fmt, *args_):  # latency matters; access logs don't
            pass

        def _reply(self, code: int, body: bytes, ctype: str, extra=None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _reply_json(self, code: int, obj: dict) -> None:
            self._reply(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?")[0] == "/health":
                self._reply_json(200, engine.health())
            else:
                self._reply_json(404, {"error": "unknown path"})

        def do_POST(self) -> None:  # noqa: N802
            path, _, query = self.path.partition("?")
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self._reply_json(400, {"error": "bad Content-Length"})
                return
            body = self.rfile.read(length) if length > 0 else b""

            try:
                if path == "/calib":
                    self._reply_json(200, engine.set_calib(json.loads(body)))
                elif path == "/refine":
                    self._reply_json(200, engine.refine(json.loads(body)))
                elif path == "/occupancy":
                    stamp = self.headers.get("X-Image-Stamp")
                    pose_h = self.headers.get("X-Odom-Pose")
                    pose = (tuple(float(v) for v in pose_h.split(","))
                            if pose_h else None)
                    result = engine.process_jpeg(
                        body, stamp=float(stamp) if stamp else None, pose=pose)
                    headers = {"X-Depth-Ms": f"{result['depth_ms']:.1f}",
                               "X-Total-Ms": f"{result['total_ms']:.1f}"}
                    if "fmt=npz" in query:
                        buf = io.BytesIO()
                        np.savez_compressed(
                            buf, grid=result["grid"], debug=result["debug"],
                            meta=json.dumps(result["meta"]),
                            depth_ms=result["depth_ms"], total_ms=result["total_ms"])
                        self._reply(200, buf.getvalue(),
                                    "application/octet-stream", headers)
                    else:
                        ok, png = cv2.imencode(".png", result["debug"])
                        if not ok:
                            raise ValueError("PNG encode failed")
                        self._reply(200, png.tobytes(), "image/png", headers)
                else:
                    self._reply_json(404, {"error": "unknown path"})
            except NoCalibrationError as exc:
                # ONLY this maps to 409 — it's the client's cue to re-send calib.
                self._reply_json(409, {"error": str(exc)})
            except Exception as exc:  # keep the server alive if one frame fails
                self._reply_json(400, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main() -> None:
    args = parse_args()
    engine = OccupancyEngine(args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(engine))
    server.daemon_threads = True
    print(f"[server] occupancy server ready on port {args.port} "
          f"(tunnel this port from the Jetson, same as policy_server.py)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] shutting down", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()