#!/usr/bin/env python3
"""Live camera-only occupancy generation for the Jetson.

Phase-1 runtime:
    /image_raw + /camera_info
        -> rectify
        -> Depth Anything V2 Metric Indoor
        -> depth back-projection
        -> RANSAC floor fit
        -> obstacle/floor separation
        -> pseudo-LiDAR + free-space raycasting
        -> /local_occupancy_image

The production runtime intentionally does not read rosbag files, LiDAR,
or write videos/images to disk.
"""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import sys
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Pure occupancy math. These functions intentionally have no ROS/PyTorch imports
# so they can be unit-tested on a development machine.
# -----------------------------------------------------------------------------


def backproject_depth(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    step: int = 8,
    max_depth: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project a rectified metric-depth image into camera optical XYZ.

    Camera optical convention used here:
      X: right, Y: down, Z: forward.

    Returns:
      xyz: Nx3 float32 points in meters.
      uv:  Nx2 int32 source pixels [u, v] corresponding to xyz.
    """
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")
    if step < 1:
        raise ValueError("step must be >= 1")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx/fy must be positive")

    h, w = depth.shape
    vs = np.arange(0, h, step, dtype=np.int32)
    us = np.arange(0, w, step, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    zz = depth[vv, uu].astype(np.float32, copy=False)

    valid = np.isfinite(zz) & (zz > 0.0) & (zz <= max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    u = uu[valid].astype(np.float32)
    v = vv[valid].astype(np.float32)
    z = zz[valid]
    x = (u - float(cx)) * z / float(fx)
    y = (v - float(cy)) * z / float(fy)

    xyz = np.column_stack((x, y, z)).astype(np.float32, copy=False)
    uv = np.column_stack((uu[valid], vv[valid])).astype(np.int32, copy=False)
    return xyz, uv


def fit_floor_ransac(
    points_xyz: np.ndarray,
    threshold: float = 0.08,
    iterations: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit camera-space floor model Y = a*X + b*Z + c using RANSAC."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must be Nx3")
    if len(pts) < 3:
        raise ValueError("at least 3 points are required for RANSAC")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    rng = rng or np.random.default_rng()
    A = np.column_stack((pts[:, 0], pts[:, 2], np.ones(len(pts))))
    y = pts[:, 1]

    best_mask = None
    best_count = -1
    best_median = math.inf

    for _ in range(max(1, int(iterations))):
        idx = rng.choice(len(pts), size=3, replace=False)
        sample_A = A[idx]
        if abs(np.linalg.det(sample_A)) < 1e-9:
            continue
        coeff = np.linalg.solve(sample_A, y[idx])
        residual = np.abs(y - A @ coeff)
        mask = residual <= threshold
        count = int(mask.sum())
        if count < 3:
            continue
        median = float(np.median(residual[mask]))
        if count > best_count or (count == best_count and median < best_median):
            best_count = count
            best_median = median
            best_mask = mask

    if best_mask is None or int(best_mask.sum()) < 3:
        raise RuntimeError("RANSAC could not find a floor plane")

    coeff, *_ = np.linalg.lstsq(A[best_mask], y[best_mask], rcond=None)
    residual = np.abs(y - A @ coeff)
    inliers = residual <= threshold
    return coeff.astype(np.float32), inliers


def _nearest_obstacle_per_bin(
    obstacle_pts: np.ndarray, angle_resolution_deg: float
) -> Dict[int, Tuple[float, float, float]]:
    result: Dict[int, Tuple[float, float, float]] = {}
    if len(obstacle_pts) == 0:
        return result

    x = obstacle_pts[:, 0]
    z = obstacle_pts[:, 2]
    r = np.hypot(x, z)
    angles_deg = np.degrees(np.arctan2(x, z))
    bins = np.rint(angles_deg / angle_resolution_deg).astype(np.int32)

    order = np.lexsort((r, bins))
    bins_sorted = bins[order]
    first = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        first[1:] = bins_sorted[1:] != bins_sorted[:-1]

    for idx in order[first]:
        b = int(bins[idx])
        result[b] = (float(x[idx]), float(z[idx]), float(r[idx]))
    return result


def _floor_extent_per_bin(
    floor_pts: np.ndarray,
    angle_resolution_deg: float,
    min_points: int,
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    if len(floor_pts) == 0:
        return result

    x = floor_pts[:, 0]
    z = floor_pts[:, 2]
    r = np.hypot(x, z)
    angles_deg = np.degrees(np.arctan2(x, z))
    bins = np.rint(angles_deg / angle_resolution_deg).astype(np.int32)

    for b in np.unique(bins):
        vals = r[bins == b]
        if len(vals) >= min_points:
            # Robust long-range visible-floor evidence; avoids one stray depth point.
            result[int(b)] = float(np.percentile(vals, 90.0))
    return result


def build_camera_occupancy(
    floor_pts: np.ndarray,
    obstacle_pts: np.ndarray,
    lateral_half_m: float = 5.0,
    forward_m: float = 16.0,
    resolution: float = 0.10,
    angle_resolution_deg: float = 0.25,
    min_floor_points_per_bin: int = 3,
    max_obstacle_connection_m: float = 0.30,
) -> np.ndarray:
    """Build a camera-centric local occupancy grid.

    Internal values follow ROS occupancy semantics:
      -1 unknown, 0 free, 100 occupied.

    Rows increase forward from the camera; columns increase to camera-right.
    Free-space rays always stop before the nearest obstacle in that direction.
    """
    if resolution <= 0 or lateral_half_m <= 0 or forward_m <= 0:
        raise ValueError("grid dimensions/resolution must be positive")
    if angle_resolution_deg <= 0:
        raise ValueError("angle_resolution_deg must be positive")

    width = int(math.ceil((2.0 * lateral_half_m) / resolution))
    height = int(math.ceil(forward_m / resolution))
    grid = np.full((height, width), -1, dtype=np.int16)

    obstacle_bins = _nearest_obstacle_per_bin(
        np.asarray(obstacle_pts, dtype=np.float32), angle_resolution_deg
    )
    floor_bins = _floor_extent_per_bin(
        np.asarray(floor_pts, dtype=np.float32),
        angle_resolution_deg,
        min_floor_points_per_bin,
    )

    origin_col = int(round(lateral_half_m / resolution))
    origin_col = min(max(origin_col, 0), width - 1)
    origin = (origin_col, 0)  # cv2 uses (col, row)

    all_bins = sorted(set(floor_bins) | set(obstacle_bins))
    free_margin = max(resolution, 0.05)

    for b in all_bins:
        angle = math.radians(b * angle_resolution_deg)
        obstacle = obstacle_bins.get(b)
        obstacle_range = obstacle[2] if obstacle else math.inf
        floor_range = floor_bins.get(b)

        # Seeing an obstacle itself proves the line of sight before it was observed,
        # even if floor evidence in that exact angular bin is sparse.
        observed_range = floor_range if floor_range is not None else obstacle_range
        if math.isfinite(observed_range):
            free_range = min(observed_range, obstacle_range - free_margin)
            free_range = min(free_range, forward_m)
            if free_range > 0.0:
                x = math.sin(angle) * free_range
                z = math.cos(angle) * free_range
                col = int(round((x + lateral_half_m) / resolution))
                row = int(round(z / resolution))
                if 0 <= col < width and row >= 0:
                    row = min(row, height - 1)
                    cv2.line(grid, origin, (col, row), 0, thickness=1, lineType=cv2.LINE_8)

        if obstacle is not None:
            ox, oz, _ = obstacle
            if 0.0 <= oz < forward_m and -lateral_half_m <= ox < lateral_half_m:
                col = int(round((ox + lateral_half_m) / resolution))
                row = int(round(oz / resolution))
                if 0 <= col < width and 0 <= row < height:
                    grid[row, col] = 100

    # Connect nearby pseudo-LiDAR obstacle endpoints so walls are not dotted.
    obstacle_items = sorted(obstacle_bins.items())
    for (b1, p1), (b2, p2) in zip(obstacle_items, obstacle_items[1:]):
        angle_gap = abs(b2 - b1) * angle_resolution_deg
        if angle_gap > 1.0:
            continue
        x1, z1, _ = p1
        x2, z2, _ = p2
        if math.hypot(x2 - x1, z2 - z1) > max_obstacle_connection_m:
            continue
        if not (0.0 <= z1 < forward_m and 0.0 <= z2 < forward_m):
            continue
        c1 = int(round((x1 + lateral_half_m) / resolution))
        r1 = int(round(z1 / resolution))
        c2 = int(round((x2 + lateral_half_m) / resolution))
        r2 = int(round(z2 / resolution))
        if all((0 <= c1 < width, 0 <= c2 < width, 0 <= r1 < height, 0 <= r2 < height)):
            cv2.line(grid, (c1, r1), (c2, r2), 100, thickness=1, lineType=cv2.LINE_8)

    return grid


def occupancy_to_debug_image(grid: np.ndarray, scale: int = 4) -> np.ndarray:
    """Convert occupancy grid to gray/free/occupied image with robot at bottom."""
    img = np.full(grid.shape, 127, dtype=np.uint8)
    img[grid == 0] = 255
    img[grid == 100] = 0
    img = np.flipud(img)
    if scale > 1:
        img = cv2.resize(
            img,
            (img.shape[1] * scale, img.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return img


# -----------------------------------------------------------------------------
# Live ROS2 node. Imports are intentionally lazy so the math above stays usable
# in unit tests on non-ROS development machines.
# -----------------------------------------------------------------------------


def run_ros_node() -> None:
    try:
        import torch
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from cv_bridge import CvBridge
    except ImportError as exc:
        raise SystemExit(
            f"Missing runtime dependency: {exc}. See README.md before running the node."
        ) from exc

    class CameraOccupancyNode(Node):
        MODEL_MAP = {
            "small": ("vits", 64, [48, 96, 192, 384]),
            "base": ("vitb", 128, [96, 192, 384, 768]),
            "large": ("vitl", 256, [256, 512, 1024, 1024]),
        }

        def __init__(self) -> None:
            super().__init__("camera_occupancy_node")

            # ROS topics and model selection.
            self.declare_parameter("image_topic", "/image_raw")
            self.declare_parameter("camera_info_topic", "/camera_info")
            self.declare_parameter("debug_image_topic", "/local_occupancy_image")
            self.declare_parameter("model_size", "small")
            self.declare_parameter("depth_repo", str(Path.home() / "Trajectory_Generation_and_Control/occupancy/Depth-Anything-V2/metric_depth"))
            self.declare_parameter("checkpoint", "")
            self.declare_parameter("input_size", 518)
            self.declare_parameter("warmup_runs", 3)

            # OGM/depth settings copied from the successful offline prototype.
            self.declare_parameter("max_depth", 20.0)
            self.declare_parameter("sample_step", 8)
            self.declare_parameter("floor_candidate_y_fraction", 0.42)
            self.declare_parameter("ransac_threshold", 0.08)
            self.declare_parameter("ransac_iterations", 100)
            self.declare_parameter("floor_tolerance", 0.10)
            self.declare_parameter("obstacle_min_height", 0.10)
            self.declare_parameter("obstacle_max_height", 1.50)
            self.declare_parameter("grid_lateral_half_m", 5.0)
            self.declare_parameter("grid_forward_m", 16.0)
            self.declare_parameter("grid_resolution", 0.10)
            self.declare_parameter("angle_resolution_deg", 0.25)
            self.declare_parameter("min_floor_points_per_bin", 3)
            self.declare_parameter("max_obstacle_connection_m", 0.30)
            self.declare_parameter("debug_scale", 4)
            self.declare_parameter("latency_report_every", 30)

            self.image_topic = str(self.get_parameter("image_topic").value)
            self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
            self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)

            self.bridge = CvBridge()
            self._calib_lock = threading.Lock()
            self._map1 = None
            self._map2 = None
            self._rect_k = None
            self._calib_size = None
            self._calib_ready = False

            self._latest_lock = threading.Lock()
            self._latest_image = None
            self._stop = threading.Event()
            self._rng = np.random.default_rng()
            self._latency = deque(maxlen=100)
            self._depth_latency = deque(maxlen=100)
            self._processed = 0

            self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 1)
            self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, qos_profile_sensor_data)
            self.create_subscription(Image, self.image_topic, self._image_cb, qos_profile_sensor_data)

            self.device = self._select_device(torch)
            self.model = self._load_depth_model(torch)
            self._warm_up(torch)

            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

            self.get_logger().info(
                f"LIVE occupancy ready. image={self.image_topic}, camera_info={self.camera_info_topic}, "
                f"output={self.debug_image_topic}, device={self.device}"
            )

        def _select_device(self, torch_module):
            if not torch_module.cuda.is_available():
                raise RuntimeError(
                    "CUDA-enabled PyTorch is required for the live Jetson node, but "
                    "torch.cuda.is_available() is False. Do not run with the CPU-only system torch."
                )
            torch_module.backends.cudnn.benchmark = True
            return torch_module.device("cuda")

        def _load_depth_model(self, torch_module):
            model_size = str(self.get_parameter("model_size").value).lower()
            if model_size not in self.MODEL_MAP:
                raise ValueError("model_size must be one of: small, base, large")

            encoder, features, out_channels = self.MODEL_MAP[model_size]
            depth_repo = Path(str(self.get_parameter("depth_repo").value)).expanduser().resolve()
            if not depth_repo.exists():
                raise FileNotFoundError(
                    f"Depth Anything metric_depth directory not found: {depth_repo}"
                )
            sys.path.insert(0, str(depth_repo))
            try:
                from depth_anything_v2.dpt import DepthAnythingV2
            except ImportError as exc:
                raise ImportError(
                    f"Could not import Depth Anything from {depth_repo}: {exc}"
                ) from exc

            checkpoint_param = str(self.get_parameter("checkpoint").value).strip()
            if checkpoint_param:
                checkpoint = Path(checkpoint_param).expanduser()
            else:
                checkpoint = depth_repo / "checkpoints" / f"depth_anything_v2_metric_hypersim_{encoder}.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint}. Copy the {model_size} Metric Indoor/Hypersim checkpoint there."
                )

            max_depth = float(self.get_parameter("max_depth").value)
            self.get_logger().info(f"Loading Depth Anything V2 {model_size} from {checkpoint}")
            model = DepthAnythingV2(
                encoder=encoder,
                features=features,
                out_channels=out_channels,
                max_depth=max_depth,
            )
            state = torch_module.load(str(checkpoint), map_location="cpu")
            model.load_state_dict(state)
            model = model.to(self.device).eval()
            self.get_logger().info(f"Depth model loaded on {torch_module.cuda.get_device_name(0)}")
            return model

        def _warm_up(self, torch_module) -> None:
            runs = int(self.get_parameter("warmup_runs").value)
            input_size = int(self.get_parameter("input_size").value)
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.get_logger().info(f"GPU warm-up: {runs} throwaway inference runs")
            with torch_module.inference_mode():
                for _ in range(max(0, runs)):
                    _ = self.model.infer_image(dummy, input_size)
                    torch_module.cuda.synchronize()
            self.get_logger().info("GPU warm-up complete")

        def _camera_info_cb(self, msg) -> None:
            if msg.width <= 0 or msg.height <= 0:
                self.get_logger().error("CameraInfo has invalid width/height")
                return
            K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            D = np.asarray(msg.d, dtype=np.float64)
            R = np.asarray(msg.r, dtype=np.float64).reshape(3, 3)
            P = np.asarray(msg.p, dtype=np.float64).reshape(3, 4)
            if K[0, 0] <= 0 or K[1, 1] <= 0:
                self.get_logger().error("CameraInfo K contains zero/invalid focal lengths; waiting for valid calibration")
                return

            newK = P[:, :3].copy()
            if newK[0, 0] <= 0 or newK[1, 1] <= 0:
                self.get_logger().warn("CameraInfo P is invalid; falling back to K for rectification")
                newK = K.copy()

            size = (int(msg.width), int(msg.height))
            signature = (
                size,
                tuple(np.round(K.ravel(), 10)),
                tuple(np.round(D.ravel(), 10)),
                tuple(np.round(R.ravel(), 10)),
                tuple(np.round(newK.ravel(), 10)),
            )
            with self._calib_lock:
                if getattr(self, "_calib_signature", None) == signature:
                    return
                map1, map2 = cv2.initUndistortRectifyMap(
                    K, D, R, newK, size, cv2.CV_32FC1
                )
                self._map1 = map1
                self._map2 = map2
                self._rect_k = newK.astype(np.float64)
                self._calib_size = size
                self._calib_signature = signature
                self._calib_ready = True
            self.get_logger().info(
                "CameraInfo accepted: "
                f"{size[0]}x{size[1]}, rectified fx={newK[0,0]:.3f}, fy={newK[1,1]:.3f}, "
                f"cx={newK[0,2]:.3f}, cy={newK[1,2]:.3f}"
            )

        def _image_cb(self, msg) -> None:
            # Replacing this single slot is deliberate: stale images never queue up.
            with self._latest_lock:
                self._latest_image = msg

        def _worker_loop(self) -> None:
            while not self._stop.is_set():
                with self._latest_lock:
                    msg = self._latest_image
                    self._latest_image = None
                if msg is None:
                    time.sleep(0.002)
                    continue
                try:
                    self._process_image(msg)
                except Exception as exc:  # Keep the live node alive if one frame fails.
                    self.get_logger().error(f"Frame skipped: {type(exc).__name__}: {exc}")

        def _process_image(self, msg) -> None:
            import torch

            with self._calib_lock:
                if not self._calib_ready:
                    # CameraInfo normally arrives quickly; skip images until it does.
                    return
                map1 = self._map1
                map2 = self._map2
                rect_k = self._rect_k.copy()
                calib_size = self._calib_size

            t0 = time.perf_counter()
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            h, w = bgr.shape[:2]
            if (w, h) != calib_size:
                raise RuntimeError(
                    f"Image is {w}x{h}, but CameraInfo is {calib_size[0]}x{calib_size[1]}"
                )
            rectified = cv2.remap(bgr, map1, map2, interpolation=cv2.INTER_LINEAR)

            torch.cuda.synchronize()
            td0 = time.perf_counter()
            with torch.inference_mode():
                depth = self.model.infer_image(
                    rectified,
                    int(self.get_parameter("input_size").value),
                )
            torch.cuda.synchronize()
            depth_ms = (time.perf_counter() - td0) * 1000.0

            xyz, uv = backproject_depth(
                depth,
                fx=float(rect_k[0, 0]),
                fy=float(rect_k[1, 1]),
                cx=float(rect_k[0, 2]),
                cy=float(rect_k[1, 2]),
                step=int(self.get_parameter("sample_step").value),
                max_depth=float(self.get_parameter("max_depth").value),
            )
            if len(xyz) < 100:
                raise RuntimeError("Too few valid depth points")

            floor_y_frac = float(self.get_parameter("floor_candidate_y_fraction").value)
            candidate_mask = uv[:, 1] >= int(h * floor_y_frac)
            floor_candidates = xyz[candidate_mask]
            if len(floor_candidates) < 100:
                raise RuntimeError("Too few lower-image points for floor RANSAC")

            coeff, _ = fit_floor_ransac(
                floor_candidates,
                threshold=float(self.get_parameter("ransac_threshold").value),
                iterations=int(self.get_parameter("ransac_iterations").value),
                rng=self._rng,
            )
            a, b, c = [float(v) for v in coeff]
            floor_y = a * xyz[:, 0] + b * xyz[:, 2] + c
            signed_height = floor_y - xyz[:, 1]  # positive means above the fitted floor

            floor_tol = float(self.get_parameter("floor_tolerance").value)
            floor_mask = np.abs(signed_height) <= floor_tol
            min_h = float(self.get_parameter("obstacle_min_height").value)
            max_h = float(self.get_parameter("obstacle_max_height").value)
            obstacle_mask = (signed_height >= min_h) & (signed_height <= max_h)

            floor_pts = xyz[floor_mask]
            obstacle_pts = xyz[obstacle_mask]
            grid = build_camera_occupancy(
                floor_pts=floor_pts,
                obstacle_pts=obstacle_pts,
                lateral_half_m=float(self.get_parameter("grid_lateral_half_m").value),
                forward_m=float(self.get_parameter("grid_forward_m").value),
                resolution=float(self.get_parameter("grid_resolution").value),
                angle_resolution_deg=float(self.get_parameter("angle_resolution_deg").value),
                min_floor_points_per_bin=int(self.get_parameter("min_floor_points_per_bin").value),
                max_obstacle_connection_m=float(self.get_parameter("max_obstacle_connection_m").value),
            )
            debug = occupancy_to_debug_image(
                grid, scale=int(self.get_parameter("debug_scale").value)
            )
            out = self.bridge.cv2_to_imgmsg(debug, encoding="mono8")
            out.header = msg.header
            self.debug_pub.publish(out)

            total_ms = (time.perf_counter() - t0) * 1000.0
            self._depth_latency.append(depth_ms)
            self._latency.append(total_ms)
            self._processed += 1
            report_every = int(self.get_parameter("latency_report_every").value)
            if report_every > 0 and self._processed % report_every == 0:
                mean_depth = float(np.mean(self._depth_latency))
                mean_total = float(np.mean(self._latency))
                fps = 1000.0 / mean_total if mean_total > 0 else 0.0
                self.get_logger().info(
                    f"processed={self._processed} depth={mean_depth:.1f} ms "
                    f"total={mean_total:.1f} ms (~{fps:.2f} FPS), "
                    f"floor_pts={len(floor_pts)}, obstacle_pts={len(obstacle_pts)}"
                )

        def destroy_node(self):
            self._stop.set()
            if hasattr(self, "_worker") and self._worker.is_alive():
                self._worker.join(timeout=2.0)
            return super().destroy_node()

    rclpy.init()
    node = None
    try:
        node = CameraOccupancyNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    run_ros_node()
