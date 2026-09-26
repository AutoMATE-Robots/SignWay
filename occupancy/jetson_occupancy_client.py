#!/usr/bin/env python3
"""Jetson-side thin client for the MSI occupancy server.

Replaces local depth inference with an HTTP round trip (same pattern as
pepper_vla_node.py -> policy_server.py):

    /image_raw + /camera_info
        -> JPEG encode newest frame only
        -> POST http://localhost:<port>/occupancy   (SSH tunnel to MSI)
        -> decode PNG reply
        -> /local_occupancy_image                    (rqt_image_view as before)

Calibration is forwarded to the server whenever the CameraInfo signature
changes, and re-sent automatically if the server restarts (HTTP 409).

Requires only rclpy, cv_bridge, cv2, numpy, urllib (stdlib) — no torch,
no Depth Anything on the Jetson.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class OccupancyClientNode(Node):
    def __init__(self) -> None:
        super().__init__("occupancy_client_node")
        self.declare_parameter("server_url", "http://localhost:8890")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("debug_image_topic", "/local_occupancy_image")
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("request_timeout_s", 2.0)
        self.declare_parameter("latency_report_every", 30)

        self.server_url = str(self.get_parameter("server_url").value).rstrip("/")
        self.bridge = CvBridge()

        self._latest_lock = threading.Lock()
        self._latest_image = None
        self._stop = threading.Event()

        self._calib_lock = threading.Lock()
        self._calib_payload = None      # dict ready to POST
        self._calib_signature = None    # what we last built
        self._calib_acked = None        # signature the server has confirmed

        self._latency = deque(maxlen=100)
        self._server_ms = deque(maxlen=100)
        self._processed = 0
        self._consecutive_failures = 0

        self.debug_pub = self.create_publisher(
            Image, str(self.get_parameter("debug_image_topic").value), 1)
        self.create_subscription(
            CameraInfo, str(self.get_parameter("camera_info_topic").value),
            self._camera_info_cb, qos_profile_sensor_data)
        self.create_subscription(
            Image, str(self.get_parameter("image_topic").value),
            self._image_cb, qos_profile_sensor_data)

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"occupancy client ready -> {self.server_url} "
            "(start the SSH tunnel + MSI server first)")

    # ------------------------------------------------------------- callbacks
    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if msg.width <= 0 or msg.height <= 0:
            return
        K = list(msg.k)
        if K[0] <= 0 or K[4] <= 0:
            return
        payload = {
            "width": int(msg.width), "height": int(msg.height),
            "K": K, "D": list(msg.d), "R": list(msg.r), "P": list(msg.p),
        }
        signature = json.dumps(payload, sort_keys=True)
        with self._calib_lock:
            if signature != self._calib_signature:
                self._calib_signature = signature
                self._calib_payload = payload
                self._calib_acked = None  # force a (re)send

    def _image_cb(self, msg: Image) -> None:
        # Single slot on purpose: stale frames never queue up.
        with self._latest_lock:
            self._latest_image = msg

    # ----------------------------------------------------------------- HTTP
    def _post(self, path: str, body: bytes, ctype: str, timeout: float):
        req = urllib.request.Request(
            self.server_url + path, data=body, method="POST",
            headers={"Content-Type": ctype})
        return urllib.request.urlopen(req, timeout=timeout)

    def _ensure_calib(self, timeout: float) -> bool:
        with self._calib_lock:
            payload = self._calib_payload
            signature = self._calib_signature
            acked = self._calib_acked
        if payload is None:
            return False  # no CameraInfo yet
        if acked == signature:
            return True
        resp = self._post("/calib", json.dumps(payload).encode(),
                          "application/json", timeout)
        if resp.status == 200:
            with self._calib_lock:
                self._calib_acked = signature
            self.get_logger().info("calibration accepted by server")
            return True
        return False

    # --------------------------------------------------------------- worker
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            with self._latest_lock:
                msg = self._latest_image
                self._latest_image = None
            if msg is None:
                time.sleep(0.002)
                continue
            try:
                self._process(msg)
                self._consecutive_failures = 0
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                self._consecutive_failures += 1
                if self._consecutive_failures in (1, 10) or self._consecutive_failures % 100 == 0:
                    self.get_logger().error(
                        f"server unreachable ({self._consecutive_failures}x): {exc} — "
                        "is the SSH tunnel + MSI server up?")
                time.sleep(0.5)
            except Exception as exc:  # keep the client alive if one frame fails
                self.get_logger().error(f"frame skipped: {type(exc).__name__}: {exc}")

    def _process(self, msg: Image) -> None:
        timeout = float(self.get_parameter("request_timeout_s").value)
        if not self._ensure_calib(timeout):
            return  # waiting for CameraInfo, like the local node did

        t0 = time.perf_counter()
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ok, jpeg = cv2.imencode(
            ".jpg", bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.get_parameter("jpeg_quality").value)])
        if not ok:
            raise RuntimeError("JPEG encode failed")

        try:
            resp = self._post("/occupancy", jpeg.tobytes(), "image/jpeg", timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                # Server restarted and lost calibration — re-send next cycle.
                with self._calib_lock:
                    self._calib_acked = None
                self.get_logger().warn("server lost calibration; re-sending")
                return
            raise
        png = resp.read()
        server_ms = float(resp.headers.get("X-Total-Ms", "nan"))

        debug = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if debug is None:
            raise RuntimeError("could not decode server PNG reply")
        out = self.bridge.cv2_to_imgmsg(debug, encoding="mono8")
        out.header = msg.header
        self.debug_pub.publish(out)

        total_ms = (time.perf_counter() - t0) * 1000.0
        self._latency.append(total_ms)
        if np.isfinite(server_ms):
            self._server_ms.append(server_ms)
        self._processed += 1
        every = int(self.get_parameter("latency_report_every").value)
        if every > 0 and self._processed % every == 0:
            mt = float(np.mean(self._latency))
            ms = float(np.mean(self._server_ms)) if self._server_ms else float("nan")
            self.get_logger().info(
                f"processed={self._processed} round-trip={mt:.1f} ms "
                f"(server {ms:.1f} ms, network+codec {mt - ms:.1f} ms, ~{1000.0/mt:.2f} FPS)")

    def destroy_node(self):
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = OccupancyClientNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
