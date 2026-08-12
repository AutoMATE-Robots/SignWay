#!/usr/bin/env python3
"""
pepper_vla_node.py -- runs on the JETSON. Subscribes to the camera, asks the policy
server for a trajectory, converts it to velocity commands with pure pursuit.

    /c1/image_raw  --(HTTP)-->  policy_server  --(8 waypoints)-->  pure pursuit  -->  /cmd_vel
                                                                \--> /pepper_vla/path (nav_msgs/Path, for rviz
                                                                     and later for the occupancy module)

SAFETY -- read before first run:
  * First test with the WHEELS OFF THE GROUND. Watch /cmd_vel, confirm the sign of
    angular.z matches the prompted direction, THEN put it down.
  * max_speed defaults deliberately low. Raise it only after a clean run.
  * If the server errors, times out, or waypoints go stale, this publishes ZERO velocity.
  * Keep a human on the e-stop. There is no obstacle avoidance in this version.

Run:
    export POLICY_URL=http://<workstation-ip>:8000/predict
    python3 pepper_vla_node.py --ros-args -p prompt:=turn_right -p max_speed:=0.15
"""
import base64
import math
import os
import time

import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image

import cv2


class PepperVLANode(Node):
    def __init__(self):
        super().__init__("pepper_vla")

        self.declare_parameter("image_topic", "/c1/image_raw")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("prompt", "straight")        # hardcoded for v1; VLM later
        self.declare_parameter("infer_hz", 4.0)             # request rate to the server
        self.declare_parameter("max_speed", 0.15)           # m/s  -- START LOW
        self.declare_parameter("max_omega", 0.8)            # rad/s clamp
        self.declare_parameter("lookahead_idx", 4)          # which waypoint to chase (1..8)
        self.declare_parameter("stale_after", 0.8)          # s; older -> stop
        self.declare_parameter("jpeg_quality", 92)
        self.declare_parameter("speed_from_model", True)    # use waypoint spacing as speed
        self.declare_parameter("dry_run", False)            # True = never publish cmd_vel

        gp = lambda n: self.get_parameter(n).value
        self.prompt = gp("prompt")
        self.infer_dt = 1.0 / max(0.5, gp("infer_hz"))
        self.max_speed = gp("max_speed")
        self.max_omega = gp("max_omega")
        self.look_idx = int(gp("lookahead_idx"))
        self.stale_after = gp("stale_after")
        self.jpeg_q = int(gp("jpeg_quality"))
        self.speed_from_model = gp("speed_from_model")
        self.dry_run = gp("dry_run")

        self.url = os.environ.get("POLICY_URL", "http://127.0.0.1:8000/predict")
        self.bridge = CvBridge()
        self.session = requests.Session()

        self.last_req = 0.0
        self.seq = 0
        self.traj = None          # (8,2) waypoints, robot frame at capture time
        self.traj_stamp = 0.0

        self.sub = self.create_subscription(
            Image, gp("image_topic"), self.on_image, QoSPresetProfiles.SENSOR_DATA.value)
        self.pub_cmd = self.create_publisher(Twist, gp("cmd_topic"), 10)
        self.pub_path = self.create_publisher(Path, "/pepper_vla/path", 10)

        # control loop runs faster than inference: tracks the last trajectory between updates
        self.timer = self.create_timer(0.05, self.control_step)   # 20 Hz

        self.get_logger().info(
            f"pepper_vla up | prompt='{self.prompt}' | server={self.url} | "
            f"max_speed={self.max_speed} | dry_run={self.dry_run}")

    # ---------------- perception -> waypoints ----------------
    def on_image(self, msg: Image):
        now = time.time()
        if now - self.last_req < self.infer_dt:
            return                                    # throttle to infer_hz
        self.last_req = now

        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
            if not ok:
                return
            payload = {
                "image_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "prompt": self.prompt,
                "seq": self.seq,
            }
            self.seq += 1
            r = self.session.post(self.url, json=payload, timeout=1.0)
            r.raise_for_status()
            out = r.json()
            if "waypoints" not in out:
                self.get_logger().warn(f"server: {out.get('error')}")
                return
            self.traj = np.asarray(out["waypoints"], dtype=float)   # (8,2)
            self.traj_stamp = time.time()
            self.publish_path(msg.header)
            self.get_logger().info(
                f"seq={out.get('seq')} lat={out.get('latency_ms')}ms "
                f"wp8=({self.traj[-1][0]:+.3f},{self.traj[-1][1]:+.3f})",
                throttle_duration_sec=1.0)
        except requests.exceptions.RequestException as e:
            self.get_logger().warn(f"server unreachable: {e}", throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().error(f"on_image: {type(e).__name__}: {e}")

    def publish_path(self, header):
        """Publish the predicted trajectory -- rviz now, occupancy module later."""
        p = Path()
        p.header.frame_id = "base_link"
        p.header.stamp = header.stamp
        for x, y in self.traj:
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)
        self.pub_path.publish(p)

    # ---------------- waypoints -> motion (pure pursuit) ----------------
    def control_step(self):
        cmd = Twist()

        if self.traj is None or (time.time() - self.traj_stamp) > self.stale_after:
            self.publish(cmd)                          # stale/no trajectory -> STOP
            return

        i = min(max(1, self.look_idx), len(self.traj)) - 1
        tx, ty = float(self.traj[i][0]), float(self.traj[i][1])

        if tx <= 0.01:                                 # target behind/at robot -> stop
            self.publish(cmd)
            return

        # speed: waypoint spacing encodes the demonstrator's speed.
        # waypoints are 0.1s apart (10Hz dataset), so |wp1| / 0.1 = m/s.
        if self.speed_from_model:
            step = float(np.hypot(self.traj[0][0], self.traj[0][1]))
            v = np.clip(step / 0.1, 0.0, self.max_speed)
        else:
            v = self.max_speed

        # pure pursuit curvature toward the lookahead point
        L2 = tx * tx + ty * ty
        curvature = 2.0 * ty / L2 if L2 > 1e-6 else 0.0
        omega = float(np.clip(v * curvature, -self.max_omega, self.max_omega))

        cmd.linear.x = float(v)
        cmd.angular.z = omega
        self.publish(cmd)

    def publish(self, cmd: Twist):
        if self.dry_run:
            self.get_logger().info(
                f"[dry_run] v={cmd.linear.x:+.3f} w={cmd.angular.z:+.3f}",
                throttle_duration_sec=0.5)
            return
        self.pub_cmd.publish(cmd)

    def stop(self):
        self.pub_cmd.publish(Twist())


def main():
    rclpy.init()
    node = PepperVLANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()                                    # always leave the robot stopped
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
