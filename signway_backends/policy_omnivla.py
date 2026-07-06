"""OmniVLA policy backend that calls the omni_server over TCP.

Implements signway_core.interfaces.Policy, so the FSM and sim can't tell the model is remote —
this is what keeps the habitat (3.9) and omnivla (3.10) envs cleanly separated. Runs in the
habitat env; signway_core is pure Python so it imports fine there.

    from signway_backends.policy_omnivla import OmniVLAClient
    policy = OmniVLAClient(host="127.0.0.1", port=5555)
    fsm = FSM(policy, safety, cfg)

Quick connectivity check against a running server:
    python -m signway_backends.policy_omnivla --ping
"""
from __future__ import annotations

import socket
from typing import Tuple

import numpy as np

from signway_core.interfaces import Policy
from signway_backends._wire import recv_msg, send_msg


class OmniVLAClient(Policy):
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float = 30.0):
        self.host, self.port, self.timeout = host, port, timeout

    def _rpc(self, obj):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        try:
            send_msg(s, obj)
            return recv_msg(s)
        finally:
            s.close()

    def ping(self):
        return self._rpc({"cmd": "ping"})

    def predict_waypoints(self, images, goal_robot: Tuple[float, float, float]) -> np.ndarray:
        resp = self._rpc({"images": list(images), "goal_robot": tuple(goal_robot)})
        if "error" in resp:
            raise RuntimeError(f"omni_server error: {resp['error']}")
        return np.asarray(resp["wp"], float).reshape(-1, 2)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--ping", action="store_true", help="just check connectivity")
    a = ap.parse_args()

    c = OmniVLAClient(host=a.host, port=a.port)
    if a.ping:
        print("ping ->", c.ping())
    else:
        img = np.zeros((64, 64, 3), np.uint8)
        wp = c.predict_waypoints([img, img], (3.0, 1.0, 0.3))
        print("waypoints (8x2):\n", np.round(wp, 3))
