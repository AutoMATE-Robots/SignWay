"""Bridge test — runs the server (mock backend) in a thread and checks the client round-trip.
Proves the socket protocol, serialization, and Policy client without the model or a GPU. When
the server is launched with --backend omnivla on the GPU node, only the model load differs.
"""
import socket
import threading
import time

import numpy as np

from signway_backends.omni_server import serve
from signway_backends.omni_backend import make_backend
from signway_backends.policy_omnivla import OmniVLAClient


def _bound_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(4)
    return s, s.getsockname()[1]


def test_bridge_roundtrip():
    sock, port = _bound_socket()
    threading.Thread(target=serve, kwargs=dict(backend=make_backend("mock"), sock=sock),
                     daemon=True).start()
    time.sleep(0.2)

    client = OmniVLAClient(port=port)
    pong = client.ping()
    assert pong.get("ok") is True and pong.get("backend") == "mock"

    imgs = [np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8)]
    wp = client.predict_waypoints(imgs, (3.0, 1.0, 0.3))   # goal leans left
    assert wp.shape == (8, 2)
    assert wp[-1, 1] > 0                                    # chunk leans left toward the goal
