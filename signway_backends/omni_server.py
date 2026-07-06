"""OmniVLA inference server.

Runs in the `omnivla` env on the GPU node. Loads the model once and serves waypoint chunks over
TCP so the sim (in the habitat env, Python 3.9) can call it without sharing an env. One request
per connection — simple and robust; at ~0.5 Hz the reconnect cost is nothing.

    # real model (point --omnivla-dir at the OmniVLA fork's inference/ dir):
    python -m signway_backends.omni_server --backend omnivla \
        --omnivla-dir ~/SignNav/inference --host 127.0.0.1 --port 5555

    # bridge test, no model:
    python -m signway_backends.omni_server --backend mock --port 5555

Request : {"images": [np.ndarray, ...], "goal_robot": (fwd, left, dtheta)}  or  {"cmd": "ping"}
Response: {"wp": np.ndarray(N,2), "ms": float}  or  {"ok": True, "backend": str}  or  {"error": str}
"""
from __future__ import annotations

import argparse
import os
import socket
import time
import traceback

import numpy as np

from signway_backends.omni_backend import make_backend
from signway_backends._wire import recv_msg, send_msg


def handle(conn, backend) -> None:
    try:
        req = recv_msg(conn)
        if isinstance(req, dict) and req.get("cmd") == "ping":
            send_msg(conn, {"ok": True, "backend": backend.name})
            return
        images = req["images"]
        goal = tuple(req["goal_robot"])
        t0 = time.time()
        wp = np.asarray(backend.predict_waypoints(images, goal), float).reshape(-1, 2)
        send_msg(conn, {"wp": wp, "ms": (time.time() - t0) * 1e3})
    except Exception as e:  # noqa: BLE001 — report any failure back to the client
        traceback.print_exc()
        try:
            send_msg(conn, {"error": repr(e)})
        except Exception:
            pass
    finally:
        conn.close()


def serve(backend, host: str = "127.0.0.1", port: int = 5555, sock=None, on_ready=None) -> None:
    if sock is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(4)
    actual_port = sock.getsockname()[1]
    if on_ready:
        on_ready(actual_port)
    print(f"[omni_server] backend={backend.name} listening on {host}:{actual_port}", flush=True)
    while True:
        conn, _addr = sock.accept()
        handle(conn, backend)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="omnivla", choices=["omnivla", "mock"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--omnivla-dir", default=None,
                    help="path to the OmniVLA fork's inference/ dir (so `import run_omnivla` works)")
    ap.add_argument("--checkpoint", default=None, help="path to the omnivla-original checkpoint dir")
    args = ap.parse_args()

    if args.omnivla_dir:
        import sys
        sys.path.insert(0, os.path.abspath(args.omnivla_dir))

    kw = {}
    if args.checkpoint:
        kw["checkpoint_dir"] = args.checkpoint
    elif args.omnivla_dir:
        kw["checkpoint_dir"] = os.path.join(os.path.abspath(args.omnivla_dir), "omnivla-original")

    backend = make_backend(args.backend, **kw)
    serve(backend, args.host, args.port)


if __name__ == "__main__":
    main()
