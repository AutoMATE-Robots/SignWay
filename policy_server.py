#!/usr/bin/env python3
import base64
import io
import os
import sys
import time

if not any("signway" in a.lower() for a in sys.argv):
    sys.argv.append("--signway")

for _k in ("OFT_REPO", "SIGNWAY_TOOLS", "SIGNWAY_DS"):
    _p = os.environ.get(_k)
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from eval_openloop import load_policy, predict

CKPT = os.environ["CKPT"]
H = 8
OK_PROMPTS = ("straight", "turn_left", "turn_right", "stop")

print("[server] loading:", CKPT, flush=True)
t0 = time.time()
POLICY = load_policy(CKPT)
print("[server] ready in %.1fs" % (time.time() - t0), flush=True)

WARM = np.zeros((480, 640, 3), dtype=np.uint8)
predict(POLICY, WARM, "straight", H)
print("[server] warmup done -- listening", flush=True)

app = Flask(__name__)


@app.post("/health")
def health():
    t = time.time()
    traj = predict(POLICY, WARM, "straight", H)
    ms = round(1000 * (time.time() - t), 1)
    wp = np.asarray(traj).tolist()
    return jsonify(ok=True, latency_ms=ms, sample=wp)


@app.post("/predict")
def go():
    t = time.time()
    try:
        d = request.get_json(force=True)
        p = d.get("prompt", "straight")
        if p not in OK_PROMPTS:
            return jsonify(error="bad prompt"), 400
        raw = base64.b64decode(d["image_b64"])
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        img = np.asarray(im, dtype=np.uint8)
        traj = np.asarray(predict(POLICY, img, p, H), dtype=float)
        ms = round(1000 * (time.time() - t), 1)
        return jsonify(waypoints=traj.tolist(), prompt=p,
                       seq=d.get("seq"), latency_ms=ms)
    except Exception as e:
        return jsonify(error=repr(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=False)
