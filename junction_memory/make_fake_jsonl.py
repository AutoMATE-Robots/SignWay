#!/usr/bin/env python3
"""
make_fake_jsonl.py — synthesize a dump_features-style evidence JSONL for the
synthetic demo bag, so the LIVE wiring (real EvidenceGate + Reasoner) can be
exercised end-to-end without running docTR over a real bag.

Record schema matches what render_video.py reads:
    {frame, plate_id, R, ell, R_struct, text, box}

Evidence rises as the robot closes on node A on each of its two approaches,
which is what makes the gate arm and then fire.

    python make_fake_jsonl.py --bag demo/demo_square_bag \
        --odom-csv demo/odom_calibrated.csv --out demo/square_evidence.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
try:                                    # run from the repo root
    from junction_memory.replay_ar_memory import (open_bag,
                                                  pick_image_connection)
except ImportError:                     # run from inside junction_memory/
    from replay_ar_memory import open_bag, pick_image_connection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--odom-csv", required=True)
    ap.add_argument("--topic", default="/c1/image_raw")
    ap.add_argument("--s-first", type=float, default=25.0,
                    help="arc length at the first arrival at A")
    ap.add_argument("--tail", type=float, default=8.0,
                    help="metres travelled past A on the second pass")
    ap.add_argument("--visible-from", type=float, default=20.0)
    ap.add_argument("--visible-to", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    od = np.loadtxt(a.odom_csv, delimiter=",", skiprows=1)
    ot, ox, oy = od[:, 0], od[:, 1], od[:, 2]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ox), np.diff(oy)))])
    s_a1, s_a2 = a.s_first, float(s[-1] - a.tail)

    reader = open_bag(a.bag)
    conns = pick_image_connection(reader, a.topic)
    rows = []
    try:
        for i, (conn, tns, raw) in enumerate(reader.messages(connections=conns)):
            ti = tns * 1e-9
            s_now = float(np.interp(ti, ot, s))
            for tag, s_node in (("A1", s_a1), ("A2", s_a2)):
                d = s_node - s_now
                if not (a.visible_to < d < a.visible_from):
                    continue
                # closer sign -> more relevant text read and more legible
                frac = 1.0 - (d - a.visible_to) / (a.visible_from - a.visible_to)
                R = float(np.clip(0.35 + 0.62 * frac, 0.0, 0.97))
                ell = float(np.clip(0.20 + 0.78 * frac, 0.0, 0.97))
                px = 160 + 18 * frac
                half = 8 + 52 * frac
                rows.append({
                    "frame": i, "plate_id": f"plate_{tag}",
                    "R": round(R, 3), "ell": round(ell, 3), "R_struct": 1.0,
                    "text": "goal_A <- / goal_E ^",
                    "box": [round(px - half, 1), round(96 - 0.45 * half, 1),
                            round(px + half, 1), round(96 + 0.45 * half, 1)],
                })
    finally:
        reader.close()

    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    frames = sorted({r["frame"] for r in rows})
    print(f"wrote {a.out}: {len(rows)} records over {len(frames)} frames "
          f"(first {frames[0]}, last {frames[-1]})")
    for tag in ("A1", "A2"):
        f = [r["frame"] for r in rows if r["plate_id"] == f"plate_{tag}"]
        q = [round(r["R"] * r["ell"], 2) for r in rows
             if r["plate_id"] == f"plate_{tag}"]
        print(f"  plate_{tag}: frames {f[0]}-{f[-1]}, q {min(q):.2f}..{max(q):.2f}")


if __name__ == "__main__":
    main()
