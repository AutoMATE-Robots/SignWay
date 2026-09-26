#!/usr/bin/env python3
"""
replay_ar_memory.py
-------------------
Adaptive-Reasoning replay with junction memory as a live sub-component.
Streams a ROS 2 bag (or a pre-extracted odom CSV) through the ONLINE memory
runtime + an annotation-driven AR gate, and renders an mp4 where every memory
event is captioned as it happens:

    "LATERAL BEAM lock on n_001 — 22.4 m out"
    "corridor e_001 recognised (same direction) => approaching n_001"
    "MEMORY RECALL at n_001: goal_E -> STRAIGHT ... VLM call not needed"
    "AR gate arming point reached but necessity is ZERO ..."

Fast loop first (seconds, odom only), video second (reads the image topic):

    # 1. fast validation — no images touched, seconds to run
    python replay_ar_memory.py --csv odom_calibrated.csv \
        --annotations square_bag.yaml --no-video --out out_fast

    # 2. full replay video from the 35 GB bag (odom from the calibrated CSV)
    python replay_ar_memory.py --bag /path/rosbag2_2026_08_22-20_29_37 \
        --csv odom_calibrated.csv --annotations square_bag.yaml --out out_video

Odometry preference order: --csv (already yaw-calibrated by calibrate_yaw.py)
else the bag's odom topic with --yaw-scale applied online. The bag reader
inherits odom_memory_probe's typestore fallback, so bags written without
embedded message definitions (README §7.3) work unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Importable either as a flat script (python replay_ar_memory.py ...) or as
# part of a package (from junction_memory.replay_ar_memory import HUD), so the
# same file serves the standalone toolchain and the in-repo AR integration.
try:
    from memory_runtime import (MemoryRuntime, YawScaler, wrap_pi, LATERAL_M,
                                PROXIMITY_R)
except ImportError:                                     # imported as a package
    from .memory_runtime import (MemoryRuntime, YawScaler, wrap_pi, LATERAL_M,
                                 PROXIMITY_R)

try:                            # room number -> printed range ("6-118" is
    from goal_matcher import match as _goal_match       # inside "6-115 to
except ImportError:             # 6-189"); same wiring as render_video_memory
    try:
        from .goal_matcher import match as _goal_match
    except ImportError:
        _goal_match = None

try:                                   # colocated colleague tooling (preferred)
    import odom_memory_probe as P
except Exception:                      # minimal inline fallback
    P = None


def _gate_shim():
    """Imported lazily: only the standalone CLI needs the annotation shim."""
    try:
        from ar_gate_shim import ARGateShim
    except ImportError:
        from .ar_gate_shim import ARGateShim
    return ARGateShim


# ----------------------------------------------------------------------------
# bag access (typestore fallback identical to odom_memory_probe, §7.3)
# ----------------------------------------------------------------------------
def _guess_typestore(bag_path):
    if P is not None:
        return P._guess_typestore(bag_path)
    import sqlite3, glob, re
    from rosbags.typesys import Stores, get_typestore
    name = None
    for db in glob.glob(str(Path(bag_path) / "*.db3")):
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            row = c.execute("SELECT ros_distro FROM schema").fetchone()
            c.close()
            if row and row[0]:
                name = str(row[0]).strip().upper()
                break
        except Exception:
            pass
    if name is None:
        try:
            m = re.search(r"ros_distro:\s*(\w+)",
                          (Path(bag_path) / "metadata.yaml").read_text())
            if m:
                name = m.group(1).upper()
        except Exception:
            pass
    key = f"ROS2_{name}" if name else None
    if key and hasattr(Stores, key):
        print(f"[info] bag has no embedded typedefs; using typestore {key}")
        return get_typestore(getattr(Stores, key))
    print("[info] bag has no embedded typedefs; defaulting to ROS2_HUMBLE")
    return get_typestore(Stores.ROS2_HUMBLE)


def open_bag(bag_path):
    from rosbags.highlevel import AnyReader
    try:
        r = AnyReader([Path(bag_path)])
        r.open()
        return r
    except Exception as e:
        if "type definitions" not in str(e):
            raise
        r = AnyReader([Path(bag_path)], default_typestore=_guess_typestore(bag_path))
        r.open()
        return r


def quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_odom_from_bag(bag_path, topic=None):
    if P is not None:
        return P.load_odom_bag(bag_path, topic)
    reader = open_bag(bag_path)
    rows = []
    try:
        conns = [c for c in reader.connections
                 if c.msgtype == "nav_msgs/msg/Odometry"
                 and (topic is None or c.topic == topic)]
        if not conns:
            raise SystemExit("No nav_msgs/msg/Odometry in bag")
        for conn, tns, raw in reader.messages(connections=conns[:1]):
            m = reader.deserialize(raw, conn.msgtype)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            rows.append((tns * 1e-9, p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w)))
    finally:
        reader.close()
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def pick_image_connection(reader, topic_hint):
    img_types = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
    conns = [c for c in reader.connections if c.msgtype in img_types]
    if topic_hint:
        exact = [c for c in conns if c.topic == topic_hint]
        pref = exact or [c for c in conns if c.topic.startswith(topic_hint)]
        if pref:
            conns = pref
        else:
            avail = sorted({(c.topic, c.msgtype) for c in reader.connections})
            raise SystemExit(f"image topic {topic_hint!r} not found. Topics:\n  "
                             + "\n  ".join(f"{t} [{m}]" for t, m in avail))
    if not conns:
        raise SystemExit("no image topic in bag — use --no-video")
    conns.sort(key=lambda c: (0 if c.topic.startswith("/c1") else 1, c.topic))
    if len({c.topic for c in conns}) > 1:
        print(f"[warn] multiple image topics; using {conns[0].topic}")
    return [c for c in conns if c.topic == conns[0].topic]


def decode_image(msg, msgtype):
    import cv2
    if msgtype.endswith("CompressedImage"):
        buf = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    enc = (msg.encoding or "bgr8").lower()
    h, w = msg.height, msg.width
    data = np.frombuffer(msg.data, np.uint8)
    if enc in ("bgr8", "rgb8"):
        img = data.reshape(h, msg.step // 1)[:, : w * 3].reshape(h, w, 3)
        if enc == "rgb8":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)
    if enc in ("bgra8", "rgba8"):
        img = data.reshape(h, -1)[:, : w * 4].reshape(h, w, 4)
        img = img[:, :, [2, 1, 0]] if enc == "rgba8" else img[:, :, :3]
        return np.ascontiguousarray(img)
    if enc in ("mono8", "8uc1"):
        img = data.reshape(h, -1)[:, :w]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc in ("mono16", "16uc1"):
        img16 = np.frombuffer(msg.data, np.uint16).reshape(h, -1)[:, :w]
        img = (img16 / max(img16.max(), 1) * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise SystemExit(f"unsupported image encoding {enc!r}")


# ----------------------------------------------------------------------------
# HUD renderer
# ----------------------------------------------------------------------------

def _ascii(s):
    """cv2 Hershey fonts are ASCII-only; keep captions readable."""
    return (str(s).replace("\u2014", "-").replace("\u2013", "-")
            .replace("\u2192", "->").replace("\u00b0", " deg")
            .encode("ascii", "replace").decode().replace("?", " ")
            if any(ord(c) > 126 for c in str(s)) else str(s))

CLR_BG = (26, 26, 30)
CLR_PANEL = (38, 38, 44)
CLR_TEXT = (235, 235, 235)
CLR_DIM = (150, 150, 155)
CLR_TRACE = (95, 95, 100)
CLR_EDGE = (235, 170, 90)        # stored corridors (light blue, BGR)
CLR_EDGE_SAME = (120, 220, 120)  # matched same-direction
CLR_EDGE_REV = (90, 90, 235)     # matched reverse
CLR_NODE = (0, 165, 255)         # orange
CLR_MEM = (110, 220, 130)        # memory green
CLR_VLM = (0, 150, 255)          # vlm orange
CLR_WARN = (80, 80, 230)

MAJOR_BANNER_KINDS = {
    "BEAM_LOCK", "EDGE_MATCH", "MEMORY_RECALL", "VLM_CALL", "GATE_SUPPRESSED",
    "GATE_CALL_CANCELLED", "NODE_VISIT_STRAIGHT", "NODE_CREATED",
    "NODE_REVISIT", "DIRECTORY_STORED", "GOAL_SWITCH", "VERIFY_MISMATCH",
}


class HUD:
    W, H = 1280, 720
    CAM = (8, 8, 872, 656)          # x, y, w, h
    MAP = (888, 8, 384, 384)
    STAT = (888, 400, 384, 262)
    TICK = (8, 668, 1264, 46)

    def __init__(self, bounds, rt, gate):
        import cv2
        self.cv2 = cv2
        self.rt, self.gate = rt, gate
        (x0, x1, y0, y1) = bounds
        pad = 2.0
        mx, my, mw, mh = self.MAP
        sx = (mw - 20) / max(x1 - x0 + 2 * pad, 1e-6)
        sy = (mh - 20) / max(y1 - y0 + 2 * pad, 1e-6)
        self.scale = min(sx, sy)
        self.wx0, self.wy1 = x0 - pad, y1 + pad
        self.banner = None            # (msg, color, until_t)
        self.stars = []               # (x, y, kind, until_t)
        self.tick_hist = []

    def w2p(self, x, y):
        mx, my, mw, mh = self.MAP
        px = mx + 10 + (x - self.wx0) * self.scale
        py = my + 10 + (self.wy1 - y) * self.scale
        return int(px), int(py)

    # ------------------------------------------------------------------
    def ingest_events(self, evs, t_now):
        for e in evs:
            if e.kind in MAJOR_BANNER_KINDS or e.severity in ("major", "warn"):
                color = CLR_MEM if e.kind in ("MEMORY_RECALL", "GATE_SUPPRESSED",
                                              "GATE_CALL_CANCELLED") else \
                        CLR_VLM if e.kind in ("VLM_CALL", "DIRECTORY_STORED") else \
                        CLR_WARN if e.severity == "warn" else (200, 200, 210)
                self.banner = (e.msg, color, t_now + 2.6)
            self.tick_hist.append((e.msg, e.severity))
            self.tick_hist = self.tick_hist[-3:]
            if e.kind == "MEMORY_RECALL" and self.rt.t:
                self.stars.append((self.rt.x[-1], self.rt.y[-1], "recall",
                                   t_now + 6.0))
            if e.kind == "VLM_CALL" and self.rt.t:
                self.stars.append((self.rt.x[-1], self.rt.y[-1], "vlm",
                                   t_now + 6.0))

    # ------------------------------------------------------------------
    def render(self, cam_bgr, t_now, t_rel, frame_idx):
        cv2 = self.cv2
        c = np.full((self.H, self.W, 3), CLR_BG, np.uint8)
        # camera
        cx, cy, cw, ch = self.CAM
        if cam_bgr is not None:
            ih, iw = cam_bgr.shape[:2]
            s = min(cw / iw, ch / ih)
            nw, nh = int(iw * s), int(ih * s)
            img = cv2.resize(cam_bgr, (nw, nh))
            ox, oy = cx + (cw - nw) // 2, cy + (ch - nh) // 2
            c[oy:oy + nh, ox:ox + nw] = img
        src = self.gate.prompt_source if self.gate else "default"
        bcol = CLR_MEM if src == "memory" else CLR_VLM if src == "vlm" else (90, 90, 95)
        cv2.rectangle(c, (cx - 2, cy - 2), (cx + cw + 1, cy + ch + 1), bcol, 2)

        self._minimap(c)
        self._status(c, t_rel, frame_idx)
        self._ticker(c)
        self._banner(c, t_now)
        return c

    # ------------------------------------------------------------------
    def _minimap(self, c):
        cv2 = self.cv2
        mx, my, mw, mh = self.MAP
        cv2.rectangle(c, (mx, my), (mx + mw, my + mh), CLR_PANEL, -1)
        cv2.rectangle(c, (mx, my), (mx + mw, my + mh), (70, 70, 78), 1)
        rt = self.rt
        if not rt.t:
            return
        # past trace (decimated)
        pts = []
        step = max(1, len(rt.x) // 1200)
        for i in range(0, len(rt.x), step):
            pts.append(self.w2p(rt.x[i], rt.y[i]))
        if len(pts) > 1:
            cv2.polylines(c, [np.array(pts, np.int32)], False, CLR_TRACE, 1,
                          cv2.LINE_AA)
        # stored edges
        active = {}
        for eid, et in rt.etracks.items():
            if et.announced and et.us:
                same = None
                for e in reversed(rt.events):
                    if e.kind == "EDGE_MATCH" and e.data.get("edge") == eid:
                        same = e.data.get("same_direction")
                        break
                active[eid] = same
        for eid, rec in rt.edges.items():
            col = CLR_EDGE
            th = 2
            if eid in active:
                col = CLR_EDGE_SAME if active[eid] else CLR_EDGE_REV
                th = 3
            ep = [self.w2p(px, py) for px, py in rec.pts[::2]]
            if len(ep) > 1:
                cv2.polylines(c, [np.array(ep, np.int32)], False, col, th,
                              cv2.LINE_AA)
        # beam ribbon toward candidate/locked nodes
        rx, ry = rt.x[-1], rt.y[-1]
        for nid, bt in rt.beams.items():
            if bt.run.start_s is None and not bt.locked:
                continue
            node = rt.nodes[nid]
            p0, p1 = self.w2p(rx, ry), self.w2p(node.x, node.y)
            band = c.copy()
            width = max(3, int(2 * LATERAL_M * self.scale))
            col = CLR_MEM if bt.locked else (100, 160, 110)
            cv2.line(band, p0, p1, col, width, cv2.LINE_AA)
            cv2.addWeighted(band, 0.30, c, 0.70, 0, c)
            cv2.line(c, p0, p1, col, 1, cv2.LINE_AA)
        # nodes
        for nid, n in rt.nodes.items():
            p = self.w2p(n.x, n.y)
            filled = -1 if n.routing else 2
            cv2.circle(c, p, 6, CLR_NODE, filled, cv2.LINE_AA)
            if n.routing:
                cv2.circle(c, p, 9, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(c, p, int(PROXIMITY_R * self.scale), (110, 110, 120),
                           1, cv2.LINE_AA)
            cv2.putText(c, nid.replace("n_00", "n"), (p[0] + 8, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_DIM, 1, cv2.LINE_AA)
        # event stars
        t_now = rt.t[-1]
        self.stars = [s for s in self.stars if s[3] > t_now]
        for sx, sy, kind, _ in self.stars:
            p = self.w2p(sx, sy)
            col = CLR_MEM if kind == "recall" else CLR_VLM
            cv2.drawMarker(c, p, col, cv2.MARKER_STAR, 16, 2, cv2.LINE_AA)
        # robot
        p = self.w2p(rx, ry)
        psi = rt.yaw_u[-1]
        tip = self.w2p(rx + 1.6 * math.cos(psi), ry + 1.6 * math.sin(psi))
        cv2.arrowedLine(c, p, tip, (255, 255, 255), 2, cv2.LINE_AA,
                        tipLength=0.45)

    # ------------------------------------------------------------------
    def _chip(self, c, x, y, text, color, filled=True):
        cv2 = self.cv2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        if filled:
            cv2.rectangle(c, (x, y - th - 6), (x + tw + 12, y + 6), color, -1)
            cv2.putText(c, text, (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (15, 15, 15), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(c, (x, y - th - 6), (x + tw + 12, y + 6), color, 1)
            cv2.putText(c, text, (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        color, 1, cv2.LINE_AA)
        return x + tw + 20

    def _status(self, c, t_rel, frame_idx):
        cv2 = self.cv2
        sx, sy, sw, sh = self.STAT
        cv2.rectangle(c, (sx, sy), (sx + sw, sy + sh), CLR_PANEL, -1)
        cv2.rectangle(c, (sx, sy), (sx + sw, sy + sh), (70, 70, 78), 1)
        rt, gate = self.rt, self.gate
        y = sy + 26
        goal = getattr(rt, "current_goal", None) or "-"
        self._chip(c, sx + 10, y, f"GOAL {goal}", (200, 200, 120))
        y += 34
        if gate:
            st = gate.state
            col = {"NO_SIGN": (120, 120, 125), "ARMED": (90, 190, 230),
                   "PENDING": (60, 170, 250), "DECIDED": CLR_VLM,
                   "SUPPRESSED": CLR_MEM}.get(st, CLR_DIM)
            nx = self._chip(c, sx + 10, y, f"AR GATE {st}", col)
            hit = rt.resolves(goal) if goal != "-" else None
            infl = getattr(gate, "in_flight", None)
            if infl is not None:
                nec = (f"VLM IN FLIGHT {infl[0]:.1f}s - {infl[1]:.1f} m "
                       f"travelled, no decision yet")
            else:
                nec = "necessity ZERO (memory)" if hit else \
                      ("necessity pending" if st in ("ARMED", "PENDING")
                       else "necessity idle")
            ncol = (CLR_VLM if getattr(gate, "in_flight", None) is not None
                    else (CLR_MEM if hit else CLR_DIM))
            cv2.putText(c, nec, (sx + 12, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, ncol, 1, cv2.LINE_AA)
            y += 52
            src = gate.prompt_source.upper()
            scol = CLR_MEM if src == "MEMORY" else CLR_VLM if src == "VLM" \
                else (120, 120, 125)
            self._chip(c, sx + 10, y,
                       f"PROMPT {str(gate.standing_prompt).upper()}  [{src}]",
                       scol)
            y += 34
            s = gate.summary()
            cv2.putText(c, f"VLM calls: {s['vlm_calls']}   "
                           f"always-invoke: {s['counterfactual_always_invoke']}"
                           f"   saved: {s['calls_saved']}",
                        (sx + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        CLR_TEXT, 1, cv2.LINE_AA)
            y += 26
        exp = rt.expected_node
        if exp and exp in rt.nodes and rt.t:
            n = rt.nodes[exp]
            rng = math.hypot(n.x - rt.x[-1], n.y - rt.y[-1])
            leg = rt.legibility_prior()
            legtxt = "" if leg is None else \
                ("  | sign: readable" if leg else "  | sign: not visible")
            cv2.putText(c, f"expected: {exp} ({rng:.1f} m){legtxt}",
                        (sx + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        CLR_MEM, 1, cv2.LINE_AA)
            y += 26
        st = rt.standing
        if st:
            cv2.putText(c, f"standing: {st['goal']} -> "
                           f"{st['action'].upper()} ({st['source']}, "
                           f"lead {st['lead_m']:.0f} m)",
                        (sx + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        CLR_MEM, 1, cv2.LINE_AA)
            y += 24
        cv2.putText(c, f"t+{t_rel:7.1f}s  s={rt.s[-1] if rt.s else 0:7.1f}m  "
                       f"frame {frame_idx}  nodes {len(rt.nodes)} "
                       f"edges {len(rt.edges)}",
                    (sx + 12, sy + sh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    CLR_DIM, 1, cv2.LINE_AA)

    def _ticker(self, c):
        cv2 = self.cv2
        tx, ty, tw, th = self.TICK
        cv2.rectangle(c, (tx, ty), (tx + tw, ty + th), CLR_PANEL, -1)
        yy = ty + 16
        for i, (msg, sev) in enumerate(reversed(self.tick_hist)):
            col = CLR_TEXT if i == 0 else CLR_DIM
            if sev == "warn":
                col = CLR_WARN
            cv2.putText(c, _ascii(msg)[:150], (tx + 8, yy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, col, 1, cv2.LINE_AA)
            yy += 15

    def _banner(self, c, t_now):
        if not self.banner:
            return
        msg, col, until = self.banner
        if t_now > until:
            self.banner = None
            return
        cv2 = self.cv2
        cx, cy, cw, ch = self.CAM
        band = c.copy()
        cv2.rectangle(band, (cx, cy), (cx + cw, cy + 64), (12, 12, 14), -1)
        cv2.addWeighted(band, 0.72, c, 0.28, 0, c)
        cv2.rectangle(c, (cx, cy), (cx + 6, cy + 64), col, -1)
        # wrap to two lines
        words, lines, cur = _ascii(msg).split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 78 and cur:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        for i, ln in enumerate(lines[:2]):
            cv2.putText(c, _ascii(ln), (cx + 16, cy + 26 + 26 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------
def write_outputs(out, rt, gate, args):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "events.jsonl", "w") as f:
        for e in rt.events:
            f.write(json.dumps(e.jsonable()) + "\n")
    rt.save(out / "graph.json")

    lines = []
    lines.append(f"bag: {rt.bag}")
    if rt.s:
        lines.append(f"path length: {rt.s[-1]:.1f} m, "
                     f"duration: {rt.t[-1] - rt.t[0]:.1f} s, "
                     f"odom samples: {len(rt.t)}")
    lines.append(f"graph: {len(rt.nodes)} nodes, {len(rt.edges)} edges")
    if gate:
        s = gate.summary()
        lines.append("")
        lines.append(f"VLM calls with memory:      {s['vlm_calls']}")
        lines.append(f"always-invoke counterfact.: {s['counterfactual_always_invoke']}")
        lines.append(f"calls saved:                {s['calls_saved']}")
        lines.append(f"reuse rate:                 {s['reuse_rate']:.2f}")
    recs = [e for e in rt.events if e.kind == "MEMORY_RECALL"]
    prox = {e.data["node"]: e.data["range_m"] for e in rt.events
            if e.kind == "PROXIMITY_FIRE"}
    if recs:
        lines.append("")
        lines.append("memory recalls (lead = metres before the junction):")
        for e in recs:
            d = e.data
            base = prox.get(d["node"])
            cmp_ = f"  (proximity baseline: {base:.1f} m)" if base else ""
            lines.append(f"  {d['node']}  goal={d['goal']}  ->{d['action']}"
                         f"  via {d['source']}  lead {d['lead_m']:.1f} m"
                         f"  flip margin {d['flip_margin_deg']:.0f} deg{cmp_}")
    lines.append("")
    counts = {}
    for e in rt.events:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    lines.append("event counts: " + ", ".join(f"{k}={v}" for k, v in
                                              sorted(counts.items())))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    _final_map(out / "final_map.png", rt)


def _final_map(path, rt):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.plot(rt.x, rt.y, lw=0.8, color="#999", label="trajectory", zorder=1)
        for eid, rec in rt.edges.items():
            ax.plot(rec.pts[:, 0], rec.pts[:, 1], lw=2.2, alpha=0.75,
                    color="#1f77b4", zorder=2)
            mid = rec.pts[len(rec.pts) // 2]
            ax.annotate(eid.replace("e_00", "e"), mid, fontsize=7,
                        color="#1f77b4", textcoords="offset points",
                        xytext=(3, 3))
        for nid, n in rt.nodes.items():
            face = "darkorange" if n.routing else "none"
            ax.scatter([n.x], [n.y], s=170, facecolors=face,
                       edgecolors="darkorange", lw=2, zorder=5)
            ax.annotate(nid.replace("n_00", "n"), (n.x, n.y), fontsize=9,
                        textcoords="offset points", xytext=(8, 8))
        for e in rt.events:
            i = np.searchsorted(rt.s, e.s)
            i = min(i, len(rt.x) - 1)
            if e.kind == "MEMORY_RECALL":
                ax.scatter([rt.x[i]], [rt.y[i]], marker="*", s=260, c="green",
                           zorder=6, label="memory recall")
            elif e.kind == "VLM_CALL":
                ax.scatter([rt.x[i]], [rt.y[i]], marker="^", s=140,
                           c="darkorange", zorder=6, label="VLM call")
            elif e.kind == "BEAM_LOCK":
                ax.scatter([rt.x[i]], [rt.y[i]], marker=".", s=90, c="green",
                           zorder=4)
            elif e.kind == "PROXIMITY_FIRE":
                ax.scatter([rt.x[i]], [rt.y[i]], marker="x", s=70, c="#888",
                           zorder=4)
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=8, loc="best")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("junction memory — final graph and decision sources")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[info] wrote {path}")
    except Exception as e:
        print(f"[warn] final map failed: {e}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", help="rosbag2 directory (images and/or odom)")
    ap.add_argument("--csv", help="odom CSV (t,x,y,yaw) — use the CALIBRATED "
                                  "one from calibrate_yaw.py")
    ap.add_argument("--annotations", help="per-bag YAML (goals + sign windows)")
    ap.add_argument("--out", default="out_replay")
    ap.add_argument("--odom-topic", default=None)
    ap.add_argument("--image-topic", default=None,
                    help="default: auto (prefers /c1*)")
    ap.add_argument("--yaw-scale", type=float, default=1.0,
                    help="rotational scale fix k (from calibrate_yaw). Applied "
                         "to bag odom; ignored when --csv is given")
    ap.add_argument("--no-video", action="store_true",
                    help="fast loop: odom + events only, no image decoding")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth image frame (memory still sees "
                         "every odom sample)")
    ap.add_argument("--start-s", type=float, default=None)
    ap.add_argument("--end-s", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None,
                    help="output video fps (default: measured / stride)")
    ap.add_argument("--verify-tau", type=float, default=None,
                    help="re-read when cached conf below tau AND sign legible")
    args = ap.parse_args()

    if not args.bag and not args.csv:
        ap.error("need --bag and/or --csv")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stills").mkdir(exist_ok=True)

    # ---------------- odometry (prepass) ----------------
    if args.csv:
        a = np.loadtxt(args.csv, delimiter=",", skiprows=1)
        t, x, y, yaw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        print(f"[info] odom from CSV: {len(t)} samples "
              f"({'calibrated' if 'calib' in args.csv else 'as-is'})")
        if args.yaw_scale != 1.0:
            print("[warn] --yaw-scale ignored: CSV assumed already calibrated")
    else:
        print("[info] extracting odom from bag (images are skipped — fast)")
        t, x, y, yaw = load_odom_from_bag(args.bag, args.odom_topic)
        if args.yaw_scale != 1.0:
            sc = YawScaler(args.yaw_scale)
            xc, yc, yc2 = [], [], []
            for i in range(len(t)):
                a_, b_, c_ = sc.step(x[i], y[i], yaw[i])
                xc.append(a_); yc.append(b_); yc2.append(c_)
            x, y, yaw = np.array(xc), np.array(yc), np.array(yc2)
            print(f"[info] applied yaw scale k={args.yaw_scale}")
        else:
            print("[warn] no --yaw-scale: if this bag has the ~20%% rotational "
                  "error, run calibrate_yaw.py first (README §7.1)")

    bounds = (float(np.min(x)), float(np.max(x)),
              float(np.min(y)), float(np.max(y)))
    t0 = float(t[0])

    rt = MemoryRuntime(bag_name=Path(args.bag or args.csv).name,
                       verify_tau=args.verify_tau, goal_matcher=_goal_match)
    gate = _gate_shim()(args.annotations, rt) if args.annotations else None
    if gate is None:
        print("[warn] no --annotations: memory will map the run but no VLM "
              "reads exist to store or recall")

    # ---------------- fast loop (no video) ----------------
    if args.no_video or not args.bag:
        t_start = time.time()
        for i in range(len(t)):
            if gate:
                gate.step(frame=None, t_rel=float(t[i] - t0))
            rt.step(float(t[i]), float(x[i]), float(y[i]), float(yaw[i]))
        print(f"[info] fast loop done in {time.time() - t_start:.1f} s")
        write_outputs(out, rt, gate, args)
        return

    # ---------------- video loop ----------------
    import cv2

    reader = open_bag(args.bag)
    img_conns = pick_image_connection(reader, args.image_topic)
    n_imgs = sum(c.msgcount for c in img_conns)
    print(f"[info] image topic {img_conns[0].topic} "
          f"({img_conns[0].msgtype.split('/')[-1]}, {n_imgs} frames)")

    hud = HUD(bounds, rt, gate)
    writer = None
    fps = args.fps
    k = 0                      # odom cursor
    frame_idx = -1
    rendered = 0
    t_prev_img = None
    dt_est = []
    t_wall = time.time()
    still_kinds_seen = set()

    try:
        for conn, tns, raw in reader.messages(connections=img_conns):
            ti = tns * 1e-9
            # advance memory over every odom sample up to this image
            while k < len(t) and t[k] <= ti:
                if gate:
                    gate.step(frame=frame_idx, t_rel=float(t[k] - t0))
                evs = rt.step(float(t[k]), float(x[k]), float(y[k]),
                              float(yaw[k]))
                hud.ingest_events(evs, t_now=float(t[k]))
                k += 1
            frame_idx += 1
            rt.set_frame(frame_idx)
            t_rel = ti - t0
            if gate:
                gate.step(frame=frame_idx, t_rel=t_rel)

            if args.start_s is not None and t_rel < args.start_s:
                continue
            if args.end_s is not None and t_rel > args.end_s:
                break
            if args.max_frames is not None and rendered >= args.max_frames:
                break
            if frame_idx % args.stride:
                continue

            msg = reader.deserialize(raw, conn.msgtype)
            img = decode_image(msg, conn.msgtype)
            if t_prev_img is not None:
                dt_est.append(ti - t_prev_img)
            t_prev_img = ti
            canvas = hud.render(img, t_now=ti, t_rel=t_rel,
                                frame_idx=frame_idx)
            if writer is None:
                if fps is None:
                    med = np.median(dt_est) if dt_est else 0.1
                    fps = max(2.0, min(30.0, (1.0 / med) / args.stride))
                writer = cv2.VideoWriter(str(out / "replay.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (HUD.W, HUD.H))
                print(f"[info] writing {out/'replay.mp4'} at {fps:.1f} fps")
            writer.write(canvas)
            rendered += 1
            # stills at the first occurrence of each major event kind
            for e in rt.events[-6:]:
                if e.kind in MAJOR_BANNER_KINDS and e.kind not in still_kinds_seen \
                        and abs(e.t - ti) < 1.0:
                    still_kinds_seen.add(e.kind)
                    cv2.imwrite(str(out / "stills" /
                                    f"{rendered:05d}_{e.kind}.png"), canvas)
            if rendered % 250 == 0:
                el = time.time() - t_wall
                print(f"  frame {frame_idx}/{n_imgs}  t+{t_rel:.0f}s  "
                      f"rendered {rendered}  ({el:.0f}s wall, "
                      f"{rendered / max(el, 1e-6):.1f} fps)")
    finally:
        reader.close()
        if writer is not None:
            writer.release()

    # flush odom past the last image
    while k < len(t):
        if gate:
            gate.step(frame=frame_idx, t_rel=float(t[k] - t0))
        rt.step(float(t[k]), float(x[k]), float(y[k]), float(yaw[k]))
        k += 1

    print(f"[info] rendered {rendered} frames")
    write_outputs(out, rt, gate, args)


if __name__ == "__main__":
    main()