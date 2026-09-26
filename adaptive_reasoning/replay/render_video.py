"""
render_video.py — watch exactly what the gate saw and did, frame by frame.

Overlays, per frame:
  * every tracked plate's box — teal if relevant (R > 0.5), grey otherwise —
    with "R=.. ℓ=.." above it and the track's read text below;
  * a HUD strip: the q_t bar with the τ notch, the gate state pill
    (NO_SIGN grey / ARMED yellow / PENDING orange / DECIDED green — same
    colors as the paper figures), the standing decision, call counters for
    ours vs. a reactive necessity-only gate, and event banners at the
    annotated flip and junction frames;
  * a fire flash when a call is sent, and "VLM (SIM)" tag on the simulated
    response — this replay does NOT call a real VLM; the response is the
    oracle stand-in (fires on the plate, returns --decision after --latency
    seconds), clearly labeled until E2 wires cached real answers.

The gate that runs here is the real EvidenceGate over the jsonl's per-frame
evidence — the same numbers the npz/figures use — so the video, the timeline
figure, and the sweep can never silently disagree.

Usage (after dump_features has produced the jsonl for the bag):
  python -m adaptive_reasoning.replay.render_video \
      --bag ~/SignWay/ros2_bags/rosbag2-keller-t1 \
      --jsonl $SCRATCH/ar_replay/rosbag2-keller-t1.jsonl \
      --tau 0.55 --decision turn_right --flip 159 --junction 293 \
      --out figs/t1_replay.mp4
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..gate import EvidenceGate, GateConfig, PlateEvidence, VLMResponse, new_goal
from ..memory import Memory

# gate-state colors, BGR (cv2), matching figstyle
STATE_BGR = {"NO_SIGN": (219, 213, 209), "ARMED": (21, 204, 250),
             "PENDING": (60, 146, 251), "DECIDED": (128, 222, 74)}
TEAL, GREY, WHITE, RED = (110, 118, 15), (128, 128, 128), (255, 255, 255), (60, 60, 220)


def load_jsonl(path: str) -> dict[int, list[dict]]:
    per_frame: dict[int, list[dict]] = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        per_frame[r["frame"]].append(r)
    return per_frame


def draw_plates(frame_bgr, recs: list[dict], sc: float = 1.0) -> None:
    """The replay's plate overlay, exactly: box coloured per track when R>0.5
    (grey otherwise), 'R=.. l=..' above, OCR text below.  Shared by the video
    renderer and by run_images so standalone frames look like replay frames."""
    import cv2
    for r in recs:
        if not r.get("box"):
            continue
        x0, y0, x1, y1 = [int(v * sc) for v in r["box"]]
        col = track_color(r["plate_id"]) if r["R"] > 0.5 else GREY
        th = 3 if r["R"] > 0.5 else 1
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), col, th)
        cv2.putText(frame_bgr, f"R={r['R']:.2f} l={r['ell']:.2f}", (x0, max(y0 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(frame_bgr, r["text"][:40], (x0, min(y1 + 18, frame_bgr.shape[0] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


def track_color(plate_id: str) -> tuple[int, int, int]:
    palette = [(110, 118, 15), (237, 58, 124), (39, 119, 219), (13, 148, 136),
               (196, 90, 18), (140, 46, 167)]
    return palette[hash(plate_id) % len(palette)]


def main() -> None:
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--topic", default="/c1/image_raw")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--decision", default="turn_right", help="oracle answer for the SIM VLM")
    ap.add_argument("--vlm", choices=["sim", "real"], default="sim",
                    help="sim = oracle stand-in; real = call the actual VLM at fire time (cached)")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--reasoning", default="low", help="reasoning_effort for Gemini; 'none' for local vLLM models")
    ap.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta/openai/",
                    help="http://localhost:8000/v1 for local vLLM")
    ap.add_argument("--rpm", type=float, default=15.0, help="0 = no client-side pacing (local server)")
    ap.add_argument("--with-scene", action="store_true",
                    help="ALSO send the full camera frame (default: crops only; "
                         "pilot A/B showed scene adds 1-2s and misled t3)")
    ap.add_argument("--cache-dir", default="vlm_cache")
    ap.add_argument("--latency", type=float, default=1.2, help="simulated VLM latency, seconds")
    ap.add_argument("--flip", type=int, default=None, help="annotated flip_frame (raw idx)")
    ap.add_argument("--junction", type=int, default=None, help="annotated junction_frame (raw idx)")
    ap.add_argument("--goal", default="", help="shown in the HUD")
    ap.add_argument("--fps-out", type=float, default=10.0)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    per_frame = load_jsonl(a.jsonl)
    frames_wanted = sorted(per_frame.keys())
    stride = min(np.diff(frames_wanted)) if len(frames_wanted) > 1 else 1
    fps_eff = a.fps_out
    latency_frames = max(int(a.latency * fps_eff), 1)

    reasoner = None
    if a.vlm == "real":
        from ..reasoner import FakeVLM, OpenAICompatVLM, Reasoner
        client = (FakeVLM({a.goal: {"applicable": True, "direction": a.decision,
                                    "confidence": 0.9, "summary": "fake transport"}})
                  if a.model == "fake" else OpenAICompatVLM(
                      model=a.model, base_url=a.base_url, rpm=a.rpm,
                      reasoning_effort=None if (a.reasoning or "").lower() in ("", "none", "off")
                      else a.reasoning))
        base = Path(a.out).parent / Path(a.bag).name / (a.model + ("_scene" if a.with_scene else ""))
        run_n = 1 + max((int(d.name[3:]) for d in base.glob("run*") if d.name[3:].isdigit()),
                        default=0)
        run_dir = base / f"run{run_n}"
        reasoner = Reasoner(client, model_tag=f"{a.model}:{a.reasoning}",
                            cache_dir=Path(a.cache_dir), payload_dir=run_dir)
        print(f"payloads -> {run_dir}/callNN/ (prompt.txt, img*.jpg, answer.json)")

    gate = EvidenceGate(GateConfig(tau=a.tau, fast_path=False))
    mem, state = Memory(a.goal or "replay"), new_goal()
    pending = None
    last_answer = None
    fires, reactive_seen, reactive_fires = [], set(), 0
    flash_until = -1

    from .dump_features import RosbagSource
    src = RosbagSource(a.bag, a.topic, stride=int(stride))

    writer = None
    for raw_idx, img in src:
        recs = per_frame.get(raw_idx, [])

        # ---- the real gate, over the same numbers the figures use ----------
        if pending and raw_idx >= pending[0]:
            state, upds = gate.on_response(state, pending[1], raw_idx)
            for u in upds:
                mem.apply(u)
            pending = None
        evid = [PlateEvidence(r["plate_id"], float(r["R"]), float(r["ell"])) for r in recs]
        res = gate.tick(state, evid, mem, raw_idx)
        for u in res.memory_updates:
            mem.apply(u)
        state = res.state
        if res.fire:
            flash_until = raw_idx + 4 * stride
            if reasoner is not None:
                rec = next((r for r in recs if r["plate_id"] == res.best.plate_id), None)
                if rec and rec.get("box"):
                    # SHARED payload construction: grow the fired box to swallow
                    # nearby fragments (a far sign splits into pieces; firing on
                    # one piece used to send a crop of the bare word "B50"),
                    # apply arrow-safe margins, enforce a minimum crop size.
                    from ..evidence.detect import payload_crop
                    others = [r["box"] for r in recs if r.get("box")]
                    crop = payload_crop(img, rec["box"], others)
                else:
                    crop = img
                if a.with_scene:
                    import cv2 as _cv2
                    scene = _cv2.resize(img, (1024, int(1024 * img.shape[0] / img.shape[1])))
                else:
                    scene = None
                ans = reasoner.ask(a.goal or "goal", [rec["text"] if rec else ""],
                                   [crop], scene=scene,
                                   memory_summary=f"{len(mem)} signs consumed")
                last_answer = ans
                lat_s = a.latency if ans.cached else max(ans.latency_s, 0.1)
                fires.append((raw_idx, ans.direction, ans.applicable,
                              round(ans.latency_s, 2), ans.cached))
                pending = (raw_idx + max(int(lat_s * fps_eff), 1) * stride,
                           VLMResponse(res.best.plate_id, ans.applicable, ans.direction))
            else:
                applicable = any(r["plate_id"] == res.best.plate_id and r["R_struct"] >= 1.0
                                 for r in recs)
                fires.append(raw_idx)
                pending = (raw_idx + latency_frames * stride,
                           VLMResponse(res.best.plate_id, applicable,
                                       a.decision if applicable else None))
        new_tracks = {r["plate_id"] for r in recs} - reactive_seen
        reactive_fires += bool(new_tracks)
        reactive_seen |= new_tracks

        # ---- draw ----------------------------------------------------------
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if a.scale != 1.0:
            frame = cv2.resize(frame, None, fx=a.scale, fy=a.scale)
        sc = a.scale
        draw_plates(frame, recs, sc)

        # HUD strip
        H, W = frame.shape[:2]
        hud = np.zeros((110, W, 3), np.uint8)
        # q bar with tau notch
        q = res.q
        cv2.rectangle(hud, (10, 15), (10 + int(0.5 * W), 35), (70, 70, 70), -1)
        cv2.rectangle(hud, (10, 15), (10 + int(0.5 * W * min(q, 1.0)), 35), TEAL, -1)
        tx = 10 + int(0.5 * W * a.tau)
        cv2.line(hud, (tx, 10), (tx, 40), WHITE, 2)
        cv2.putText(hud, f"q={q:.2f}  tau={a.tau:.2f}", (10 + int(0.5 * W) + 10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        # state pill + decision
        st = state.state.value
        cv2.rectangle(hud, (10, 55), (170, 90), STATE_BGR[st], -1)
        cv2.putText(hud, st, (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        dec = state.decision or "-"
        cv2.putText(hud, f"prompt: {dec}", (185, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        cv2.putText(hud, f"calls ours:{len(fires)}  reactive:{reactive_fires}",
                    (int(W * 0.55), 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        vlm_tag = f"REAL {a.model}" if reasoner is not None else f"SIM({a.decision})"
        cv2.putText(hud, f"f{raw_idx}  goal:{a.goal}  VLM: {vlm_tag}",
                    (int(W * 0.55), 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        if last_answer is not None and state.state.value == "DECIDED":
            cv2.putText(hud, f'"{last_answer.summary[:70]}"', (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1)
        frame = np.vstack([frame, hud])

        # event banners + fire flash
        banner = None
        if a.flip is not None and abs(raw_idx - a.flip) < stride:
            banner = "ANNOTATED FLIP (human can read sign)"
        if a.junction is not None and abs(raw_idx - a.junction) < stride:
            banner = "ANNOTATED JUNCTION (decision must be executed)"
        if banner:
            cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        if raw_idx <= flash_until:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), RED, 8)
            cv2.putText(frame, "FIRE -> VLM", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)

        if writer is None:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     a.fps_out, (frame.shape[1], frame.shape[0]))
        writer.write(frame)

    if pending:      # response landed after the bag ended: apply for the final report
        state, upds = gate.on_response(state, pending[1], pending[0])
        for u in upds:
            mem.apply(u)
    if writer:
        writer.release()
    print(f"wrote {a.out}  (ours fired {len(fires)}x: {fires}; "
          f"reactive {reactive_fires}x; final state {state.state.value}, prompt {state.decision})")
    if reasoner is not None and last_answer is not None:
        print(f'last VLM answer: applicable={last_answer.applicable} '
              f'direction={last_answer.direction} conf={last_answer.confidence:.2f} '
              f'latency={last_answer.latency_s:.2f}s cached={last_answer.cached}\n'
              f'summary: {last_answer.summary}')


if __name__ == "__main__":
    main()
