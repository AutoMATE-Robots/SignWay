"""
render_submission_memory.py — clip C: the square loop, junction memory, ICRA video grade.

The loop is render_video_memory.py's, unchanged where it matters: the real
EvidenceGate over the dump_features jsonl, the real Reasoner (cached, full-plate
directory prompt), the odometry-driven MemoryRuntime, memory answering INSTEAD of
the VLM at a junction it already knows, and the standing prompt published only
when an answer lands. Only the presentation changed:

  * same canvas/panel/timeline as clips A and B (render_submission.compose)
  * new mode  RECALL  (emerald): "known junction — decision recalled", no VLM call
  * a clean mini-map in the panel: route trace + junction nodes only
    (no beam, no edge polylines, no rivals — those are internals)
  * timeline: orange = VLM call, emerald diamond = memory recall
  * no captions; author them against the final timeline

Place at: adaptive_reasoning/replay/render_submission_memory.py
Run from the repo root in the `jmem` venv (needs pillow: pip install pillow).

  python -m adaptive_reasoning.replay.render_submission_memory \
      --bag ~/SignWay/ros2_bags/rosbag2-square --topic /image_raw \
      --jsonl $SCRATCH/ar_replay/rosbag2-square.jsonl \
      --odom-csv junction_memory/odom_rosbag2-square_cal.csv \
      --goals square_goals.yaml --tau 0.55 \
      --vlm real --model gemini-3.1-flash-lite \
      --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
      --latency 4.0 --fixed-latency --h264 --out $SCRATCH/video/clipC_square.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ..gate import EvidenceGate, GateConfig, PlateEvidence, VLMResponse, new_goal
from ..memory import Memory
from .render_submission import Fonts, compose, set_layout, W, H

from junction_memory.memory_runtime import MemoryRuntime, directory_from_vlm, norm_goal
from junction_memory.plate_prompt import extract_directory, install, PLATE_PROMPT
try:
    from junction_memory.goal_matcher import match as _goal_match
except ImportError:
    _goal_match = None
from junction_memory.replay_ar_memory import (open_bag, pick_image_connection,
                                              decode_image)

AR_FROM_MEM = {"left": "turn_left", "right": "turn_right", "straight": "straight", "back": "stop"}
MEM_FROM_AR = {"turn_left": "left", "turn_right": "right", "straight": "straight"}
NODE_RE = re.compile(r"\bn_\d+\b")


_DIR_WORDS = [("left", "left"), ("\u2190", "left"), ("right", "right"), ("\u2192", "right"),
              ("straight", "straight"), ("up", "straight"), ("\u2191", "straight"),
              ("back", "back"), ("down", "back"), ("\u2193", "back")]


def prose_directory(raw: str) -> list[dict]:
    """Harvest '"6-115 to 6-189" has a straight arrow' style lines into directory entries.
    Used when the model reads the plate in prose but returns no structured directory."""
    out = []
    for line in (raw or "").splitlines():
        labels = re.findall(r'"([^"]{2,60})"', line)
        if not labels:
            continue
        low = line.lower()
        d = next((v for k, v in _DIR_WORDS if k in low), None)
        if d is None:
            continue
        for lab in labels:
            if re.search(r"\d", lab) or len(lab.split()) <= 4:
                out.append({"label": lab, "direction": d})
    return out


def load_jsonl(path):
    per_frame = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        per_frame[r["frame"]].append(r)
    return per_frame


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--odom-csv", required=True, help="CALIBRATED odom (calibrate_yaw.py output)")
    ap.add_argument("--goals", required=True, help="mission goal schedule YAML")
    ap.add_argument("--topic", default="/image_raw")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--vlm", choices=["sim", "real"], default="real")
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta/openai/")
    ap.add_argument("--rpm", type=float, default=15.0)
    ap.add_argument("--decision", default="turn_right", help="SIM-only oracle answer")
    ap.add_argument("--sim-directory", default=None, help="SIM-only, e.g. '6-220=turn_left,6-117=straight'")
    ap.add_argument("--cache-dir", default="vlm_cache")
    ap.add_argument("--latency", type=float, default=4.0)
    ap.add_argument("--fixed-latency", action="store_true", help="always display --latency for a call")
    ap.add_argument("--verify-tau", type=float, default=None)
    ap.add_argument("--min-plate-h", type=float, default=0.0,
                    help="hold the VLM call while the plate is shorter than this (px); 0 = off")
    ap.add_argument("--hold-fire", type=float, default=1.0)
    ap.add_argument("--hold-answer", type=float, default=1.5)
    ap.add_argument("--hide-irrelevant", action="store_true", default=True)
    ap.add_argument("--fps-out", type=float, default=30.0)
    ap.add_argument("--start-s", type=float, default=None)
    ap.add_argument("--end-s", type=float, default=None)
    ap.add_argument("--font", default=None)
    ap.add_argument("--timeline", action="store_true", help="draw the bottom strip (off by default for this clip)")
    ap.add_argument("--h264", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    F = Fonts(a.font)
    set_layout(timeline=a.timeline)
    outdir = Path(a.out).parent / (Path(a.out).stem + "_memory")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- odometry (calibrated) ----------------------------------------------
    od = np.loadtxt(a.odom_csv, delimiter=",", skiprows=1)
    ot, ox, oy, oyaw = od[:, 0], od[:, 1], od[:, 2], od[:, 3]
    m = 1.0
    bounds = (float(ox.min()) - m, float(ox.max()) + m, float(oy.min()) - m, float(oy.max()) + m)
    print(f"[info] odom {len(ot)} samples, {ot[-1] - ot[0]:.0f} s span")

    schedule = sorted((yaml.safe_load(open(a.goals)) or {}).get("goals", []), key=lambda g: g["from_s"])
    if not schedule:
        raise SystemExit("--goals YAML has no goals: list")

    per_frame = load_jsonl(a.jsonl)
    frames = sorted(per_frame)
    n = len(frames)
    pos_of = {f: i for i, f in enumerate(frames)}
    stride = int(min(np.diff(frames))) if n > 1 else 1
    bag_fps_eff = 10.0
    repeat = max(int(round(a.fps_out / bag_fps_eff)), 1)
    print(f"[info] evidence on {n} frames, stride {stride}")

    # ---- gate + reasoner (as in render_video_memory) ------------------------
    gate = EvidenceGate(GateConfig(tau=a.tau, fast_path=False))
    ar_mem, state = Memory(schedule[0]["goal"]), new_goal()
    sim_dir = {}
    for kv in (a.sim_directory or "").split(","):
        if "=" in kv:
            k_, v_ = kv.split("=", 1)
            sim_dir[k_.strip()] = v_.strip()
    reasoner = None
    if a.vlm == "real":
        from ..reasoner import FakeVLM, OpenAICompatVLM, Reasoner
        if a.model == "fake":
            client = FakeVLM({"": {"applicable": True, "direction": a.decision, "confidence": 0.9,
                                   "summary": "fake", "directory": [{"label": k_, "direction": v_}
                                                                     for k_, v_ in sim_dir.items()]}})
        else:
            client = OpenAICompatVLM(model=a.model, base_url=a.base_url, rpm=a.rpm,
                                     reasoning_effort=None if a.reasoning.lower() in ("", "none", "off")
                                     else a.reasoning)
        tag = f"{a.model}:{a.reasoning}:dirv2"   # dirv1 answers were made with the OLD prompt: new key
        from .. import reasoner as _reasoner_mod
        install(_reasoner_mod)              # module-level (harmless), and...
        reasoner = Reasoner(client, model_tag=tag, cache_dir=Path(a.cache_dir),
                            payload_dir=outdir / "payloads")
        reasoner.prompt_template = PLATE_PROMPT   # ...the instance field the reasoner actually reads
        print("[info] full-plate directory prompt bound to the Reasoner instance")

    rt = MemoryRuntime(bag_name=Path(a.bag).name, verify_tau=a.verify_tau, goal_matcher=_goal_match)

    # ---- presentation state --------------------------------------------------
    vlm_calls = memory_answers = memory_standing = 0
    fires, lands, recalls, bands = [], [], [], []
    node_xy: dict[str, tuple] = {}        # node id -> anchor pose (robot pose at creation)
    known: set[str] = set()               # nodes that answered a recall
    n_events_seen = 0
    standing, prompt_source = "straight", "default"
    pending = None                        # (land_idx, VLMResponse, s_at_fire, lat_s, start_idx)
    sent_crop = last_answer = None
    flash_state = None                    # ("fire"|"land"|"recall", decision)
    writer = None
    tmp_out = a.out if not a.h264 else str(Path(a.out).with_suffix(".raw.mp4"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def emit(frame_rgb, k):
        nonlocal writer
        if writer is None:
            writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps_out, (W, H))
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for _ in range(k):
            writer.write(bgr)

    def harvest_events():
        """Turn runtime events into map state; internals (beam/edges) are ignored."""
        nonlocal n_events_seen
        for e in rt.events[n_events_seen:]:
            kind = getattr(e, "kind", "")
            msg = getattr(e, "msg", None) or getattr(e, "message", None) or str(e)
            ids = NODE_RE.findall(str(msg))
            if kind in ("NODE_CREATED", "NODE_FROM_SIGN") and ids and kk > 0:
                node_xy.setdefault(ids[0], (float(ox[kk - 1]), float(oy[kk - 1])))
            if kind in ("MEMORY_RECALL", "VLM_CALL_AVOIDED", "MEMORY_STANDING") and ids:
                known.add(ids[0])
        n_events_seen = len(rt.events)

    reader = open_bag(a.bag)
    conns = pick_image_connection(reader, a.topic)
    kk = 0
    t0_img = None
    try:
        for i, (conn, tns, raw) in enumerate(reader.messages(connections=conns)):
            ti = tns * 1e-9
            if t0_img is None:
                t0_img = ti
            t_rel = ti - t0_img
            while kk < len(ot) and ot[kk] <= ti:
                rt.step(float(ot[kk]), float(ox[kk]), float(oy[kk]), float(oyaw[kk]))
                kk += 1
            goal = schedule[0]["goal"]
            for entry in schedule:
                if t_rel >= entry["from_s"]:
                    goal = entry["goal"]
            if getattr(rt, "current_goal", None) is not None and norm_goal(rt.current_goal) != norm_goal(goal):
                rt._emit("GOAL_SWITCH", f"goal switched: {rt.current_goal} -> {goal}", "major", goal=goal)
                if rt.standing and norm_goal(rt.standing["goal"]) != norm_goal(goal):
                    rt.standing = None
                ar_mem, state = Memory(goal), new_goal()
                standing, prompt_source = "straight", "default"
            rt.current_goal = goal
            if i % stride:
                continue
            raw_idx = i
            pos = pos_of.get(raw_idx, len(bands))
            recs = per_frame.get(raw_idx, [])
            rt.observe_plates([r["plate_id"] for r in recs])
            img_bgr = decode_image(reader.deserialize(raw, conn.msgtype), conn.msgtype)
            img = img_bgr[:, :, ::-1].copy()          # RGB for the presentation layer

            # ---- answer landing --------------------------------------------
            landed_now = False
            if pending and raw_idx >= pending[0]:
                state, upds = gate.on_response(state, pending[1], raw_idx)
                for u in upds:
                    ar_mem.apply(u)
                if state.decision:
                    standing, prompt_source = state.decision, "vlm"
                    travelled = rt.s[-1] - pending[2] if rt.s else 0.0
                    rt._emit("VLM_ANSWER_LANDED", f"VLM answer usable after {pending[3]:.1f} s "
                             f"({travelled:.1f} m travelled): {state.decision}", "major")
                    landed_now = True
                    lands.append(pos)
                pending = None
            evid = [PlateEvidence(r["plate_id"], float(r["R"]), float(r["ell"])) for r in recs]
            res = gate.tick(state, evid, ar_mem, raw_idx)
            for u in res.memory_updates:
                ar_mem.apply(u)

            # ---- memory publishes on its own schedule ----------------------
            recalled_now = False
            hit_now = rt.resolves(goal)
            if hit_now is not None and prompt_source != "memory":
                standing, prompt_source = AR_FROM_MEM.get(hit_now["action"], "straight"), "memory"
                memory_standing += 1
                recalled_now = True
                recalls.append(pos)
                known.add(hit_now["node"])
                rt._emit("MEMORY_STANDING", f"standing prompt from memory: '{goal}' -> {standing} at "
                         f"{hit_now['node']} ({hit_now['lead_m']:.1f} m out)", "major",
                         node=hit_now["node"], goal=goal, action=standing, lead_m=hit_now["lead_m"])

            fired_now = False
            vetoed = False
            if res.fire:
                hit = rt.resolves(goal)
                rec = next((r for r in recs if r["plate_id"] == res.best.plate_id), None)
                plate_h = float(rec["box"][3] - rec["box"][1]) if rec and rec.get("box") else 0.0
                if hit is not None:
                    # memory answers instead of the VLM: zero latency, no call
                    memory_answers += 1
                    direction = AR_FROM_MEM.get(hit["action"], "straight")
                    rt._emit("VLM_CALL_AVOIDED", f"node {hit['node']} known: memory answers '{goal}' -> "
                             f"{direction}; no VLM call", "major", node=hit["node"], goal=goal)
                    state, upds = gate.on_response(res.state, VLMResponse(res.best.plate_id, True, direction),
                                                   raw_idx)
                    for u in upds:
                        ar_mem.apply(u)
                    standing, prompt_source = direction, "memory"
                    known.add(hit["node"])
                    if not recalled_now:
                        recalls.append(pos)
                        recalled_now = True
                elif plate_h < a.min_plate_h:
                    vetoed = True                          # keep acquiring; retry next frame
                else:
                    state = res.state
                    fired_now = True
                    fires.append(pos)
                    vlm_calls += 1
                    from ..evidence.detect import payload_crop
                    crop = (payload_crop(img, rec["box"], [r["box"] for r in recs if r.get("box")])
                            if rec and rec.get("box") else img)
                    sent_crop = Image.fromarray(crop)
                    if reasoner is not None:
                        ans = reasoner.ask(goal, [rec["text"] if rec else ""], [crop], scene=None,
                                           memory_summary=f"{len(ar_mem)} signs consumed")
                        last_answer = ans
                        applicable, direction = ans.applicable, ans.direction
                        plate_dir = directory_from_vlm(getattr(ans, "directory", None)
                                                       or extract_directory(ans.raw)
                                                       or prose_directory(ans.raw))
                        lat = a.latency if (ans.cached or a.fixed_latency) else max(ans.latency_s, 0.1)
                    else:
                        applicable = any(r["plate_id"] == res.best.plate_id and r["R_struct"] >= 1.0
                                         for r in recs)
                        direction = a.decision if applicable else None
                        plate_dir = directory_from_vlm([{"label": k_, "direction": v_}
                                                        for k_, v_ in sim_dir.items()]) if applicable else {}
                        lat = a.latency
                    pending = (raw_idx + max(int(lat * bag_fps_eff), 1) * stride,
                               VLMResponse(res.best.plate_id, applicable, direction),
                               rt.s[-1] if rt.s else 0.0, float(lat), raw_idx)
                    if applicable and direction:
                        if not plate_dir:
                            plate_dir = {norm_goal(goal): MEM_FROM_AR.get(direction, "straight")}
                            rt._emit("DIRECTORY_PARTIAL", "VLM returned no plate directory", "warn")
                        rt.provide_directory(plate_dir, goal=goal, decision=MEM_FROM_AR.get(direction, "straight"),
                                             conf=getattr(last_answer, "confidence", None),
                                             frame=raw_idx, plate_id=res.best.plate_id)
            if not res.fire:
                state = res.state          # vetoed: keep pre-tick state; fired/recalled: already set
            rt.set_frame(raw_idx)
            harvest_events()

            # ---- display mode ----------------------------------------------
            gs = state.state.value
            if prompt_source == "memory" and gs != "PENDING":
                disp = "RECALL"
            elif vetoed:
                disp = "ARMED"
            else:
                disp = gs
            bands.append(disp)

            if a.start_s is not None and t_rel < a.start_s:
                continue
            if a.end_s is not None and t_rel > a.end_s:
                break

            best_rec = next((r for r in recs if res.best and r["plate_id"] == res.best.plate_id), None)
            best_crop = None
            if best_rec and best_rec.get("box") and disp == "ARMED":
                x0, y0, x1, y1 = [int(v) for v in best_rec["box"]]
                pad = int(0.25 * (x1 - x0))
                best_crop = Image.fromarray(img[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad])
            in_flight = None
            if pending is not None:
                in_flight = (raw_idx - pending[4]) / max(pending[0] - pending[4], 1)
            step = max(len(ox) // 4000, 1)
            mm = {"bounds": bounds,
                  "trace": list(zip(ox[:kk:step], oy[:kk:step])),
                  "nodes": [(x, y, nid in known) for nid, (x, y) in node_xy.items()],
                  "pose": (float(ox[kk - 1]), float(oy[kk - 1])) if kk > 0 else None}
            decision_disp = standing if prompt_source in ("memory", "vlm") else state.decision
            st = dict(state=disp, best_id=res.best.plate_id if res.best else None, best_rec=best_rec,
                      best_crop=best_crop, sent_crop=sent_crop, tau=a.tau, in_flight_frac=in_flight,
                      decision=decision_disp, rationale=None, n_calls=vlm_calls, goal=goal,
                      pos=pos, n=n, bands=bands, fires=fires, lands=lands, recalls=recalls,
                      junction_pos=None, hide_irrelevant=a.hide_irrelevant, minimap=mm,
                      no_timeline=not a.timeline,
                      source=("memory" if prompt_source == "memory" else "vlm"))
            emit(compose(img, recs, st, F), repeat)
            if fired_now:
                emit(compose(img, recs, dict(st, callout="sign readable \u2192 reason",
                                              callout_color=(251, 146, 60)), F),
                     int(a.hold_fire * a.fps_out))
            if landed_now and state.decision:
                emit(compose(img, recs, dict(st, callout=f"decision:  {state.decision.replace('_', ' ')}",
                                              callout_color=(74, 222, 128)), F),
                     int(a.hold_answer * a.fps_out))
            if recalled_now:
                emit(compose(img, recs, dict(st, callout=f"known junction \u2192 recalled:  "
                                                         f"{standing.replace('_', ' ')}",
                                              callout_color=(52, 211, 153)), F),
                     int(a.hold_answer * a.fps_out))
    finally:
        reader.close()
        if writer is not None:
            writer.release()

    # event times for authoring captions / setting the goal-switch time
    with open(outdir / "events.txt", "w") as fh:
        for e in rt.events:                      # every event: this is the debug file
            if True:
                fh.write(f"{getattr(e, 't', 0) - ot[0]:7.1f}s  {e.kind:18s} "
                         f"{getattr(e, 'msg', None) or getattr(e, 'message', '')}\n")
    print(f"wrote {a.out}  vlm_calls={vlm_calls} memory_answers={memory_answers} "
          f"memory_standing={memory_standing} nodes={len(node_xy)} known={len(known)}")
    print(f"event timeline -> {outdir / 'events.txt'}  (use it to set the goal switch and captions)")
    if last_answer is not None:
        print(f"last VLM answer: {last_answer.direction} "
              f"directory={getattr(last_answer, 'directory', None) or prose_directory(last_answer.raw)}")

    if a.h264 and writer is not None:
        import shutil
        if shutil.which("ffmpeg") is None:
            print(f"ffmpeg not on PATH: raw video kept at {tmp_out} (re-encode later in the ar env)")
            return
        encs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        vcodec = (["-c:v", "libx264", "-crf", "18"] if "libx264" in encs else ["-c:v", "libopenh264", "-b:v", "12M"])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_out, *vcodec, "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", a.out], check=True)
        Path(tmp_out).unlink()


if __name__ == "__main__":
    main()