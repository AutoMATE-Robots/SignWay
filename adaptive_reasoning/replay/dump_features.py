"""
dump_features.py — E1.  Run the evidence pipeline over recorded frames once;
every later experiment (calibration, τ sweep, baselines, timeline figure)
reads the resulting arrays and never touches images again.

Per processed frame, per plate track:  R_struct, R_sem, R, φ (6 features),
ℓ, resolvable, direction, text.  Output per bag:

  <out>/<bag>.npz     dense arrays on the processed-frame grid:
                      frame_idx (raw indices), plate{i}_id/label/R/ell/phi/
                      resolvable, junction_frame/flip_frame (grid indices),
                      fps (effective), vlm_latency_frames (placeholder 0)
  <out>/<bag>.jsonl   one record per (frame, track) with full detail

Frame sources are pluggable so the pipeline is testable without ROS:
  * RosbagSource   — rosbags AnyReader on /c1/image_raw (raw or compressed)
  * ImageDirSource — a directory of frames named <idx>.png (phone captures)
Both yield (raw_frame_idx, RGB ndarray).  OCR runs at NATIVE resolution.

MSI usage (conda env `ar`; needs: pip install rosbags python-doctr[torch]):
  python -m adaptive_reasoning.replay.dump_features \
      --csv annotations/ar_extra.csv --bags-root ~/SignWay/ros2_bags \
      --out $SCRATCH/ar_replay --stride auto
`--stride auto` = fps//10 (the dataset's 10 Hz convention).
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol

import numpy as np

from ..evidence.buffer import BufferConfig, EvidenceBuffer
from ..evidence.detect import DetectConfig, OcrEngine, detect, payload_crop
from ..evidence.features import phi, phi_vector
from ..evidence.legibility import Legibility
from ..evidence.relevance import Goal, relevance_lines
from ..evidence.resolve import Plate, PlateLine, resolve


# ----------------------------------------------------------------------------
# Frame sources
# ----------------------------------------------------------------------------

class FrameSource(Protocol):
    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]: ...


class RosbagSource:
    """(raw_frame_idx, RGB frame) from a ros2 bag at native resolution."""

    def __init__(self, bag_path: str, topic: str = "/c1/image_raw", stride: int = 1):
        self.bag_path, self.topic, self.stride = Path(bag_path), topic, stride

    def __iter__(self):
        from rosbags.highlevel import AnyReader  # lazy; pure-python, safe in `ar` env
        from rosbags.typesys import Stores, get_typestore
        with AnyReader([self.bag_path],
                       default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
            conns = [c for c in reader.connections if c.topic == self.topic]
            if not conns:
                raise RuntimeError(f"topic {self.topic} not in {self.bag_path}")
            i = 0
            for conn, _, raw in reader.messages(connections=conns):
                if i % self.stride == 0:
                    msg = reader.deserialize(raw, conn.msgtype)
                    yield i, _to_rgb(msg)
                i += 1


def _to_rgb(msg) -> np.ndarray:
    """sensor_msgs Image (rgb8/bgr8/mono8) or CompressedImage → RGB ndarray."""
    if hasattr(msg, "format"):                       # CompressedImage
        from PIL import Image
        import io
        return np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, msg.step // 3 if msg.step else w, 3)[:, :w]
        return img[..., ::-1].copy() if enc == "bgr8" else img.copy()
    if enc == "mono8":
        g = buf.reshape(h, msg.step if msg.step else w)[:, :w]
        return np.stack([g] * 3, axis=-1)
    if enc in ("yuv422", "uyvy", "yuv422_yuy2", "yuyv"):
        row = buf.reshape(h, msg.step if msg.step else w * 2)[:, : w * 2].astype(np.float32)
        if enc in ("yuv422", "uyvy"):          # U Y V Y
            u, y0, v, y1 = row[:, 0::4], row[:, 1::4], row[:, 2::4], row[:, 3::4]
        else:                                   # Y U Y V
            y0, u, y1, v = row[:, 0::4], row[:, 1::4], row[:, 2::4], row[:, 3::4]
        Y = np.empty((h, w), np.float32); Y[:, 0::2] = y0; Y[:, 1::2] = y1
        U = np.repeat(u, 2, axis=1)[:, :w] - 128.0
        V = np.repeat(v, 2, axis=1)[:, :w] - 128.0
        rgb = np.stack([Y + 1.402 * V,
                        Y - 0.344136 * U - 0.714136 * V,
                        Y + 1.772 * U], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8)
    raise ValueError(f"unsupported encoding {msg.encoding}")


class ImageDirSource:
    """Frames named <idx>.png/.jpg in a directory (phone/handheld captures)."""

    def __init__(self, folder: str, stride: int = 1):
        self.folder, self.stride = Path(folder), stride

    def __iter__(self):
        from PIL import Image
        files = sorted(self.folder.glob("*.[pj][np]g"),
                       key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
        for f in files:
            idx = int("".join(c for c in f.stem if c.isdigit()) or 0)
            if idx % self.stride == 0:
                yield idx, np.asarray(Image.open(f).convert("RGB"))


# ----------------------------------------------------------------------------
# Scene descriptor (for the IROS-style Key-Frame-Compare baseline)
# ----------------------------------------------------------------------------

class SceneEncoder:
    """Per-frame 4x4 patch-grid descriptor.  Primary: SigLIP patch embeddings
    pooled to a 4x4 grid (what the IROS paper's KFC compares, patch-level so
    corners/junctions differ from straight hallways).  Fallback when SigLIP is
    unavailable: a 4x4 grid of colour means + edge energy — flagged in the npz
    so the baseline's caption can say which was used.  Stored NOW because raw
    bags are deleted after the dump."""

    def __init__(self):
        self.kind = "grid_proxy"
        self._model = None
        try:
            import torch
            from transformers import AutoModel, AutoProcessor  # type: ignore
            name = "google/siglip-base-patch16-224"
            self._proc = AutoProcessor.from_pretrained(name)
            self._model = AutoModel.from_pretrained(name).vision_model.eval()
            self._dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._dev)
            self.kind = "siglip_b16_224_grid4"
            print("SceneEncoder: SigLIP patch grid")
        except Exception as e:  # noqa: BLE001
            print(f"SceneEncoder: SigLIP unavailable ({type(e).__name__}); using grid proxy")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._model is not None:
            import torch
            from PIL import Image
            with torch.no_grad():
                inp = self._proc(images=Image.fromarray(img), return_tensors="pt").to(self._dev)
                tok = self._model(**inp).last_hidden_state[0]          # (196, D)
                g = int(round(tok.shape[0] ** 0.5))                   # 14
                grid = tok.reshape(g, g, -1)
                pooled = grid.reshape(4, g // 4, 4, g // 4, -1).mean(dim=(1, 3))  # (4,4,D)
                return pooled.reshape(16, -1).float().cpu().numpy().astype(np.float16)
        H, W = img.shape[:2]
        f = img.astype(np.float32)
        gray = f @ np.array([0.299, 0.587, 0.114], np.float32)
        edge = np.abs(np.diff(gray, axis=1))
        out = np.zeros((16, 4), np.float32)
        for i in range(4):
            for j in range(4):
                ys, xs = slice(i * H // 4, (i + 1) * H // 4), slice(j * W // 4, (j + 1) * W // 4)
                out[i * 4 + j, :3] = f[ys, xs].reshape(-1, 3).mean(0) / 255.0
                out[i * 4 + j, 3] = edge[ys, xs].mean() / 255.0
        return out.astype(np.float16)


# ----------------------------------------------------------------------------
# The replay
# ----------------------------------------------------------------------------

@dataclass
class TrackLog:
    plate_id: str
    label: str
    R: dict[int, float]
    R_struct: dict[int, float]
    ell: dict[int, float]
    phi: dict[int, np.ndarray]
    resolvable: dict[int, bool]
    direction: dict[int, Optional[str]]


def replay_bag(source: FrameSource, goal: Goal, engine: OcrEngine,
               legibility: Legibility,
               det_cfg: Optional[DetectConfig] = None,
               buf_cfg: Optional[BufferConfig] = None,
               jsonl_path: Optional[Path] = None,
               crops_dir: Optional[Path] = None, crop_every: int = 5,
               scene_encoder: Optional["SceneEncoder"] = None):
    """Run the evidence pipeline; return (frames_processed, {plate_id: TrackLog}).
    With crops_dir set, every track's margin-expanded crop is saved every
    `crop_every`-th sighting — the raw material for E2 labeling, stored NOW so
    the raw bags never need to be touched again."""
    buf = EvidenceBuffer(buf_cfg)
    logs: dict[str, TrackLog] = {}
    frames: list[int] = []
    crop_count: dict[str, int] = {}
    scene: list[np.ndarray] = []
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
    if jsonl_path:
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    jf = open(jsonl_path, "w") if jsonl_path else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # placeholder-ℓ warning is per-run, not per-frame
        import sys, time as _time
        _t0 = _time.monotonic()
        for proc_idx, (raw_idx, img) in enumerate(source):
            if proc_idx % 10 == 0 and proc_idx:
                rate = proc_idx / (_time.monotonic() - _t0)
                print(f"\r    frame {proc_idx} (raw {raw_idx})  {rate:.1f} f/s",
                      end="", flush=True, file=sys.stderr)
            frames.append(raw_idx)
            if scene_encoder is not None:
                scene.append(scene_encoder(img))
            _plates = detect(img, engine, det_cfg)
            _frame_boxes = [pl.box for pl in _plates if pl.box is not None]
            for tr in buf.update(_plates, proc_idx):
                obs = tr.obs
                rel = relevance_lines([l.text for l in obs.lines], goal)
                res = resolve(Plate([PlateLine(l.text, list(l.arrows)) for l in obs.lines]),
                              goal, rel=rel)
                p = phi(obs, tr.agreement_frac)
                ell = legibility(p)
                lg = logs.setdefault(tr.plate_id, TrackLog(tr.plate_id, obs.text[:60],
                                                           {}, {}, {}, {}, {}, {}))
                lg.R[raw_idx] = rel.score
                lg.R_struct[raw_idx] = max((l.struct for l in rel.lines), default=0.0)
                lg.ell[raw_idx] = ell
                lg.phi[raw_idx] = phi_vector(p)
                lg.resolvable[raw_idx] = res.resolved
                lg.direction[raw_idx] = res.vla_prompt
                lg.label = max(lg.label, obs.text[:60], key=len)
                crop_path = None
                if crops_dir is not None and obs.box is not None:
                    n = crop_count.get(tr.plate_id, 0)
                    if n % max(crop_every, 1) == 0:
                        from PIL import Image
                        # SAME payload construction the live gate uses
                        pc = payload_crop(img, obs.box, _frame_boxes, det_cfg)
                        if pc.size:
                            safe = "".join(c if c.isalnum() else "_" for c in tr.plate_id)[:40]
                            cp = crops_dir / f"{safe}_f{raw_idx:05d}.jpg"
                            Image.fromarray(pc.astype("uint8")).save(cp, quality=90)
                            crop_path = cp.name
                    crop_count[tr.plate_id] = n + 1
                if jf:
                    jf.write(json.dumps({
                        "frame": raw_idx, "plate_id": tr.plate_id, "text": obs.text,
                        "box": list(obs.box) if obs.box is not None else None,
                        "R": rel.score, "R_struct": lg.R_struct[raw_idx], "ell": ell,
                        "phi": {k: float(v) for k, v in p.items()},
                        "resolvable": res.resolved, "direction": res.vla_prompt,
                        "arrows": [a for l in obs.lines for a in l.arrows],
                        "crop": crop_path,
                    }) + "\n")
    if jf:
        jf.close()
    if scene_encoder is not None:
        logs["__scene__"] = np.stack(scene) if scene else np.zeros((0, 16, 4), np.float16)  # type: ignore
        logs["__scene_kind__"] = scene_encoder.kind  # type: ignore
    return frames, logs


def save_npz(out_path: Path, frames: list[int], logs: dict[str, TrackLog],
             ann: Optional[dict] = None) -> None:
    """Dense arrays on the processed-frame grid, top tracks first (by peak R·ℓ)."""
    grid = {f: i for i, f in enumerate(frames)}
    T = len(frames)
    scene = logs.pop("__scene__", None)
    scene_kind = logs.pop("__scene_kind__", None)
    order = sorted(logs.values(),
                   key=lambda lg: -max((lg.R[f] * lg.ell[f] for f in lg.R), default=0.0))
    data: dict[str, np.ndarray] = {"frame_idx": np.array(frames)}
    for i, lg in enumerate(order):
        R = np.zeros(T); Rs = np.zeros(T); ell = np.zeros(T)
        resv = np.zeros(T, bool); PHI = np.zeros((T, 6))
        for f in lg.R:
            j = grid[f]
            R[j], Rs[j], ell[j], resv[j] = lg.R[f], lg.R_struct[f], lg.ell[f], lg.resolvable[f]
            PHI[j] = lg.phi[f]
        data.update({f"plate{i}_id": np.array(lg.plate_id), f"plate{i}_label": np.array(lg.label),
                     f"plate{i}_R": R, f"plate{i}_R_struct": Rs, f"plate{i}_ell": ell,
                     f"plate{i}_phi": PHI, f"plate{i}_resolvable": resv})
    if ann:
        for k in ("flip_frame", "junction_frame", "sign_visible_frame"):
            if ann.get(k, "").strip():
                raw = int(ann[k])
                data[k] = np.array(min(range(T), key=lambda j: abs(frames[j] - raw)))
        data["fps"] = np.array(float(ann["fps"]) / max(int(ann.get("_stride", 1)), 1))
        data["decision"] = np.array(ann.get("decision", ""))
        data["goal"] = np.array(ann.get("goal", ""))
    data["vlm_latency_frames"] = np.array(0)   # filled by E3
    if scene is not None:
        data["scene_grid"] = scene                     # (T, 16, D) float16
        data["scene_kind"] = np.array(scene_kind)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data)


def contact_sheet(crops_dir: Path, out_path: Path, cols: int = 6, cell: int = 260,
                  max_crops: int = 48) -> None:
    """One glance = 'are the arrows inside the crops?' for a whole bag."""
    from PIL import Image, ImageDraw
    files = sorted(crops_dir.glob("*.jpg"))
    if not files:
        return
    step = max(len(files) // max_crops, 1)
    files = files[::step][:max_crops]
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        im = Image.open(f)
        im.thumbnail((cell - 8, cell - 8))
        x, y = (i % cols) * cell, (i // cols) * (cell + 16)
        sheet.paste(im, (x + 4, y + 4))
        d.text((x + 4, y + cell - 8), f.stem[-18:], fill=(200, 200, 200))
    sheet.save(out_path, quality=88)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="ar_extra.csv (omit with --harvest)")
    ap.add_argument("--bags-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", default="/c1/image_raw")
    ap.add_argument("--stride", default="auto", help="'auto' = fps//10")
    ap.add_argument("--only", default=None, help="process a single bag name")
    ap.add_argument("--harvest", action="store_true",
                    help="no annotations yet: dump crops/jsonl/φ with a dummy goal. "
                         "R/ℓ in the npz are provisional; they are recomputed from the "
                         "jsonl once real AR annotations exist. Raw bags never needed again.")
    ap.add_argument("--fps", type=int, default=30, help="native fps for --harvest bags")
    ap.add_argument("--no-scene-embed", action="store_true",
                    help="skip per-frame scene descriptors (needed by the IROS-style baseline)")
    ap.add_argument("--crop-every", type=int, default=5,
                    help="save each track's crop every Nth sighting (0 = off)")
    ap.add_argument("--calib", default="adaptive_reasoning/calib/w.json")
    ap.add_argument("--goals-dir", default="adaptive_reasoning/goals")
    a = ap.parse_args()

    from ..evidence.detect import DocTREngine
    engine = DocTREngine()
    scene_enc = None if a.no_scene_embed else SceneEncoder()
    leg = Legibility.load(a.calib)
    if not leg.calibrated:
        print("NOTE: legibility is UNCALIBRATED — ℓ values are placeholder until T6.")

    if a.harvest:
        if not a.only:
            raise SystemExit("--harvest needs --only <bag[,bag2,...]>")
        rows_all = [{"bag": b, "goal": "0-000", "fps": str(a.fps), "decision": "",
                     "flip_frame": "", "junction_frame": "", "sign_visible_frame": ""}
                    for b in a.only.split(",")]
    else:
        if not a.csv:
            raise SystemExit("need --csv (or --harvest)")
        rows_all = [r for r in csv.DictReader(open(a.csv)) if not a.only or r["bag"] == a.only]
    for bag_i, row in enumerate(rows_all, 1):
        print(f"[{bag_i}/{len(rows_all)}]", end=" ")
        stride = int(row["fps"]) // 10 if a.stride == "auto" else int(a.stride)
        row["_stride"] = stride
        gfile = Path(a.goals_dir) / f"{row['goal']}.json"
        goal = Goal.load(gfile) if gfile.exists() else Goal.parse(row["goal"])
        out = Path(a.out) / row["bag"]
        print(f"{row['bag']}: stride {stride}, goal {goal.text} "
              f"({'descriptors' if goal.descriptors else 'no descriptors'})")
        bag_path = Path(a.bags_root) / row["bag"]
        if not bag_path.exists():
            print(f"SKIP {row['bag']}: not found under {a.bags_root} (pull from R2), continuing")
            continue
        src = RosbagSource(str(bag_path), a.topic, stride)
        crops_dir = (Path(a.out) / f"{row['bag']}_crops") if a.crop_every else None
        frames, logs = replay_bag(src, goal, engine, leg,
                                  jsonl_path=out.with_suffix(".jsonl"),
                                  crops_dir=crops_dir, crop_every=a.crop_every,
                                  scene_encoder=scene_enc)
        save_npz(out.with_suffix(".npz"), frames, logs, ann=row)
        n_crops = len(list(crops_dir.glob("*.jpg"))) if crops_dir else 0
        if crops_dir and n_crops:
            contact_sheet(crops_dir, Path(a.out) / f"{row['bag']}_contactsheet.jpg")
        print(file=__import__("sys").stderr)
        print(f"  {len(frames)} frames, {len(logs)} plate tracks, {n_crops} crops → {out}.npz")


if __name__ == "__main__":
    main()
