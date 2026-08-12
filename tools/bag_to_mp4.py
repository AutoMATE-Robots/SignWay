"""Convert ROS2 bags to mp4, one video per image topic, ready for tools/dino_video.py.

    python tools/bag_to_mp4.py --bags ros2_bags --out videos

No ROS2 installation required — the `rosbags` package reads both the sqlite3 (.db3) and mcap
bag formats and deserialises sensor_msgs directly (`pip install rosbags`).

Output naming matches the videos already in videos/: <bag_name>_<topic_slug>.mp4, so
/cam_0/image_raw from bag test3_0-003 becomes test3_0-003_cam_0_image_raw.mp4.

Playback speed comes from the message timestamps, not a guess: the real capture rate is
computed per topic and written into the mp4, so detector timings and anything measured off
these videos correspond to real robot time. Frames are written in bag order.

Both raw sensor_msgs/msg/Image and sensor_msgs/msg/CompressedImage are handled. Depth topics
are skipped by default (a depth map rendered as video is a visualisation, not detector input);
--include-depth normalises and writes them anyway if you want to eyeball them.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")

# encodings that carry a depth or single-channel measurement rather than a picture
DEPTH_ENCODINGS = ("16uc1", "32fc1", "mono16", "16sc1")


def topic_slug(topic: str) -> str:
    """/cam_0/image_raw -> cam_0_image_raw"""
    return re.sub(r"[^A-Za-z0-9]+", "_", topic).strip("_")


def find_bags(root: Path) -> list:
    """A bag is either a directory holding metadata.yaml, or a standalone .mcap/.db3 file."""
    if root.is_file() and root.suffix in (".mcap", ".db3"):
        return [root]
    bags = []
    if (root / "metadata.yaml").exists():
        return [root]
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and (entry / "metadata.yaml").exists():
            bags.append(entry)
        elif entry.is_file() and entry.suffix in (".mcap", ".db3"):
            bags.append(entry)
    return bags


def raw_to_rgb(data: np.ndarray, height: int, width: int, step: int,
               encoding: str) -> np.ndarray | None:
    """sensor_msgs/Image buffer -> (h, w, 3) uint8 RGB. None if the encoding is unsupported.

    `step` is the row stride in bytes and is NOT always width*channels — rows can be padded,
    so every reshape goes through step and then trims. Getting this wrong produces a sheared
    image that still looks like a picture, which is the worst kind of bug.
    """
    enc = encoding.lower()
    buf = np.frombuffer(data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        rows = buf[:height * step].reshape(height, step)[:, :width * 3]
        img = rows.reshape(height, width, 3)
        return img[..., ::-1] if enc == "bgr8" else img

    if enc in ("rgba8", "bgra8"):
        rows = buf[:height * step].reshape(height, step)[:, :width * 4]
        img = rows.reshape(height, width, 4)[..., :3]
        return img[..., ::-1] if enc == "bgra8" else img

    if enc in ("mono8", "8uc1"):
        rows = buf[:height * step].reshape(height, step)[:, :width]
        return np.repeat(rows.reshape(height, width, 1), 3, axis=2)

    if enc in DEPTH_ENCODINGS:
        dtype = np.float32 if enc == "32fc1" else (
            np.int16 if enc == "16sc1" else np.uint16)
        vals = np.frombuffer(data, dtype=dtype)[:height * width]
        if vals.size < height * width:
            return None
        d = vals.reshape(height, width).astype(np.float32)
        finite = d[np.isfinite(d) & (d > 0)]
        if finite.size == 0:
            return np.zeros((height, width, 3), np.uint8)
        lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
        norm = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
        g = (norm * 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)

    return None


def compressed_to_rgb(data: bytes) -> np.ndarray | None:
    """CompressedImage payload is a jpeg/png byte stream — let PIL decode it."""
    import io

    from PIL import Image as PILImage
    try:
        return np.asarray(PILImage.open(io.BytesIO(bytes(data))).convert("RGB"))
    except Exception:
        return None


def convert_bag(bag: Path, out_dir: Path, topics_wanted, include_depth: bool,
                fps_override: float, max_frames: int) -> list:
    from rosbags.highlevel import AnyReader

    results = []
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.msgtype in IMAGE_TYPES]
        if topics_wanted:
            conns = [c for c in conns if c.topic in topics_wanted]
        if not conns:
            print(f"  {bag.name}: no image topics", flush=True)
            return results

        for conn in conns:
            slug = topic_slug(conn.topic)
            # first pass is unnecessary: collect frames while reading, then write once we know
            # the true rate from the timestamps
            frames, stamps = [], []
            skipped_enc = None
            for _, t_ns, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                if conn.msgtype.endswith("CompressedImage"):
                    img = compressed_to_rgb(msg.data)
                else:
                    enc = str(msg.encoding).lower()
                    if enc in DEPTH_ENCODINGS and not include_depth:
                        skipped_enc = enc
                        break
                    img = raw_to_rgb(msg.data, msg.height, msg.width, msg.step, enc)
                    if img is None:
                        skipped_enc = enc
                        break
                if img is None:
                    continue
                frames.append(img)
                stamps.append(t_ns)
                if max_frames and len(frames) >= max_frames:
                    break

            if skipped_enc:
                why = ("depth topic (use --include-depth)" if skipped_enc in DEPTH_ENCODINGS
                       else f"unsupported encoding '{skipped_enc}'")
                print(f"  {conn.topic}: skipped — {why}", flush=True)
                continue
            if not frames:
                print(f"  {conn.topic}: no decodable frames", flush=True)
                continue

            if fps_override:
                fps = fps_override
            elif len(stamps) > 1:
                span_s = (stamps[-1] - stamps[0]) / 1e9
                fps = (len(stamps) - 1) / span_s if span_s > 0 else 30.0
            else:
                fps = 30.0
            fps = float(np.clip(fps, 1.0, 120.0))

            import imageio.v2 as imageio
            out_path = out_dir / f"{bag.stem if bag.is_file() else bag.name}_{slug}.mp4"
            writer = imageio.get_writer(str(out_path), fps=fps)
            h, w = frames[0].shape[:2]
            for img in frames:
                if img.shape[:2] != (h, w):          # a topic that changes resolution mid-bag
                    from PIL import Image as PILImage
                    img = np.asarray(PILImage.fromarray(img).resize((w, h)))
                writer.append_data(img)
            writer.close()

            dur = len(frames) / fps
            print(f"  {conn.topic} -> {out_path.name}  "
                  f"{len(frames)} frames, {w}x{h}, {fps:.1f} fps, {dur:.1f}s", flush=True)
            results.append({"bag": bag.name, "topic": conn.topic, "path": str(out_path),
                            "frames": len(frames), "fps": round(fps, 2),
                            "size": [w, h]})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", default="ros2_bags",
                    help="directory of bags (or a single bag directory / .mcap file)")
    ap.add_argument("--out", default="videos")
    ap.add_argument("--topic", action="append", default=[],
                    help="only this topic; repeatable. Default: every image topic")
    ap.add_argument("--include-depth", action="store_true",
                    help="also render depth topics as normalised grayscale video")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="override output fps; default is the bag's real capture rate")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames per topic")
    ap.add_argument("--list", action="store_true",
                    help="just list the image topics in each bag and exit")
    args = ap.parse_args()

    root = Path(args.bags)
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    bags = find_bags(root)
    if not bags:
        raise SystemExit(f"no ROS2 bags found under {root} "
                         "(expected directories containing metadata.yaml, or .mcap files)")

    print(f"{len(bags)} bag(s) under {root}\n", flush=True)

    if args.list:
        from rosbags.highlevel import AnyReader
        for bag in bags:
            print(f"{bag.name}:", flush=True)
            with AnyReader([bag]) as reader:
                for c in reader.connections:
                    mark = "  *" if c.msgtype in IMAGE_TYPES else "   "
                    print(f"{mark} {c.topic}  [{c.msgtype}]  {c.msgcount} msgs", flush=True)
            print("", flush=True)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for bag in bags:
        print(f"{bag.name}:", flush=True)
        try:
            all_results += convert_bag(bag, out_dir, set(args.topic), args.include_depth,
                                       args.fps, args.max_frames)
        except Exception as e:
            print(f"  FAILED: {e.__class__.__name__}: {e}", flush=True)
        print("", flush=True)

    print(f"{len(all_results)} video(s) written to {out_dir}/", flush=True)
    if all_results:
        total = sum(r["frames"] for r in all_results)
        print(f"{total} frames total. Next:\n"
              f"  python tools/dino_video.py --video '{out_dir}/*.mp4' --every 3 "
              f"--model IDEA-Research/grounding-dino-base --out dino_videos", flush=True)


if __name__ == "__main__":
    main()