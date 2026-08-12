#!/usr/bin/env python3
"""
train_deadline.py -- two-stage training for the deadline estimator.

Stage 1 (--extract): read bags, embed every labeled frame with frozen DINOv2-S,
cache features per bag to <out>/features/<bag>.npz. GPU pass, run once; re-runs
skip cached bags.

Stage 2 (--train): load cached features + labels, train the two-head MLP.
Split by BAG using the annotation split column (val bags never leak).
Reports: cls accuracy/AUC-ish, distance MAE per bucket, and **q10 coverage**
(fraction of val frames with true d >= predicted q10) -- target ~0.90; the
shortfall p90 becomes the gate's calibrated margin.

    python -m adaptive_reasoning.deadline.train_deadline \
        --labels $SCRATCH/deadline_labels/deadline_labels_all.csv \
        --bag-root /users/1/munda057/SignWay/ros2_bags \
        --out $SCRATCH/deadline_model --extract --train
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_reasoning.config import bootstrap  # noqa: E402
from adaptive_reasoning.deadline.model import (  # noqa: E402
    QUANTILES, build_head, embed_frames, load_backbone, pinball_loss, save_metrics)


def load_labels(path: Path):
    per_bag = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            per_bag[r["bag"]].append(dict(
                frame_index=int(r["frame_index"]), d_m=float(r["d_m"]),
                junction_ahead=int(r["junction_ahead"]),
                past_junction=int(r["past_junction"]), split=r["split"]))
    return per_bag


def extract(args, per_bag):
    bootstrap()
    from bag_to_episode import read_bag

    import torch  # noqa: F401  (fail early with a clear message if missing)

    feat_dir = Path(args.out) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    backbone = load_backbone(args.device)
    for bag, rows in per_bag.items():
        fpath = feat_dir / f"{bag}.npz"
        if fpath.exists():
            print(f"[skip cached] {bag}")
            continue
        frames, _ = read_bag(Path(args.bag_root) / bag, args.image_topic,
                             args.odom_topic, args.ros_distro)
        # subsample for training efficiency: every args.frame_stride-th labeled frame
        keep = [r for r in rows if r["frame_index"] % args.frame_stride == 0
                and r["frame_index"] < len(frames)]
        imgs = [frames[r["frame_index"]][1] for r in keep]
        feats = embed_frames(backbone, imgs, args.device)
        np.savez_compressed(
            fpath, feats=feats,
            frame_index=np.array([r["frame_index"] for r in keep]),
            d_m=np.array([r["d_m"] for r in keep], dtype=np.float32),
            junction_ahead=np.array([r["junction_ahead"] for r in keep], dtype=np.int8),
            past_junction=np.array([r["past_junction"] for r in keep], dtype=np.int8))
        print(f"[extracted] {bag}: {len(keep)} frames")


def train(args, per_bag):
    import torch

    feat_dir = Path(args.out) / "features"
    Xs, ys_cls, ys_d, splits = [], [], [], []
    for bag, rows in per_bag.items():
        fpath = feat_dir / f"{bag}.npz"
        if not fpath.exists():
            print(f"[warn] no features for {bag} (run --extract)")
            continue
        z = np.load(fpath)
        m = z["past_junction"] == 0            # train only on pre-onset frames
        Xs.append(z["feats"][m])
        ys_cls.append(z["junction_ahead"][m])
        ys_d.append(z["d_m"][m])
        splits += [rows[0]["split"]] * int(m.sum())
    X = np.concatenate(Xs); yc = np.concatenate(ys_cls).astype(np.float32)
    yd = np.concatenate(ys_d).astype(np.float32); sp = np.array(splits)
    tr, va = sp != "val", sp == "val"
    print(f"train frames {tr.sum()}  val frames {va.sum()}  "
          f"(junction_ahead: {int(yc[tr].sum())}/{int(yc[va].sum())})")

    dev = args.device
    head = build_head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X[tr]).float().to(dev)
    yct = torch.from_numpy(yc[tr]).to(dev)
    ydt = torch.from_numpy(yd[tr]).to(dev)
    bce = torch.nn.BCEWithLogitsLoss()

    n = Xt.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            logit, q = head(Xt[idx])
            loss = bce(logit, yct[idx])
            m = yct[idx] > 0.5                      # d supervised only where ahead
            if m.any():
                loss = loss + pinball_loss(q[m], ydt[idx][m])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {tot / n:.4f}")

    # ---- validation: the numbers the gate's margin comes from -----------------
    head.eval()
    with torch.inference_mode():
        Xv = torch.from_numpy(X[va]).float().to(dev)
        logit, q = head(Xv)
        p = torch.sigmoid(logit).cpu().numpy()
        qv = np.sort(q.cpu().numpy(), axis=1)      # monotone
    ycv, ydv = yc[va], yd[va]
    cls_acc = float((((p > 0.5) == (ycv > 0.5))).mean())
    m = ycv > 0.5
    mae = float(np.abs(qv[m, 1] - ydv[m]).mean()) if m.any() else float("nan")
    cover = float((ydv[m] >= qv[m, 0]).mean()) if m.any() else float("nan")
    short = np.maximum(qv[m, 0] - ydv[m], 0.0)     # metres q10 OVERestimated d
    short_p90 = float(np.percentile(short, 90)) if m.any() else float("nan")

    buckets = {}
    for lo in range(0, 15, 3):
        bm = m & (ydv >= lo) & (ydv < lo + 3)
        if bm.any():
            buckets[f"{lo}-{lo+3}m"] = dict(
                mae_q50=float(np.abs(qv[bm, 1] - ydv[bm]).mean()),
                q10_coverage=float((ydv[bm] >= qv[bm, 0]).mean()), n=int(bm.sum()))

    metrics = dict(cls_acc=cls_acc, d_mae_q50=mae, q10_coverage=cover,
                   q10_shortfall_p90_m=short_p90, buckets=buckets,
                   note=("gate margin_s >= q10_shortfall_p90_m / typical_speed; "
                         "coverage target ~0.90"))
    out = Path(args.out)
    torch.save({"head": head.state_dict(),
                "meta": dict(quantiles=QUANTILES, metrics=metrics)},
               out / "deadline_head.pt")
    save_metrics(out / "metrics.json", metrics)
    print("\n== validation ==")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nsaved -> {out / 'deadline_head.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--bag-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--frame-stride", type=int, default=2,
                    help="embed every Nth labeled frame (training efficiency)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    args = ap.parse_args()

    per_bag = load_labels(Path(args.labels))
    if args.extract:
        extract(args, per_bag)
    if args.train:
        train(args, per_bag)
    if not (args.extract or args.train):
        print("nothing to do: pass --extract and/or --train")


if __name__ == "__main__":
    main()
