#!/usr/bin/env python3
"""Paper figure: per-condition trajectory overlays + error-variance box plots.

Consumes the series.npy dumps that eval_openloop.py already writes -- no GPU,
no bag re-read required (unless you want exact odom poses, see --odom).

Layout (mirrors the ToD reference figure):
  Row 1  world-frame path: ground truth (dashed black) + predicted 8-waypoint
         trajectories overlaid every --every frames, one color per checkpoint.
  Row 2  |endpoint lateral error| box plots (cm) per checkpoint.
  Row 3  |endpoint forward error| box plots (cm) per checkpoint -- reported
         separately per the lateral-first convention (speed-dominated).

Pose source for row 1:
  * exact:        --odom BAG:PATH  where PATH is an .npz with arrays
                  frames (raw indices matching series "frames"), x, y, yaw.
  * reconstruct:  default fallback -- chains actual_traj step displacements
                  with the chord-to-arc heading relation (dyaw = 2*atan2(dy,dx)).
                  Good for previews; use odom for the camera-ready version.

Usage:
  python fig_error_variance.py \
    --run "v10@6k:e1:$S/v10_6000/eval_e1/series.npy" \
    --run "v10@6k:e2:..." --run "v8@9k:e1:..." ... \
    [--odom "e1:odom_e1.npz"] [--window all|prompted] [--every 12] \
    [--title e1="e1 - right turn (held-out)"] \
    --out figures/fig_error_variance
"""
import argparse
import os
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COL2 = 7.16  # IEEE two-column full text width, inches
COLORS = ["#D62728", "#2CA02C", "#9467BD", "#1F77B4", "#FF7F0E", "#8C564B"]
GT = "#000000"


def load_series(path):
    d = np.load(path, allow_pickle=True)
    return d.item() if hasattr(d, "item") and d.dtype == object else dict(d)


def _step_label(n):
    n = int(n)
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


def derive_labels(payload, path):
    """(checkpoint_label, bag_label) from what the dump recorded about itself.

    Falls back to directory names when the paths don't match the usual
    oft_ckpts_vN / rosbag2-<building>-<id> conventions.
    """
    import re

    ckpt = str(payload.get("checkpoint", "") or "")
    ver = re.search(r"oft_ckpts_(v\d+)", ckpt)
    step = re.search(r"--(\d+)_chkpt", ckpt)
    if ver and step:
        cl = f"{ver.group(1)}@{_step_label(step.group(1))}"
    elif ver:
        cl = ver.group(1)
    elif ckpt:
        cl = os.path.basename(ckpt.rstrip("/"))[:24]
    else:
        cl = os.path.basename(os.path.dirname(os.path.dirname(path)))

    bag = str(payload.get("bag", "") or "")
    base = os.path.basename(bag.rstrip("/"))
    bl = base.rsplit("-", 1)[-1] if base else os.path.basename(
        os.path.dirname(path))
    return cl, bl


def _sort_key(label):
    """Natural-ish ordering: v8@9k < v10@6k < v10@18k; e1 < e2 < t1."""
    import re

    nums = [int(x) for x in re.findall(r"\d+", label)]
    return (re.sub(r"\d+", "", label), nums)


def discover(sweep_dirs):
    """Walk sweep dirs for series.npy; return {(ckpt_label, bag_label): path}."""
    found = {}
    for root_dir in sweep_dirs:
        for dirpath, _, files in os.walk(root_dir):
            if "series.npy" not in files:
                continue
            path = os.path.join(dirpath, "series.npy")
            try:
                payload = load_series(path)
                key = derive_labels(payload, path)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {path}: {e}")
                continue
            if key in found:
                print(f"  [warn] duplicate {key}; keeping {found[key]}")
                continue
            found[key] = path
    return found


def prompted_mask(d):
    """Frames inside the standing-decision window [flip, turn_done]."""
    frames = np.asarray(d["frames"])
    flip = d.get("flip_frame", 0) or 0
    done = d.get("turn_done_frame", None)
    m = frames >= int(flip)
    if done not in (None, "-", "None"):
        try:
            m &= frames <= int(done)
        except (TypeError, ValueError):
            pass
    return m


def chain_poses(actual_traj):
    """Reconstruct world poses by chaining per-step GT displacements.

    Position update: wp1 (displacement to the next strided frame, in the
    current frame) rotated into the running world frame. Heading update:
    direction of the wp1->wp2 chord in the current frame, which approximates
    the heading at the next frame relative to this one. Good for previews;
    use --odom for the exact camera-ready version.
    """
    traj = np.asarray(actual_traj, dtype=np.float64)
    n = len(traj)
    poses = np.zeros((n, 3), dtype=np.float64)
    for i in range(n - 1):
        dx, dy = traj[i, 0]
        x, y, yaw = poses[i]
        poses[i + 1, 0] = x + dx * np.cos(yaw) - dy * np.sin(yaw)
        poses[i + 1, 1] = y + dx * np.sin(yaw) + dy * np.cos(yaw)
        cx, cy = traj[i, 1] - traj[i, 0]          # wp1 -> wp2 chord
        dyaw = np.arctan2(cy, cx) if np.hypot(cx, cy) > 1e-4 else 0.0
        poses[i + 1, 2] = yaw + dyaw
    return poses


def poses_from_odom(npz_path, series_frames):
    z = np.load(npz_path)
    idx = {int(f): i for i, f in enumerate(np.asarray(z["frames"]))}
    rows = []
    for f in np.asarray(series_frames):
        i = idx.get(int(f))
        rows.append([z["x"][i], z["y"][i], z["yaw"][i]] if i is not None
                    else [np.nan] * 3)
    p = np.asarray(rows, dtype=np.float64)
    bad = np.isnan(p[:, 0]).mean()
    if bad > 0.5 and len(series_frames) <= len(z["frames"]):
        # frame-index conventions differ (raw vs step index) -- align by order
        print(f"[odom] index match failed ({bad:.0%} miss); "
              "falling back to positional alignment")
        n = len(series_frames)
        p = np.stack([z["x"][:n], z["y"][:n], z["yaw"][:n]], axis=1)
    # re-anchor to start at origin, heading +x, so panels are comparable
    x0, y0, yaw0 = p[0]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    q = p.copy()
    q[:, 0] = c * (p[:, 0] - x0) - s * (p[:, 1] - y0)
    q[:, 1] = s * (p[:, 0] - x0) + c * (p[:, 1] - y0)
    q[:, 2] = p[:, 2] - yaw0
    return q


def to_world(wp_local, pose):
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.empty_like(wp_local, dtype=np.float64)
    out[:, 0] = x + wp_local[:, 0] * c - wp_local[:, 1] * s
    out[:, 1] = y + wp_local[:, 0] * s + wp_local[:, 1] * c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", default=[],
                    help="sweep dir to walk for series.npy (repeatable); "
                         "labels are derived from each dump's own metadata")
    ap.add_argument("--run", action="append", default=[],
                    help="LABEL:BAG:PATH/to/series.npy (repeatable); "
                         "use instead of or alongside --sweep")
    ap.add_argument("--list", action="store_true",
                    help="print what was discovered and exit (run this first)")
    ap.add_argument("--ckpts", default=None,
                    help="comma-separated checkpoint labels to keep, in order")
    ap.add_argument("--bags", default=None,
                    help="comma-separated bag labels to keep, in order")
    ap.add_argument("--odom", action="append", default=[],
                    help="BAG:PATH/to/odom.npz (frames,x,y,yaw); else reconstruct")
    ap.add_argument("--title", action="append", default=[],
                    help='BAG="panel title" override')
    ap.add_argument("--window", choices=["all", "prompted"], default="all",
                    help="frames pooled into the box plots")
    ap.add_argument("--every", type=int, default=12,
                    help="overlay a predicted trajectory every N frames")
    ap.add_argument("--zoom", choices=["junction", "full"], default="junction",
                    help="row 1: crop to the turn window (+pad) or show the "
                         "whole drive")
    ap.add_argument("--pad", type=int, default=25,
                    help="frames of context around the turn window")
    ap.add_argument("--free-y", action="store_true",
                    help="give each box panel its own y scale (readable e3 "
                         "detail); default shares y across a row so panels "
                         "are directly comparable")
    ap.add_argument("--out", default="fig_error_variance")
    args = ap.parse_args()

    if not args.sweep and not args.run:
        ap.error("give at least one --sweep DIR or --run LABEL:BAG:PATH")

    paths = discover(args.sweep) if args.sweep else {}
    for spec in args.run:                       # explicit --run wins
        label, bag, path = spec.split(":", 2)
        paths[(label, bag)] = path

    if not paths:
        ap.error("no series.npy found — check the sweep path")

    ckpts = sorted({c for c, _ in paths}, key=_sort_key)
    bags = sorted({b for _, b in paths}, key=_sort_key)
    if args.ckpts:
        want = [s.strip() for s in args.ckpts.split(",")]
        missing = [w for w in want if w not in ckpts]
        if missing:
            ap.error(f"--ckpts not found: {missing}. Available: {ckpts}")
        ckpts = want
    if args.bags:
        want = [s.strip() for s in args.bags.split(",")]
        missing = [w for w in want if w not in bags]
        if missing:
            ap.error(f"--bags not found: {missing}. Available: {bags}")
        bags = want

    if args.list:
        print(f"\ncheckpoints: {ckpts}\nbags:        {bags}\n")
        for (c, b), p in sorted(paths.items(), key=lambda kv: kv[0]):
            mark = " " if (c in ckpts and b in bags) else "-"
            print(f" {mark} {c:>10s}  {b:<4s}  {p}")
        print("\n(rows marked '-' are filtered out; '-' with no rows means all "
              "are included)")
        return

    runs = OrderedDict()   # (ckpt label, bag) -> series dict
    for c in ckpts:
        for b in bags:
            if (c, b) in paths:
                runs[(c, b)] = load_series(paths[(c, b)])
    if not runs:
        ap.error("filters left nothing to plot")
    odom = dict(s.split(":", 1) for s in args.odom)
    titles = dict(s.split("=", 1) for s in args.title)
    color = {c: COLORS[i % len(COLORS)] for i, c in enumerate(ckpts)}

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    ncol = len(bags)
    fig, axes = plt.subplots(
        3, ncol, figsize=(COL2, 5.2),
        gridspec_kw={"height_ratios": [2.1, 1, 1], "hspace": 0.55,
                     "wspace": 0.22})
    axes = np.atleast_2d(axes)
    if ncol == 1:
        axes = axes.reshape(3, 1)
    if not args.free_y:
        for r in (1, 2):                   # share y within each box-plot row
            for j in range(1, ncol):
                axes[r, j].sharey(axes[r, 0])
                axes[r, j].tick_params(labelleft=False)

    # world poses per bag (from the first checkpoint that has this bag --
    # GT is identical across checkpoints)
    poses = {}
    for bag in bags:
        d = next(runs[(c, bag)] for c in ckpts if (c, bag) in runs)
        if bag in odom:
            poses[bag] = poses_from_odom(odom[bag], d["frames"])
        else:
            poses[bag] = chain_poses(np.asarray(d["actual_traj"]))

    # ---- row 1 display windows ------------------------------------------
    # junction zoom: crop turn bags to |endpoint lateral| > 30% of peak,
    # padded; straight bags get a central segment of matching length so the
    # panels stay at comparable scale.
    first = {bag: next(runs[(c, bag)] for c in ckpts if (c, bag) in runs)
             for bag in bags}
    windows = {}
    if args.zoom == "junction":
        turn_lens = []
        for bag in bags:
            al = np.abs(np.asarray(first[bag]["actual_lat"], dtype=np.float64))
            if al.max() * 100.0 >= 2.0:            # a real turn is 10-20 cm
                idx = np.where(al > 0.3 * al.max())[0]
                a = max(0, int(idx[0]) - args.pad)
                b = min(len(al), int(idx[-1]) + args.pad)
                windows[bag] = (a, b)
                turn_lens.append(b - a)
        span = max(turn_lens) if turn_lens else None
        for bag in bags:                            # straight bags
            if bag not in windows:
                n = len(first[bag]["actual_lat"])
                L = min(span or n, n)
                a = (n - L) // 2
                windows[bag] = (a, a + L)
    else:
        for bag in bags:
            windows[bag] = (0, len(first[bag]["actual_lat"]))

    # ---- row 1: trajectories --------------------------------------------
    # Orientation: forward (world x) is UP; lateral (world y, +left) is
    # horizontal with the axis inverted so a left turn bends left on the page.
    for j, bag in enumerate(bags):
        ax = axes[0, j]
        a, b = windows[bag]
        P = poses[bag]
        hs, vs = [P[a:b, 1]], [P[a:b, 0]]  # collect extents for limits
        for c in ckpts:
            d = runs.get((c, bag))
            if d is None:
                continue
            pred = np.asarray(d["pred_traj"], dtype=np.float64)
            for i in range(a, b, args.every):
                if np.any(np.isnan(P[i])):
                    continue
                w = to_world(pred[i], P[i])
                w = np.vstack([P[i, :2], w])          # anchor at the robot
                ax.plot(w[:, 1], w[:, 0], color=color[c], lw=0.8, alpha=0.8,
                        solid_capstyle="round", zorder=4)
                hs.append(w[:, 1])
                vs.append(w[:, 0])
        ax.plot(P[a:b, 1], P[a:b, 0], color=GT, ls=(0, (4, 2)), lw=1.1,
                zorder=3)
        h = np.concatenate(hs)
        v = np.concatenate(vs)
        hc = 0.5 * (np.nanmin(h) + np.nanmax(h))
        hr = max(0.5 * (np.nanmax(h) - np.nanmin(h)) * 1.15, 1.0)
        vpad = 0.05 * (np.nanmax(v) - np.nanmin(v))
        ax.set_xlim(hc + hr, hc - hr)      # inverted: +y (left) on the left
        ax.set_ylim(np.nanmin(v) - vpad, np.nanmax(v) + vpad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles.get(bag, bag), pad=3)
        ax.set_xlabel("y [m]", labelpad=1)
        if j == 0:
            ax.set_ylabel("x [m]  (forward)")
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)

    # ---- rows 2-3: error box plots --------------------------------------
    rows = [("pred_lat", "actual_lat", "Lateral\nError [cm]"),
            ("pred_fwd", "actual_fwd", "Forward\nError [cm]")]
    for r, (pk, ak, ylabel) in enumerate(rows, start=1):
        for j, bag in enumerate(bags):
            ax = axes[r, j]
            data, cols = [], []
            for c in ckpts:
                d = runs.get((c, bag))
                if d is None:
                    continue
                err = np.abs(np.asarray(d[pk]) - np.asarray(d[ak])) * 100.0
                if args.window == "prompted":
                    err = err[prompted_mask(d)]
                data.append(err)
                cols.append(color[c])
            bp = ax.boxplot(
                data, positions=np.arange(len(data)), widths=0.62,
                patch_artist=True, showmeans=True, showfliers=False,
                whis=1.5,
                medianprops=dict(color="black", lw=0.9),
                meanprops=dict(marker="D", markersize=3,
                               markerfacecolor="#DDDDDD",
                               markeredgecolor="black", markeredgewidth=0.5),
                boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                capprops=dict(lw=0.6))
            for patch, fc in zip(bp["boxes"], cols):
                patch.set_facecolor(fc)
                patch.set_alpha(0.45)
                patch.set_edgecolor(fc)
            ax.set_xticks([])
            pad = {1: 1.1, 2: 0.8}.get(len(data), 0.6)
            ax.set_xlim(-pad, len(data) - 1 + pad)
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25, lw=0.4)
            ax.set_axisbelow(True)

    # ---- legend ----------------------------------------------------------
    handles = [Line2D([], [], color=GT, ls=(0, (4, 2)), lw=1.1,
                      label="Ground truth")]
    handles += [Patch(facecolor=color[c], alpha=0.55, edgecolor=color[c],
                      label=c) for c in ckpts]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.6, handlelength=1.6)
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.09, right=0.99)

    for ext in ("pdf", "png"):
        path = f"{args.out}.{ext}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()