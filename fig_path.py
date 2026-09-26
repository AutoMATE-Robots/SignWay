#!/usr/bin/env python3
"""SignWay paper figure: predicted vs reference trajectory in world x-y, with
lateral error distributions underneath.

How the predicted path is built (this is the part that matters):
every frame, the policy's 8-waypoint prediction is anchored at that frame's
TRUE pose and transformed into world coordinates. The locus of predicted
endpoints is then one continuous curve spanning the whole corridor. Because
each point is independently anchored on odometry, error does NOT compound --
the separation between the two curves at any location is that frame's
prediction error and nothing else.

(Contrast with dead reckoning, where the policy's own heading estimate is
integrated: there a 2-degree error early swings everything after it, and what
you are looking at is mostly accumulated bookkeeping.)

Layout, one column per condition:
  Row 1  world x-y trajectory: reference (black dashed) vs prediction (colour).
         Each panel autoscales -- a turn and a straight corridor need different
         ranges to read well.
  Row 2  |lateral error| in cm, split by phase for turn panels.

Usage:
  python fig_path.py --sweep $S --ckpt v8@9k --bags "e2,e3" --out figures/fig_path
"""
import argparse
import os
import re
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

COL2 = 7.16
PRED = "#C8102E"
GT = "#000000"
# one per checkpoint; distinguishable in greyscale print order too
COLORS = ["#C8102E", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E", "#8C564B"]


# ---------------------------------------------------------------- discovery
def load_series(path):
    d = np.load(path, allow_pickle=True)
    return d.item() if hasattr(d, "item") and d.dtype == object else dict(d)


def _step_label(n):
    n = int(n)
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


def derive_labels(payload, path):
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
    return (re.sub(r"\d+", "", label),
            [int(x) for x in re.findall(r"\d+", label)])


def discover(sweep_dirs):
    found = {}
    for root_dir in sweep_dirs:
        for dirpath, _, files in os.walk(root_dir):
            if "series.npy" in files:
                p = os.path.join(dirpath, "series.npy")
                try:
                    found.setdefault(derive_labels(load_series(p), p), p)
                except Exception as e:  # noqa: BLE001
                    print(f"  [skip] {p}: {e}")
    return found


# ---------------------------------------------------------------- geometry
def dyaw_from_wps(wp, kmax=4):
    """Heading change over one step from the waypoint fan (bearing to wp k is
    k*dyaw/2, so dyaw = 2*bearing_k/k; averaging small k stabilises it)."""
    est = []
    for k in range(1, min(kmax, len(wp)) + 1):
        x, y = wp[k - 1]
        if np.hypot(x, y) > 1e-4:
            est.append(2.0 * np.arctan2(y, x) / k)
    return float(np.mean(est)) if est else 0.0


def odom_path(npz_path, series_frames):
    """True poses from the bag: no reconstruction, no heading model."""
    z = np.load(npz_path)
    idx = {int(f): i for i, f in enumerate(np.asarray(z["frames"]))}
    rows = [[z["x"][idx[int(f)]], z["y"][idx[int(f)]], z["yaw"][idx[int(f)]]]
            if int(f) in idx else [np.nan] * 3 for f in np.asarray(series_frames)]
    p = np.asarray(rows, dtype=np.float64)
    if np.isnan(p[:, 0]).mean() > 0.5 and len(series_frames) <= len(z["frames"]):
        n = len(series_frames)
        print("[odom] frame indices did not match; aligning by order")
        p = np.stack([z["x"][:n], z["y"][:n], z["yaw"][:n]], axis=1)
    x0, y0, yaw0 = p[0]                     # re-anchor: start at origin, +x
    c, s_ = np.cos(-yaw0), np.sin(-yaw0)
    q = np.empty_like(p)
    q[:, 0] = c * (p[:, 0] - x0) - s_ * (p[:, 1] - y0)
    q[:, 1] = s_ * (p[:, 0] - x0) + c * (p[:, 1] - y0)
    q[:, 2] = p[:, 2] - yaw0
    return q[:, :2], q[:, 2]


def rel_pose(A, B):
    """Rigid transform taking points B into frame A (2D Kabsch).

    A and B are the SAME physical points expressed in two consecutive robot
    frames, so the result is exactly the later frame's pose in the earlier
    frame's coordinates -- rotation included, no heading heuristic.
    """
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb
    num = float((B0[:, 0] * A0[:, 1] - B0[:, 1] * A0[:, 0]).sum())
    den = float((B0[:, 0] * A0[:, 0] + B0[:, 1] * A0[:, 1]).sum())
    th = np.arctan2(num, den)
    c, s = np.cos(th), np.sin(th)
    return th, ca - np.array([[c, -s], [s, c]]) @ cb


def reference_path(actual_traj, robust=5, diag=False):
    """Odometry path recovered from the dataset's own targets.

    Frame i's waypoints are positions of frames i+1..i+8 in frame i; frame
    i+1's are i+2..i+9 in frame i+1. Seven points are shared, which determines
    the relative pose exactly. Chaining those recovers the driven path without
    needing the bag.
    """
    traj = np.asarray(actual_traj, dtype=np.float64)
    n = len(traj)
    steps = [rel_pose(traj[i][1:], traj[i + 1][:-1]) for i in range(n - 1)]
    ths = np.array([t[0] for t in steps])
    ts = np.array([t[1] for t in steps])

    # One bad per-step rotation injects a heading offset that PERSISTS for the
    # rest of the path -- it shows up as a kink followed by a constant diagonal.
    # A median filter removes isolated outliers; a real turn spans many steps
    # and survives it untouched.
    # Outlier test must be LOCAL: a sustained turn is legitimately several
    # deg/step, so comparing against the whole-bag median just flags the turn.
    # An isolated bad sample is a step that disagrees with its NEIGHBOURS.
    if diag:
        w5 = np.pad(ths, 2, mode="edge")
        loc = np.array([np.median(w5[i:i + 5]) for i in range(len(ths))])
        resid = ths - loc
        scale = np.median(np.abs(resid - np.median(resid))) + 1e-12
        spikes = int((np.abs(resid) > 8 * scale).sum())
        w = int(np.argmax(np.abs(resid)))
        print(f"    [recon] rotation |max| {np.degrees(np.abs(ths).max()):.2f} "
              f"deg/step (sustained turn is fine); net heading "
              f"{np.degrees(ths.sum()):+.1f} deg; {spikes} isolated spike(s); "
              f"worst local deviation step {w} at "
              f"{np.degrees(resid[w]):+.2f} deg vs neighbours")
    if robust and robust > 1:
        w = int(robust) | 1
        pad = w // 2
        padded = np.pad(ths, pad, mode="edge")
        ths = np.array([np.median(padded[i:i + w]) for i in range(len(ths))])

    xy = np.zeros((n, 2))
    yaws = np.zeros(n)
    for i in range(n - 1):
        t = ts[i]
        c, s = np.cos(yaws[i]), np.sin(yaws[i])
        xy[i + 1] = xy[i] + [t[0] * c - t[1] * s, t[0] * s + t[1] * c]
        yaws[i + 1] = yaws[i] + ths[i]
    return xy, yaws


def to_world(wp, xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return xy + np.stack([wp[:, 0] * c - wp[:, 1] * s,
                          wp[:, 0] * s + wp[:, 1] * c], axis=1)


def endpoint_locus(traj, xy, yaws):
    """Each frame's predicted endpoint, anchored at that frame's true pose."""
    traj = np.asarray(traj, dtype=np.float64)
    out = np.empty((len(traj), 2))
    for i in range(len(traj)):
        out[i] = to_world(traj[i][-1:], xy[i], yaws[i])[0]
    return out


def phases(d, thresh_cm=2.0):
    lat = np.asarray(d["actual_lat"])
    a = np.abs(lat)
    if a.max() * 100.0 < thresh_cm:
        return [("straight", np.ones(len(lat), dtype=bool))]
    inturn = a > 0.30 * a.max()
    idx = np.where(inturn)[0]
    lo = 0
    f = d.get("flip_frame", None)
    if f not in (None, "", "None"):
        try:
            lo = int(np.searchsorted(np.asarray(d["frames"]), int(f)))
        except (TypeError, ValueError):
            lo = 0
    app = np.zeros(len(lat), dtype=bool)
    app[lo:idx[0]] = True
    trn = np.zeros(len(lat), dtype=bool)
    trn[idx[0]:idx[-1] + 1] = True
    return [("approach", app), ("turn", trn)]


# ---------------------------------------------------------------- figure
def load_extra_path(npz_path):
    """Odom path from dump_odom.py, re-anchored to start at origin heading +x.

    NOTE: a different drive. Aligned by start pose only -- context, not a
    registered comparison.
    """
    z = np.load(npz_path)
    x, y, yaw = np.asarray(z["x"]), np.asarray(z["y"]), np.asarray(z["yaw"])
    c, s_ = np.cos(-yaw[0]), np.sin(-yaw[0])
    dx, dy = x - x[0], y - y[0]
    return np.stack([c * dx - s_ * dy, s_ * dx + c * dy], axis=1)


def draw_bin(ax, x, y, h=0.55, w=0.34, label=None):
    """Schematic dustbin at world (x forward, y lateral), in metres.

    Drawn in data coordinates so it scales with the panel. Axis order is
    (horizontal = y lateral, vertical = x forward).
    """
    body = np.array([[y - w * 0.36, x],
                     [y + w * 0.36, x],
                     [y + w * 0.50, x + h],
                     [y - w * 0.50, x + h]])
    ax.add_patch(Polygon(body, closed=True, facecolor="#9AA0A6",
                         edgecolor="#4D5156", lw=0.7, zorder=6))
    lid = np.array([[y - w * 0.58, x + h],
                    [y + w * 0.58, x + h],
                    [y + w * 0.58, x + h * 1.10],
                    [y - w * 0.58, x + h * 1.10]])
    ax.add_patch(Polygon(lid, closed=True, facecolor="#4D5156",
                         edgecolor="#4D5156", lw=0.7, zorder=7))
    for f in (-0.18, 0.18):                       # a couple of body ridges
        ax.plot([y + w * f, y + w * f], [x + h * 0.12, x + h * 0.88],
                color="#4D5156", lw=0.4, alpha=0.7, zorder=8)
    if label:
        ax.annotate(label, xy=(y, x + h * 1.10), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6, color="#4D5156", zorder=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", default=[],
                    help="sweep dir to walk (repeatable -- pass several to "
                         "compare checkpoints that live in different sweeps)")
    ap.add_argument("--run", action="append", default=[],
                    help="LABEL:BAG:PATH/to/series.npy")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ckpts", "--ckpt", dest="ckpts", default=None,
                    help="comma-separated checkpoint labels, in plot order")
    ap.add_argument("--bags", default=None)
    ap.add_argument("--title", action="append", default=[],
                    help='BAG="panel title"')
    ap.add_argument("--odom", action="append", default=[],
                    help="BAG:PATH/to/odom.npz from dump_odom.py. Optional: "
                         "without it the reference path is reconstructed from "
                         "the waypoint targets by rigid alignment, which is "
                         "exact per step but accumulates slowly.")
    ap.add_argument("--deviation", action="store_true",
                    help="add an unrolled deviation row: lateral offset of "
                         "each prediction from the reference, vs distance "
                         "along the path. Pose-independent, so it can be "
                         "zoomed as far as you like.")
    ap.add_argument("--arcs", type=int, default=0,
                    help="also draw a full 8-waypoint arc every N frames "
                         "(first checkpoint only)")
    ap.add_argument("--robust", type=int, default=5,
                    help="median-filter width on the reconstructed per-step "
                         "rotation; 0 = off")
    ap.add_argument("--diag", action="store_true",
                    help="report per-step rotation stats and outliers")
    ap.add_argument("--crop", choices=["none", "turn"], default="none",
                    help="'turn' zooms each panel to the turn window")
    ap.add_argument("--crop-pad", type=float, default=0.17,
                    help="context around the turn window as a fraction of its "
                         "length; smaller = tighter zoom")
    ap.add_argument("--min-span", type=float, default=0.5,
                    help="minimum lateral width in metres for a panel")
    ap.add_argument("--smooth", type=int, default=1,
                    help="rolling mean (steps) on the predicted locus; 1 = raw")
    ap.add_argument("--stretch", type=float, default=1.0,
                    help="lateral magnification vs forward; 1 = true geometry")
    ap.add_argument("--err-max", default="turn=10,straight=1",
                    help="y-max of the lateral-error row in cm. A single "
                         "number applies to every panel; 'turn=10,straight=1' "
                         "sets them per condition; a BAG=VALUE entry overrides "
                         "one panel; 'auto' autoscales.")
    ap.add_argument("--box-width", type=float, default=0.16,
                    help="box width as a fraction of a phase slot")
    ap.add_argument("--phase-names", action="append", default=[],
                    help='rename a panel\'s phase labels, e.g. '
                         '"obst=before bin/passing bin". Useful when a swerve '
                         'trips the turn detector and would otherwise be '
                         'labelled approach/turn.')
    ap.add_argument("--turn-threshold", type=float, default=2.0,
                    help="peak |lateral| in cm above which a bag counts as a "
                         "turn (and so gets approach/turn phases and the turn "
                         "crop). Lower it for gentle manoeuvres like an "
                         "obstacle swerve.")
    ap.add_argument("--extra-path", action="append", default=[],
                    help='overlay another drive\'s odometry on one panel: '
                         '"BAG=/path/to/odom.npz,label". A different bag has '
                         'no shared frame -- aligned by start pose only, so '
                         'label it as context.')
    ap.add_argument("--extra-color", default="#1F77B4")
    ap.add_argument("--obstacle", action="append", default=[],
                    help='place a dustbin marker: "BAG=x,y" or "BAG=x,y,label" '
                         'in metres (x forward, y lateral, + = left). Each '
                         'panel prints its visible range so you can pick.')
    ap.add_argument("--equal-aspect", action="store_true")
    ap.add_argument("--out", default="fig_path")
    args = ap.parse_args()

    if not args.sweep and not args.run:
        ap.error("give --sweep DIR or --run LABEL:BAG:PATH")
    paths = discover(args.sweep) if args.sweep else {}
    for spec in args.run:
        label, bag, path = spec.split(":", 2)
        paths[(label, bag)] = path
    if not paths:
        ap.error("no series.npy found")

    ckpts = sorted({c for c, _ in paths}, key=_sort_key)
    bags = sorted({b for _, b in paths}, key=_sort_key)
    if args.list:
        print(f"\ncheckpoints: {ckpts}\nbags:        {bags}\n")
        for (c, b), p in sorted(paths.items(), key=lambda kv: kv[0]):
            print(f"   {c:>10s}  {b:<4s}  {p}")
        return

    if args.ckpts:
        want = [x.strip() for x in args.ckpts.split(",")]
        miss = [w for w in want if w not in ckpts]
        if miss:
            ap.error(f"--ckpts not found: {miss}. Available: {ckpts}")
        ckpts = want
    if args.bags:
        want = [x.strip() for x in args.bags.split(",")]
        miss = [w for w in want if w not in bags]
        if miss:
            ap.error(f"--bags not found: {miss}. Available: {bags}")
        bags = want
    if len(ckpts) > 5:
        print(f"[warn] {len(ckpts)} checkpoints is a lot for one panel; "
              "3-4 stays readable")

    runs = {}
    for c in ckpts:
        for b in bags:
            if (c, b) in paths:
                runs[(c, b)] = load_series(paths[(c, b)])
    bags = [b for b in bags if any((c, b) in runs for c in ckpts)]
    if not runs:
        ap.error("nothing to plot")

    titles = dict(x.split("=", 1) for x in args.title)
    odom = dict(x.split(":", 1) for x in args.odom)
    missing = [b for b in bags if b not in odom]
    if missing:
        print(f"[note] no --odom for {missing}: reference path is "
              "reconstructed from the waypoint targets by rigid alignment "
              "(exact per step; small errors still accumulate over the bag). "
              "Prediction error and the deviation/box rows are unaffected.")
    color = {c: COLORS[i % len(COLORS)] for i, c in enumerate(ckpts)}

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    n = len(bags)
    nrow = 3 if args.deviation else 2
    hr = [3.0, 1.25, 1.5] if args.deviation else [3.0, 1.5]
    fig, axes = plt.subplots(
        nrow, n, figsize=(COL2 * min(1.0, 0.34 + 0.33 * n),
                          4.6 + (1.5 if args.deviation else 0.0)),
        gridspec_kw={"height_ratios": hr, "hspace": 0.45, "wspace": 0.30})
    axes = np.atleast_2d(axes)
    if n == 1:
        axes = axes.reshape(nrow, 1)

    geom = {}
    for bag in bags:
        here0 = [c for c in ckpts if (c, bag) in runs]
        d0 = runs[(here0[0], bag)]
        if bag in odom:
            ref, yaws = odom_path(odom[bag], d0["frames"])
        else:
            if args.diag:
                print(f"  [{bag}]")
            ref, yaws = reference_path(np.asarray(d0["actual_traj"]),
                                       robust=args.robust, diag=args.diag)
        lo_i, hi_i = 0, len(ref)
        is_turn = len(phases(d0, args.turn_threshold)) > 1
        if args.crop == "turn" and is_turn:
            idx = np.where(phases(d0, args.turn_threshold)[-1][1])[0]
            if len(idx) > 3:
                pad = max(2, int((idx[-1] - idx[0]) * args.crop_pad))
                lo_i = max(0, idx[0] - pad)
                hi_i = min(len(ref), idx[-1] + 1 + pad)
        geom[bag] = [ref, yaws, lo_i, hi_i, is_turn]

    # a straight bag cropped to the same PATH LENGTH as the turn panels keeps
    # every panel the same size and shape, instead of a 15 m sliver
    if args.crop == "turn":
        spans = [np.linalg.norm(np.diff(g[0][g[2]:g[3]], axis=0), axis=1).sum()
                 for g in geom.values() if g[4]]
        if spans:
            target = float(np.mean(spans))
            for bag, g in geom.items():
                if g[4]:
                    continue
                ref = g[0]
                step = np.linalg.norm(np.diff(ref, axis=0), axis=1)
                sdist = np.concatenate([[0.0], np.cumsum(step)])
                mid = sdist[-1] / 2.0
                g[2] = int(np.searchsorted(sdist, max(0.0, mid - target / 2)))
                g[3] = int(np.searchsorted(sdist, min(sdist[-1],
                                                      mid + target / 2)))
                print(f"  {bag}: straight cropped to a {target:.1f} m segment "
                      "to match the turn panel")
        # match the lateral window too, or equal aspect leaves a sliver
        lat = [float(np.ptp(g[0][g[2]:g[3], 1])) for g in geom.values() if g[4]]
        if lat:
            min_span_for = {b: (args.min_span if g[4] else
                                max(args.min_span, max(lat)))
                            for b, g in geom.items()}
        else:
            min_span_for = {b: args.min_span for b in geom}
    else:
        min_span_for = {b: args.min_span for b in geom}

    for j, bag in enumerate(bags):
        axp, axe = axes[0, j], axes[-1, j]
        here = [c for c in ckpts if (c, bag) in runs]
        d0 = runs[(here[0], bag)]
        ref, yaws, lo_i, hi_i, _ = geom[bag]

        if args.arcs:
            pt = np.asarray(runs[(here[0], bag)]["pred_traj"])
            for i in range(lo_i, hi_i, args.arcs):
                w = to_world(np.vstack([[0, 0], pt[i]]), ref[i], yaws[i])
                axp.plot(w[:, 1], w[:, 0], color=color[here[0]], lw=0.5,
                         alpha=0.30, zorder=2)

        xs_all, ys_all = [ref[lo_i:hi_i, 0]], [ref[lo_i:hi_i, 1]]
        axp.plot(ref[lo_i:hi_i, 1], ref[lo_i:hi_i, 0], color=GT,
                 ls=(0, (4, 2)), lw=1.3, zorder=3)
        for c in here:
            d = runs[(c, bag)]
            loc = endpoint_locus(np.asarray(d["pred_traj"]), ref, yaws)
            if args.smooth > 1:
                w = int(args.smooth) | 1
                k = np.ones(w) / w
                pad = w // 2
                loc = np.stack([np.convolve(np.pad(loc[:, q], pad, mode="edge"),
                                            k, mode="valid")[:len(loc)]
                                for q in (0, 1)], axis=1)
            seg = loc[lo_i:hi_i]
            axp.plot(seg[:, 1], seg[:, 0], color=color[c], lw=1.1, zorder=4)
            xs_all.append(seg[:, 0])
            ys_all.append(seg[:, 1])

        Y = np.concatenate(ys_all)
        X = np.concatenate(xs_all)
        cy = 0.5 * (Y.max() + Y.min())
        hy = max(0.5 * (Y.max() - Y.min()) * 1.15,
                 min_span_for.get(bag, args.min_span) / 2)
        if len(here) > 1:
            lat = np.vstack([
                (np.asarray(runs[(c, bag)]["pred_lat"])[lo_i:hi_i]) * 100.0
                for c in here])
            sep = float((lat.max(axis=0) - lat.min(axis=0)).max())
            print(f"  {bag}: max checkpoint separation {sep:.1f} cm on a "
                  f"{hy * 200:.0f} cm wide panel = {sep / (hy * 200) * 100:.1f}%"
                  " of panel width"
                  + ("  <- visible" if sep / (hy * 200) > 0.03
                     else "  <- too small to see; the rows below carry it"))
        axp.set_xlim(cy + hy, cy - hy)            # +y (left) on the left
        axp.set_ylim(X.min() - 0.15, X.max() + 0.15)
        axp.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        if args.equal_aspect:
            axp.set_aspect("equal", adjustable="box")
        elif args.stretch and args.stretch != 1.0:
            axp.set_aspect(1.0 / args.stretch, adjustable="box")
        extra = dict(x.split("=", 1) for x in args.extra_path)
        if bag in extra:
            pp = [q.strip() for q in extra[bag].split(",")]
            ep = load_extra_path(pp[0])
            axp.plot(ep[:, 1], ep[:, 0], color=args.extra_color,
                     ls=(0, (1, 1.6)), lw=1.2, zorder=5)
        obst = dict(x.split("=", 1) for x in args.obstacle)
        if bag in obst:
            parts = [p.strip() for p in obst[bag].split(",")]
            ox, oy = float(parts[0]), float(parts[1])
            lab = parts[2] if len(parts) > 2 else None
            draw_bin(axp, ox, oy, label=lab)
            ylo, yhi = sorted(axp.get_xlim())
            xlo, xhi = axp.get_ylim()
            if not (ylo <= oy <= yhi and xlo <= ox <= xhi):
                print(f"  [warn] {bag}: bin at x={ox}, y={oy} is outside the "
                      f"visible window x[{xlo:.1f},{xhi:.1f}] "
                      f"y[{ylo:.1f},{yhi:.1f}] -- it will not show")
        print(f"  {bag}: panel shows x[{axp.get_ylim()[0]:.1f},"
              f"{axp.get_ylim()[1]:.1f}] m forward, "
              f"y[{min(axp.get_xlim()):.2f},{max(axp.get_xlim()):.2f}] m lateral")
        axp.set_title(titles.get(bag, bag), pad=3)
        axp.set_xlabel("y [m]", labelpad=1)
        axp.grid(alpha=0.25, lw=0.4)
        axp.set_axisbelow(True)

        if args.deviation:
            axd = axes[1, j]
            step = np.linalg.norm(np.diff(ref, axis=0), axis=1)
            sdist = np.concatenate([[0.0], np.cumsum(step)])
            for c in here:
                d = runs[(c, bag)]
                # lateral offset in the robot frame at each instant: the
                # anchor pose cancels, so this is model error only
                dev = (np.asarray(d["pred_traj"])[:, :, 1]
                       - np.asarray(d["actual_traj"])[:, :, 1]) * 100.0
                sd = sdist[:len(dev)]
                if len(here) == 1:      # band = spread across the 8 waypoints
                    axd.fill_between(sd, dev.min(axis=1), dev.max(axis=1),
                                     color=color[c], alpha=0.20, lw=0)
                axd.plot(sd, dev.mean(axis=1), color=color[c], lw=1.0)
            axd.axhline(0, color=GT, ls=(0, (4, 2)), lw=1.1)
            axd.set_xlabel("distance along path [m]", labelpad=1)
            axd.set_xlim(0, sdist[min(len(sdist), hi_i) - 1])
            axd.grid(alpha=0.25, lw=0.4)
            axd.set_axisbelow(True)
            if j == 0:
                axd.set_ylabel("lateral offset\nfrom reference [cm]")

        ph = phases(d0, args.turn_threshold)
        data, cols, centres, names = [], [], [], []
        group = 0.72 * args.box_width / 0.30
        w = group / max(1, len(here))
        for p, (pname, mask) in enumerate(ph):
            for k, c in enumerate(here):
                d = runs[(c, bag)]
                e = np.abs(np.asarray(d["pred_lat"])
                           - np.asarray(d["actual_lat"])) * 100.0
                data.append(e[mask])
                cols.append(color[c])
                centres.append(p + k * w - group / 2 + w / 2)
            names.append(pname)
        bp = axe.boxplot(
            data, positions=centres, widths=w * 0.62, patch_artist=True,
            showmeans=True, showfliers=False, whis=(5, 95),
            medianprops=dict(color="black", lw=0.8),
            meanprops=dict(marker="D", markersize=2.4,
                           markerfacecolor="#EEEEEE", markeredgecolor="black",
                           markeredgewidth=0.4),
            boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5),
            capprops=dict(lw=0.5))
        for patch, fc in zip(bp["boxes"], cols):
            patch.set_facecolor(fc)
            patch.set_alpha(0.45)
            patch.set_edgecolor(fc)
        pn = dict(x.split("=", 1) for x in args.phase_names)
        if bag in pn:
            custom = [t.strip() for t in pn[bag].split("/")]
            names = [custom[i] if i < len(custom) else names[i]
                     for i in range(len(names))]
        axe.set_xticks(np.arange(len(names)))
        axe.set_xticklabels(names)
        axe.set_xlim(-0.6, len(names) - 0.4)
        axe.grid(axis="y", alpha=0.25, lw=0.4)
        axe.set_axisbelow(True)
        if j == 0:
            axp.set_ylabel("x [m]")
            axe.set_ylabel("Lateral\nError [cm]")
        for c in here:
            d = runs[(c, bag)]
            e = np.abs(np.asarray(d["pred_lat"])
                       - np.asarray(d["actual_lat"])) * 100.0
            print(f"  {bag} {c:>9s}: " + ", ".join(
                f"{p} median {np.median(e[m]):.2f} cm" for p, m in ph))

    emax = {}
    if args.err_max and str(args.err_max).lower() != "auto":
        spec = str(args.err_max)
        if "=" not in spec:
            emax = {b: float(spec) for b in bags}
        else:
            kv = dict(p.split("=", 1) for p in spec.split(",") if "=" in p)
            for b in bags:
                key = ("turn" if len(phases(runs[(
                    [c for c in ckpts if (c, b) in runs][0], b)],
                    args.turn_threshold)) > 1
                    else "straight")
                if b in kv:
                    emax[b] = float(kv[b])
                elif key in kv:
                    emax[b] = float(kv[key])
    rows = [1, -1] if args.deviation else [-1]
    for r in rows:
        if r == -1 and emax:
            for j, b in enumerate(bags):
                if b in emax:
                    axes[r, j].set_ylim(0, emax[b])
            continue
        lo = min(axes[r, j].get_ylim()[0] for j in range(n))
        hi = max(axes[r, j].get_ylim()[1] for j in range(n))
        for j in range(n):
            axes[r, j].set_ylim(lo, hi)

    handles = [Line2D([], [], color=GT, ls=(0, (4, 2)), lw=1.3,
                      label="Reference"
                      + ("" if odom else " (reconstructed)"))]
    handles += [Line2D([], [], color=color[c], lw=1.1, label=c) for c in ckpts]
    for spec in args.extra_path:
        pp = spec.split("=", 1)[1].split(",")
        handles.append(Line2D([], [], color=args.extra_color,
                              ls=(0, (1, 1.6)), lw=1.2,
                              label=pp[1].strip() if len(pp) > 1
                              else "other run (odometry)"))
    fig.legend(handles=handles, loc="upper center", ncol=min(5, len(handles)),
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handlelength=1.9)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.11, right=0.99)

    for ext in ("pdf", "png"):
        p = f"{args.out}.{ext}"
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()