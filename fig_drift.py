#!/usr/bin/env python3
"""SignWay paper figure: integrated VLA path vs human odometry, and how far it
drifts apart over time.

Each frame the policy sees one real image plus the standing prompt and predicts
8 waypoints. Chaining the first step of every prediction integrates the policy's
motion into a path -- the same operation odometry performs on the real robot --
so the two paths are directly comparable and the error ACCUMULATES rather than
being reset each frame.

IMPORTANT, state this in the caption: this is integrated OPEN-LOOP prediction,
not a closed-loop rollout. The images still come from the human's drive, so the
policy is never fed the consequences of its own drift. It shows accumulated
prediction error, which is strictly easier than closed-loop control.

Layout, one column per condition (turns merged; left/right not separated):
  Row 1  world-frame path: odometry (black dashed) vs integrated VLA (colour).
  Row 2  cross-track drift in metres vs frame -- perpendicular distance from
         the integrated VLA position to the odometry path.

Usage:
  python fig_drift.py --sweep $S --list
  python fig_drift.py --sweep $S --bags "e1,e2,e3" --out figures/fig_drift
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
VLA = "#C8102E"
GT = "#000000"


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
    return (re.sub(r"\d+", "", label), [int(x) for x in re.findall(r"\d+", label)])


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
    """Heading change over ONE step, from the predicted waypoint fan.

    For roughly constant curvature the bearing to waypoint k is k*dyaw/2, so
    dyaw = 2*bearing_k / k. Averaging the first few k stabilises it against
    per-waypoint noise.

    (Using the wp1->wp2 chord instead overestimates dyaw by ~50%, which is what
    made an earlier version of this plot rotate past 90 degrees on a right-angle
    turn.)
    """
    est = []
    for k in range(1, min(kmax, len(wp)) + 1):
        x, y = wp[k - 1]
        if np.hypot(x, y) > 1e-4:
            est.append(2.0 * np.arctan2(y, x) / k)
    return float(np.mean(est)) if est else 0.0


def integrate(traj, x0=None, yaw0=0.0, n_steps=None):
    """Chain per-step predictions into a world path.

    traj: (N, H, 2). Returns (xy, yaw). Starts at x0 with heading yaw0, so the
    same routine serves whole-bag integration and bounded-horizon windows
    anchored on ground truth.
    """
    traj = np.asarray(traj, dtype=np.float64)
    n = len(traj) if n_steps is None else min(n_steps, len(traj))
    xy = np.zeros((n, 2))
    yaws = np.zeros(n)
    xy[0] = (0.0, 0.0) if x0 is None else x0
    yaw = float(yaw0)
    yaws[0] = yaw
    for i in range(n - 1):
        dx, dy = traj[i, 0]
        c, s = np.cos(yaw), np.sin(yaw)
        xy[i + 1] = xy[i] + [dx * c - dy * s, dx * s + dy * c]
        yaw += dyaw_from_wps(traj[i])
        yaws[i + 1] = yaw
    return xy, yaws


def lateral_error(xy, gt_xy, gt_yaw):
    """Time-aligned LATERAL offset: the predicted position expressed in the
    ground-truth body frame at the same instant, lateral component only.

    Keeps the longitudinal (speed) component out, per the lateral-first rule.
    """
    d = xy - gt_xy[:len(xy)]
    c, s = np.cos(-gt_yaw[:len(xy)]), np.sin(-gt_yaw[:len(xy)])
    return np.abs(s * d[:, 0] + c * d[:, 1])


def integrate_fixed_heading(traj, x0, yaws):
    """Chain predicted displacements but rotate by GROUND-TRUTH heading.

    Isolates displacement error from compounding heading error: whatever drift
    remains here is the policy getting the step wrong, not the reconstruction
    losing its bearing.
    """
    n = min(len(traj), len(yaws))
    xy = np.zeros((n, 2))
    xy[0] = x0
    for i in range(n - 1):
        dx, dy = traj[i, 0]
        c, s = np.cos(yaws[i]), np.sin(yaws[i])
        xy[i + 1] = xy[i] + [dx * c - dy * s, dx * s + dy * c]
    return xy


def windows(pred, gt_xy, gt_yaw, n_steps, stride, fixed_heading=False):
    """Bounded-horizon segments, each re-anchored on the ground-truth pose."""
    segs = []
    for a in range(0, max(1, len(pred) - n_steps), stride):
        if fixed_heading:
            xy = integrate_fixed_heading(pred[a:a + n_steps], gt_xy[a],
                                         gt_yaw[a:a + n_steps])
        else:
            xy, _ = integrate(pred[a:], x0=gt_xy[a], yaw0=gt_yaw[a],
                              n_steps=n_steps)
        segs.append((a, xy, lateral_error(xy, gt_xy[a:], gt_yaw[a:])))
    return segs


def is_right_turn(d):
    lat = np.asarray(d["actual_lat"])
    return lat[np.argmax(np.abs(lat))] < 0


def is_turn(d):
    return np.abs(np.asarray(d["actual_lat"])).max() * 100.0 >= 2.0


def draw_bin(ax, x, y, h=0.55, w=0.34, label=None):
    """Schematic dustbin at world (x forward, y lateral), metres."""
    body = np.array([[y - w * 0.36, x], [y + w * 0.36, x],
                     [y + w * 0.50, x + h], [y - w * 0.50, x + h]])
    ax.add_patch(Polygon(body, closed=True, facecolor="#9AA0A6",
                         edgecolor="#4D5156", lw=0.7, zorder=6))
    lid = np.array([[y - w * 0.58, x + h], [y + w * 0.58, x + h],
                    [y + w * 0.58, x + h * 1.10], [y - w * 0.58, x + h * 1.10]])
    ax.add_patch(Polygon(lid, closed=True, facecolor="#4D5156",
                         edgecolor="#4D5156", lw=0.7, zorder=7))
    if label:
        ax.annotate(label, xy=(y, x + h * 1.10), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6, color="#4D5156", zorder=8)


def load_extra_path(npz_path):
    """Odom path from dump_odom.py, re-anchored to start at the origin heading
    +x so it can be overlaid on another drive.

    NOTE: this is a DIFFERENT drive. Alignment is by start pose only -- there
    is no shared frame between two bags, so treat the overlay as context, not
    as a registered comparison.
    """
    z = np.load(npz_path)
    x, y, yaw = np.asarray(z["x"]), np.asarray(z["y"]), np.asarray(z["yaw"])
    c, s_ = np.cos(-yaw[0]), np.sin(-yaw[0])
    dx, dy = x - x[0], y - y[0]
    return np.stack([c * dx - s_ * dy, s_ * dx + c * dy], axis=1)


# ---------------------------------------------------------------- figure
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", default=[])
    ap.add_argument("--run", action="append", default=[],
                    help="LABEL:BAG:PATH/to/series.npy")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint label (default: the only one present)")
    ap.add_argument("--bags", default=None,
                    help="comma-separated bags to include")
    ap.add_argument("--drift-max", type=float, default=0.5,
                    help="y-max of the drift row, in metres")
    ap.add_argument("--horizon", type=float, default=5.0,
                    help="open-loop horizon in SECONDS before re-anchoring on "
                         "ground truth (0 = integrate the whole bag, which "
                         "diverges and is not what the deployed system does)")
    ap.add_argument("--stride", type=float, default=2.0,
                    help="seconds between window starts")
    ap.add_argument("--heading-gt", action="store_true",
                    help="rotate predicted displacements by ground-truth "
                         "heading (isolates displacement error from "
                         "compounding heading error)")
    ap.add_argument("--extra-path", action="append", default=[],
                    help='overlay another drive\'s odometry as context: '
                         '"LABEL=/path/to/odom.npz" from dump_odom.py. '
                         'Aligned by start pose only -- a different bag has no '
                         'shared frame, so label it as context in the caption.')
    ap.add_argument("--hide-vla", action="store_true",
                    help="omit the integrated prediction and the drift row, "
                         "leaving only odometry paths")
    ap.add_argument("--extra-color", default="#1F77B4",
                    help="colour for --extra-path lines")
    ap.add_argument("--obstacle", action="append", default=[],
                    help='"BAG=x,y[,label]" dustbin marker, metres')
    ap.add_argument("--no-mirror", action="store_true",
                    help="keep right turns unmirrored (default mirrors them "
                         "onto the left turns so 'turns' read as one condition)")
    ap.add_argument("--out", default="fig_drift")
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

    ckpt = args.ckpt or (ckpts[0] if len(ckpts) == 1 else None)
    if ckpt is None:
        ap.error(f"several checkpoints present, pick one with --ckpt: {ckpts}")
    if ckpt not in ckpts:
        ap.error(f"--ckpt not found: {ckpt}. Available: {ckpts}")
    if args.bags:
        want = [s.strip() for s in args.bags.split(",")]
        miss = [w for w in want if w not in bags]
        if miss:
            ap.error(f"--bags not found: {miss}. Available: {bags}")
        bags = want

    runs = OrderedDict((b, load_series(paths[(ckpt, b)]))
                       for b in bags if (ckpt, b) in paths)
    if not runs:
        ap.error("nothing to plot for that checkpoint")

    groups = OrderedDict()
    for b, d in runs.items():
        groups.setdefault("turn" if is_turn(d) else "straight", []).append(b)
    order = [g for g in ("turn", "straight") if g in groups]
    titles = {"turn": "Turns (held-out)", "straight": "Straight (held-out)"}

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    ncol = len(order)
    nrow = 1 if args.hide_vla else 2
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(COL2 * min(1.0, 0.42 + 0.29 * ncol),
                 3.4 if args.hide_vla else 4.6),
        gridspec_kw=({"wspace": 0.24} if args.hide_vla else
                     {"height_ratios": [2.4, 1.15], "hspace": 0.34,
                      "wspace": 0.24}))
    axes = np.atleast_2d(axes)
    if args.hide_vla:
        axes = axes.reshape(1, ncol)
    elif ncol == 1:
        axes = axes.reshape(2, 1)

    HZ = 10.0                                   # dataset rate
    n_steps = max(2, int(round(args.horizon * HZ))) if args.horizon > 0 else 0
    stride = max(1, int(round(args.stride * HZ)))
    summary = []
    for j, g in enumerate(order):
        axp = axes[0, j]
        axd = None if args.hide_vla else axes[1, j]
        curves = []
        if args.hide_vla:
            for b in groups[g]:
                d = runs[b]
                gt_xy, _gy = integrate(np.asarray(d["actual_traj"]))
                mir = (not args.no_mirror) and g == "turn" and is_right_turn(d)
                sgn = -1.0 if mir else 1.0
                axp.plot(sgn * gt_xy[:, 1], gt_xy[:, 0], color=GT,
                         ls=(0, (4, 2)), lw=1.3, zorder=3)
        for b in groups[g] if not args.hide_vla else []:
            d = runs[b]
            pred = np.asarray(d["pred_traj"])
            gt_xy, gt_yaw = integrate(np.asarray(d["actual_traj"]))
            mir = (not args.no_mirror) and g == "turn" and is_right_turn(d)
            sign = -1.0 if mir else 1.0
            axp.plot(sign * gt_xy[:, 1], gt_xy[:, 0], color=GT,
                     ls=(0, (4, 2)), lw=1.2, zorder=3)
            H = int(d.get("horizon", 8) or 8)
            if n_steps and not args.hide_vla:
                for a, xy, err in windows(pred, gt_xy, gt_yaw, n_steps,
                                          stride, args.heading_gt):
                    axp.plot(sign * xy[:, 1], xy[:, 0], color=VLA, lw=0.9,
                             alpha=0.75, solid_capstyle="round", zorder=4)
                    curves.append(err)
            else:
                xy, _ = integrate(pred)
                axp.plot(sign * xy[:, 1], xy[:, 0], color=VLA, lw=1.1,
                         alpha=0.9, zorder=4)
                curves.append(lateral_error(xy, gt_xy, gt_yaw))
            free = windows(pred, gt_xy, gt_yaw, n_steps, stride, False) \
                if n_steps else []
            lock = windows(pred, gt_xy, gt_yaw, n_steps, stride, True) \
                if n_steps else []
            if free and lock:
                k = min(n_steps, H) - 1
                print(f"  [{b}] at {H / HZ:.1f}s (one prediction): "
                      f"heading-free {np.mean([e[k] for _, _, e in free]):.3f} m, "
                      f"heading-locked {np.mean([e[k] for _, _, e in lock]):.3f} m")

        if n_steps and not args.hide_vla:
            free = []
            lock = []
            for b in groups[g]:
                d = runs[b]
                pr = np.asarray(d["pred_traj"])
                gxy, gyaw = integrate(np.asarray(d["actual_traj"]))
                free += [e for _, _, e in windows(pr, gxy, gyaw, n_steps,
                                                  stride, False)]
                lock += [e for _, _, e in windows(pr, gxy, gyaw, n_steps,
                                                  stride, True)]
            H = int(runs[groups[g][0]].get("horizon", 8) or 8)
            k = min(n_steps, H) - 1
            if free and lock:
                print(f"{g:>8s}: at {H / HZ:.1f}s (one prediction) "
                      f"heading-free {np.mean([e[k] for e in free]):.3f} m, "
                      f"heading-locked {np.mean([e[k] for e in lock]):.3f} m")
            tH = H / HZ
            if tH < n_steps / HZ:
                axd.axvline(tH, color="#555555", ls=(0, (2, 2)), lw=0.7)
                axd.annotate("prediction horizon", xy=(tH, 0.97),
                             xycoords=("data", "axes fraction"),
                             xytext=(3, -2), textcoords="offset points",
                             fontsize=6, color="#555555", ha="left", va="top")
            L = min(len(c) for c in curves)
            M = np.vstack([c[:L] for c in curves])
            t = np.arange(L) / HZ
            axd.fill_between(t, M.min(axis=0), M.max(axis=0), color=VLA,
                             alpha=0.18, lw=0)
            axd.plot(t, M.mean(axis=0), color=VLA, lw=1.3)
            axd.set_xlabel("open-loop horizon [s]", labelpad=1)
            axd.set_xlim(0, t[-1])
            summary.append((g, M.mean(axis=0)[-1], M.max()))
        elif curves and axd is not None:
            for c in curves:
                axd.plot(np.arange(len(c)) / HZ, c, color=VLA, lw=1.0)
            axd.set_xlabel("elapsed [s]", labelpad=1)
            summary.append((g, curves[0][-1], max(c.max() for c in curves)))

        xs = np.concatenate([ln.get_xdata() for ln in axp.lines])
        ys = np.concatenate([ln.get_ydata() for ln in axp.lines])
        cx = 0.5 * (xs.max() + xs.min())
        # floor the lateral span so a near-straight path doesn't collapse the
        # axis and collide its tick labels
        yspan = ys.max() - ys.min() + 1e-6
        hx = max(0.5 * (xs.max() - xs.min()) * 1.08, 0.18 * yspan)
        pady = 0.05 * yspan
        axp.set_xlim(cx + hx, cx - hx)                   # inverted: left = +y
        axp.set_ylim(ys.min() - pady, ys.max() + pady)
        axp.set_aspect("equal", adjustable="box")
        axp.xaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune=None))
        for k, spec in enumerate(args.extra_path):
            lab, pth = spec.split("=", 1)
            ep = load_extra_path(pth)
            axp.plot(ep[:, 1], ep[:, 0], color=args.extra_color,
                     ls="-" if args.hide_vla else (0, (1, 1.6)),
                     lw=1.3, zorder=5, label=lab)
            xs_e, ys_e = ep[:, 0], ep[:, 1]
        obst = dict(o.split("=", 1) for o in args.obstacle)
        for b in groups[g]:
            if b in obst:
                pp = [q.strip() for q in obst[b].split(",")]
                draw_bin(axp, float(pp[0]), float(pp[1]),
                         label=pp[2] if len(pp) > 2 else None)
        print(f"  {g}: panel x[{axp.get_ylim()[0]:.1f},{axp.get_ylim()[1]:.1f}] m"
              f", y[{min(axp.get_xlim()):.2f},{max(axp.get_xlim()):.2f}] m")
        axp.set_title(titles.get(g, g), pad=4)
        axp.set_xlabel("lateral  y [m]", labelpad=1)
        axp.grid(alpha=0.22, lw=0.4)
        axp.set_axisbelow(True)

        if axd is not None:
            axd.set_ylim(0, args.drift_max)
            axd.grid(alpha=0.22, lw=0.4)
            axd.set_axisbelow(True)
        if j == 0:
            axp.set_ylabel("forward  x [m]")
            if axd is not None:
                axd.set_ylabel("lateral drift [m]")

    lab = (f"{ckpt} ({args.horizon:g}s open-loop)" if n_steps
           else f"{ckpt} (integrated)")
    if args.heading_gt:
        lab += ", GT heading"
    handles = [Line2D([], [], color=GT, ls=(0, (4, 2)), lw=1.2,
                      label="Odometry (this bag)")]
    if not args.hide_vla:
        handles.append(Line2D([], [], color=VLA, lw=1.1, label=lab))
    for spec in args.extra_path:
        handles.append(Line2D([], [], color=args.extra_color,
                              ls="-" if args.hide_vla else (0, (1, 1.6)),
                              lw=1.3, label=spec.split("=", 1)[0]))
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.8, handlelength=2.0)
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.11, right=0.99)

    for ext in ("pdf", "png"):
        p = f"{args.out}.{ext}"
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    for g, endv, mx in summary:
        print(f"{g:>8s}: mean lateral drift at horizon end {endv:.3f} m, "
              f"worst {mx:.3f} m")


if __name__ == "__main__":
    main()