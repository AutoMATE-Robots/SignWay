#!/usr/bin/env python
"""Animated held-out timing plot (pred vs actual wp8 lateral) on pure black,
for the ICRA supplementary video's evaluation panel.

REAL DATA — point it at what the eval already saved (no rerun):
    python animate_timing.py --npy eval_v6_e1/timing.npy --decision turn_right \
        --flip-step 95 --turn-done-step 261 --out anim_e1
Accepted shapes: .npy (N,3)=step,actual,pred or (N,2)=actual,pred (steps assumed
0..N-1); transposed arrays auto-fixed; .npz or pickled dict (keys matched:
step/frame, actual/gt, pred); or two files: --npy actual.npy pred.npy.
If columns are swapped use --order pred,actual. The loader PRINTS what it
inferred (n, finals, extremes) — check those against the run before using.
(e1 raw frames 190/522 at stride 2 -> steps 95/261; e2: 95/221, turn_left.)

PREVIEW (no data; watermarked SYNTHETIC on every frame):
    python animate_timing.py --preview --decision turn_right --out preview

Outputs <out>/f_0000.png ... + <out>.mp4 (and .gif with --gif).
Pure #000000 background so it composites invisibly on the video.
"""
import argparse, subprocess, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PURPLE, GRY, TEAL, DIM, WHT = "#C68CFF", "#A8ADB4", "#3CC8B4", "#4A4F55", "#FFFFFF"
FONT = "DejaVu Sans"

def synth(decision):
    steps = np.arange(0, 265)
    on = 218
    sgn = -1.0 if decision == "turn_right" else 1.0
    rng = np.random.default_rng(3)
    actual = sgn * 0.105 / (1 + np.exp(-(steps - on) / 8.0))
    actual += np.convolve(rng.normal(0, 0.0035, steps.size), np.ones(9) / 9, "same")
    pred = sgn * 0.075 / (1 + np.exp(-(steps - on - 4) / 9.0))
    pred += np.convolve(rng.normal(0, 0.005, steps.size), np.ones(9) / 9, "same")
    return steps.astype(float), actual, pred

def _pick(d, names):
    for k in d:
        if str(k).lower() in names:
            return np.asarray(d[k]).ravel()
    return None

def load_npy(paths, order):
    if len(paths) == 2:
        a, p = (np.load(x).ravel() for x in paths)
        steps = np.arange(a.size, dtype=float)
        return steps, a, p, f"two files: actual={paths[0]}, pred={paths[1]}"
    raw = np.load(paths[0], allow_pickle=True)
    if hasattr(raw, "files"):  # npz
        raw = {k: raw[k] for k in raw.files}
    elif isinstance(raw, np.ndarray) and raw.dtype == object and raw.ndim == 0:
        raw = raw.item()  # pickled dict
    if isinstance(raw, dict):
        steps = _pick(raw, {"step", "steps", "frame", "frames", "t", "time"})
        a = _pick(raw, {"actual", "gt", "ground_truth", "actual_lat", "gt_lat", "y_true"})
        p = _pick(raw, {"pred", "prediction", "pred_lat", "y_pred"})
        if a is None or p is None:
            raise SystemExit(f"couldn't find actual/pred keys in {sorted(raw)} — "
                             "tell me the key names and I'll map them")
        if steps is None:
            steps = np.arange(a.size, dtype=float)
        return steps, a, p, f"dict keys of {paths[0]}"
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2:
        raise SystemExit(f"expected 2-D array, got shape {arr.shape}")
    if arr.shape[0] in (2, 3) and arr.shape[1] > 3:
        arr = arr.T
    if arr.shape[1] == 3:
        steps, a, p = arr[:, 0], arr[:, 1], arr[:, 2]
        how = "(N,3) = step, actual, pred"
    elif arr.shape[1] == 2:
        steps, a, p = np.arange(arr.shape[0], dtype=float), arr[:, 0], arr[:, 1]
        how = "(N,2) = actual, pred; steps = 0..N-1"
    else:
        raise SystemExit(f"can't interpret shape {arr.shape}")
    if order == "pred,actual":
        a, p = p, a
        how += " [swapped by --order]"
    return steps, a, p, how

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="columns: step,actual,pred (header ok)")
    ap.add_argument("--npy", nargs="+",
                    help=".npy/.npz (or two files: actual.npy pred.npy)")
    ap.add_argument("--order", default="actual,pred",
                    choices=["actual,pred", "pred,actual"],
                    help="column meaning for 2-column arrays")
    ap.add_argument("--preview", action="store_true", help="synthetic data, watermarked")
    ap.add_argument("--decision", default="turn_right",
                    choices=["turn_left", "turn_right", "straight"])
    ap.add_argument("--flip-step", type=float, default=95)
    ap.add_argument("--turn-done-step", type=float, default=261)
    ap.add_argument("--hz", type=float, default=10.0, help="dataset rate (steps/s)")
    ap.add_argument("--duration", type=float, default=6.0, help="sweep seconds")
    ap.add_argument("--hold", type=float, default=2.0, help="hold seconds at end")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="timing_anim")
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()

    if a.preview:
        steps, actual, pred = synth(a.decision)
    elif a.npy:
        steps, actual, pred, how = load_npy(a.npy, a.order)
        print(f"loaded {how}: n={steps.size}, "
              f"final pred {pred[-1]:+.3f} / actual {actual[-1]:+.3f}, "
              f"extreme pred {pred[np.argmax(np.abs(pred))]:+.3f} / "
              f"actual {actual[np.argmax(np.abs(actual))]:+.3f} — "
              "verify against the run before using")
    else:
        if not a.csv:
            ap.error("--npy or --csv required (or use --preview)")
        d = np.genfromtxt(a.csv, delimiter=",", names=True)
        steps, actual, pred = d["step"], d["actual"], d["pred"]

    t = steps / a.hz
    flip_t, done_t = a.flip_step / a.hz, a.turn_done_step / a.hz
    W, Hh = (int(v) for v in a.size.split("x"))
    W -= W % 2; Hh -= Hh % 2
    ymax = max(0.02, 1.25 * np.max(np.abs(np.concatenate([actual, pred]))))
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    n_sweep = int(a.duration * a.fps)
    n_hold = int(a.hold * a.fps)
    N = len(t)

    for k in range(n_sweep + n_hold):
        frac = min(1.0, (k + 1) / n_sweep)
        idx = max(2, int(frac * N))
        done = k >= n_sweep - 1

        fig = plt.figure(figsize=(W / 100, Hh / 100), dpi=100)
        fig.patch.set_facecolor("black")
        ax = fig.add_axes([0.085, 0.15, 0.875, 0.72])
        ax.set_facecolor("black")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(DIM); ax.spines[s].set_linewidth(1.6)
        ax.tick_params(colors=GRY, labelsize=15)
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(-ymax, ymax)
        ax.set_xlabel("time (s)", color=GRY, fontsize=17, family=FONT)
        ax.set_ylabel("wp8 lateral (m)", color=GRY, fontsize=17, family=FONT)
        ax.axhline(0, color=DIM, lw=1.2, ls=(0, (4, 4)))

        # standing-prompt window (known schedule, shown from frame 0)
        if a.decision != "straight":
            ax.axvspan(flip_t, done_t, color=TEAL, alpha=0.08)
            ax.text(flip_t + 0.3, ymax * 0.88, f"standing prompt: {a.decision}",
                    color=TEAL, fontsize=16, fontweight="bold", family=FONT)
            ax.text(flip_t + 0.3, ymax * 0.74,
                    "model must hold straight until the junction is visible",
                    color=GRY, fontsize=13, family=FONT)

        ax.plot(t[:idx], actual[:idx], color=GRY, lw=3.2, solid_capstyle="round")
        ax.plot(t[:idx], pred[:idx], color=PURPLE, lw=3.2, solid_capstyle="round")
        if not done:
            ax.plot([t[idx - 1]], [actual[idx - 1]], "o", ms=7, color=GRY)
            ax.plot([t[idx - 1]], [pred[idx - 1]], "o", ms=7, color=PURPLE)
        else:
            xr = t[-1] - 0.005 * (t[-1] - t[0])
            up = pred[-1] >= actual[-1]
            ax.text(xr, pred[-1] + (0.03 if up else -0.03) * ymax, "pred",
                    color=PURPLE, fontsize=16, fontweight="bold", ha="right",
                    va="bottom" if up else "top", family=FONT)
            ax.text(xr, actual[-1] + (-0.03 if up else 0.03) * ymax, "actual",
                    color=GRY, fontsize=16, fontweight="bold", ha="right",
                    va="top" if up else "bottom", family=FONT)
            fig.text(0.955, 0.055,
                     f"final: pred {pred[-1]:+.3f} m   actual {actual[-1]:+.3f} m",
                     color=GRY, fontsize=14, ha="right", family=FONT)

        # sign convention note
        fig.text(0.088, 0.055, "y: + = left turn, \u2212 = right turn",
                 color=DIM, fontsize=13, family=FONT)
        if a.title:
            fig.text(0.088, 0.925, a.title, color=WHT, fontsize=21,
                     fontweight="bold", family=FONT)
        if a.preview:
            fig.text(0.5, 0.5, "PREVIEW \u2014 SYNTHETIC DATA", color="#FF4B33",
                     fontsize=44, fontweight="bold", ha="center", va="center",
                     alpha=0.22, rotation=18, family=FONT)
            fig.text(0.955, 0.925, "synthetic data \u2014 not a run", color="#FF4B33",
                     fontsize=13, ha="right", family=FONT)

        fig.savefig(outdir / f"f_{k:04d}.png", facecolor="black", dpi=100)
        plt.close(fig)

    if shutil.which("ffmpeg"):
        mp4 = f"{a.out}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                        "-i", str(outdir / "f_%04d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", "-map_metadata", "-1", mp4],
                       check=True)
        print("wrote", mp4)
        if a.gif:
            pal = str(outdir / "pal.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
                            "-vf", "fps=20,scale=960:-1:flags=lanczos,palettegen", pal],
                           check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i", pal,
                            "-lavfi", "fps=20,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse",
                            f"{a.out}.gif"], check=True)
            print("wrote", f"{a.out}.gif")
    else:
        print(f"frames in {outdir}/ ; assemble with:\n"
              f"ffmpeg -framerate {a.fps} -i {outdir}/f_%04d.png -c:v libx264 "
              f"-pix_fmt yuv420p -crf 18 {a.out}.mp4")

if __name__ == "__main__":
    main()