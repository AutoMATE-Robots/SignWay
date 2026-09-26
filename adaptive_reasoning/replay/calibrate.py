"""
calibrate.py — E2/T6.  Turn VLM outcomes into the legibility weights w.

Sample:  from every train-split dump, take every `--every`-th frame on which a
plate is visible (ℓ trajectory low→high, so samples span the whole range),
build the same crop payload the live gate would send, ask the reasoner
(cached), and label

    y = 1  iff the answer is applicable AND direction == annotated decision.

Fit:  logistic regression on the standardized 6-feature φ, plain numpy
Newton/IRLS with L2 (--l2).  Saved to calib/w.json via Legibility.save with
fit metadata; legibility.py refuses to load it if the feature list drifts.

Eval:  reliability diagram + AUC/ECE on the held-out split (by building if two
exist, else by bag hash), plus |w| bars — Fig calib.

Smoke mode (--synthetic) fits on generated data so the whole path (fit → save
→ load → figure) is exercised before any money is spent.

Usage on MSI:
  python -m adaptive_reasoning.replay.calibrate --dumps $SCRATCH/ar_replay \
      --csv annotations/ar_extra.csv --every 5 --model gemini-3.6-flash \
      --out adaptive_reasoning/calib --fig figs/ar_calib
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..evidence.features import FEATURES
from ..evidence.legibility import Legibility
from . import figstyle
from .figstyle import C, COL_W


# ----------------------------------------------------------------------------
# Logistic fit (numpy IRLS)
# ----------------------------------------------------------------------------

def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2,
                 iters: int = 50) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Return (w, b, mean, std) with X standardized internally."""
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-9, 1.0, std)
    Z = np.hstack([(X - mean) / std, np.ones((len(X), 1))])
    beta = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ beta))
        W = np.clip(p * (1 - p), 1e-6, None)
        reg = l2 * np.eye(Z.shape[1]); reg[-1, -1] = 0.0
        H = Z.T @ (Z * W[:, None]) + reg
        g = Z.T @ (y - p) - reg @ beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta[:-1], float(beta[-1]), mean, std


def auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else 0.5


def ece(y: np.ndarray, p: np.ndarray, bins: int = 8) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] + (i == bins - 1))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


# ----------------------------------------------------------------------------
# Sample collection
# ----------------------------------------------------------------------------

def collect_samples(dumps: Path, csv_path: str, every: int,
                    reasoner=None, crops_available: bool = False):
    """(X, y, bag, building) per sample.  With no reasoner (or no stored crops
    yet), labels come from the structural oracle: y=1 iff R_struct==1 at that
    frame and completeness==1 — a stand-in clearly marked in the fit metadata;
    replaced by real VLM outcomes when E2 runs with --model."""
    X, y, bags, bldgs = [], [], [], []
    rows = {r["bag"]: r for r in csv.DictReader(open(csv_path))}
    for f in sorted(dumps.glob("*.npz")):
        row = rows.get(f.stem)
        if row is None or row["sign_situation"] != "directional":
            continue
        d = np.load(f, allow_pickle=True)
        i = 0
        while f"plate{i}_id" in d:
            R, Rs, ell = d[f"plate{i}_R"], d[f"plate{i}_R_struct"], d[f"plate{i}_ell"]
            PHI = d[f"plate{i}_phi"]
            vis = np.where((ell > 0) | (R > 0))[0][::every]
            for t in vis:
                X.append(PHI[t])
                if reasoner is not None and crops_available:
                    raise NotImplementedError("live labeling wiring lands with crop storage")
                y.append(1.0 if (Rs[t] >= 1.0 and PHI[t][FEATURES.index("completeness")] >= 1.0)
                         else 0.0)
                bags.append(f.stem); bldgs.append(row["building"])
            i += 1
    return np.array(X), np.array(y), np.array(bags), np.array(bldgs)


def split(bags: np.ndarray, bldgs: np.ndarray):
    """Held-out by building when ≥2 buildings; else by bag-name hash (~25%)."""
    if len(set(bldgs.tolist())) >= 2:
        held = min(set(bldgs.tolist()), key=lambda b: (bldgs == b).sum())
        return bldgs != held, f"building={held}"
    test = np.array([hash(b) % 4 == 0 for b in bags])
    return ~test, "bag-hash 25%"


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def fig_calib(y_te, p_te, w, out_stem):
    figstyle.use()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.7),
                                 gridspec_kw={"wspace": 0.5})
    edges = np.linspace(0, 1, 9); mids = (edges[:-1] + edges[1:]) / 2
    obs = [y_te[(p_te >= lo) & (p_te < hi + (hi == 1))].mean()
           if ((p_te >= lo) & (p_te < hi + (hi == 1))).any() else np.nan
           for lo, hi in zip(edges[:-1], edges[1:])]
    a1.plot([0, 1], [0, 1], color=C["baseline2"], lw=0.7, ls="--")
    a1.plot(mids, obs, "-o", ms=2.5, color=C["ours"])
    a1.set_xlabel(r"predicted $\ell$"); a1.set_ylabel("empirical success")
    a1.set_title(f"AUC {auc(y_te, p_te):.2f} · ECE {ece(y_te, p_te):.2f}", loc="left")
    order = np.argsort(-np.abs(w))
    a2.barh(range(len(w)), np.abs(w)[order], color=C["ours"], height=0.65)
    a2.set_yticks(range(len(w)))
    a2.set_yticklabels([FEATURES[i].replace("_", " ") for i in order])
    a2.invert_yaxis(); a2.set_xlabel(r"$|w_i|$ (standardized)")
    figstyle.save(fig, out_stem, arrays={"y_te": y_te, "p_te": p_te, "w": w})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps"); ap.add_argument("--csv")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--model", default=None, help="VLM for labeling (E2); omit for structural-oracle labels")
    ap.add_argument("--out", default="adaptive_reasoning/calib")
    ap.add_argument("--fig", default="figs/ar_calib")
    ap.add_argument("--synthetic", action="store_true", help="smoke-test the fit path")
    a = ap.parse_args()

    if a.synthetic:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, len(FEATURES)))
        true_w = np.array([1.5, 0.8, 0.6, 1.2, 0.3, 0.5])
        y = (rng.random(600) < 1 / (1 + np.exp(-(X @ true_w - 0.5)))).astype(float)
        bags = np.array([f"bag{i%12}" for i in range(600)])
        bldgs = np.array(["keller"] * 600)
        label_mode = "synthetic"
    else:
        X, y, bags, bldgs = collect_samples(Path(a.dumps), a.csv, a.every)
        label_mode = "structural-oracle" if a.model is None else a.model
        if a.model is not None:
            raise SystemExit("VLM labeling needs stored crops — wire after first E1 run "
                             "(dump crops with --save-crops), then rerun with --model")
    tr, held_desc = split(bags, bldgs)
    w, b, mean, std = fit_logistic(X[tr], y[tr], l2=a.l2)
    leg = Legibility(mean, std, w, b)
    p_te = np.array([leg(x) for x in X[~tr]])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    leg.save(out / "w.json", meta={"labels": label_mode, "n_train": int(tr.sum()),
                                   "n_test": int((~tr).sum()), "held_out": held_desc,
                                   "l2": a.l2,
                                   "auc_test": auc(y[~tr], p_te), "ece_test": ece(y[~tr], p_te)})
    fig_calib(y[~tr], p_te, w, a.fig)
    print(f"fit on {tr.sum()} samples (labels: {label_mode}), held out {(~tr).sum()} ({held_desc})")
    print(f"AUC {auc(y[~tr], p_te):.3f}  ECE {ece(y[~tr], p_te):.3f}")
    print(f"wrote {out}/w.json and {a.fig}.pdf — legibility.py now loads CALIBRATED weights"
          if label_mode != "synthetic" and label_mode != "structural-oracle" else
          f"wrote {out}/w.json (labels={label_mode} — NOT paper-grade) and {a.fig}.pdf")


if __name__ == "__main__":
    main()
