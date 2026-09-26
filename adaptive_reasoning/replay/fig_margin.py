"""
fig_margin.py — Fig. "ar-margin" (E4): how much time the evidence gives us.

For every directional approach, margin = (junction_frame − flip_frame) / fps:
the seconds between "a human could read the sign" and "the decision must be
executed".  If nearly all margins exceed the measured VLM latency L, the gate
of Eq. (gate) is anticipatory by construction and no deadline term is needed —
this figure IS that argument.

Usage:
  python -m adaptive_reasoning.replay.fig_margin ar_extra.csv --out figs/ar_margin
  # once E3 has measured latency:
  python -m adaptive_reasoning.replay.fig_margin ar_extra.csv --out figs/ar_margin --latency 2.8 --latency-sd 0.4

Until --latency is passed, the line is drawn at a PLACEHOLDER value and
labelled as such, so the draft figure can exist today and become final the
moment E3 lands.
"""
from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from . import figstyle
from .figstyle import C, COL_W


def load_margins(csv_path: str) -> tuple[np.ndarray, list[str]]:
    rows = [r for r in csv.DictReader(open(csv_path))
            if r["sign_situation"] == "directional" and r["decision_grounded"] == "1"]
    m = np.array([(int(r["junction_frame"]) - int(r["flip_frame"])) / int(r["fps"]) for r in rows])
    return m, [r["bag"] for r in rows]


def make(csv_path: str, out_stem: str, latency: float | None, latency_sd: float | None) -> None:
    figstyle.use()
    margins, bags = load_margins(csv_path)
    med = float(np.median(margins))

    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    bins = np.arange(0, np.ceil(margins.max()) + 2, 1.0)
    ax.hist(margins, bins=bins, color=C["ours"], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axvline(med, color=C["ours"], ls="--", lw=0.9)
    ax.text(med + 0.15, ax.get_ylim()[1] * 0.97, f"median {med:.1f}\u2009s",
            color=C["ours"], ha="left", va="top")

    L = latency if latency is not None else 3.0
    label = f"VLM latency $L$={L:.1f}\u2009s" if latency is not None \
        else "$L$=3.0\u2009s (PLACEHOLDER — E3)"
    if latency_sd:
        ax.axvspan(L - latency_sd, L + latency_sd, color=C["accent"], alpha=0.15, lw=0)
    ax.axvline(L, color=C["accent"], lw=1.1)
    ax.text(L + 0.15, ax.get_ylim()[1] * 0.72, label, color=C["accent"], ha="left", va="top")

    frac = float((margins > L).mean())
    ax.set_xlabel("legibility-onset margin  $(t_{junction}-t_{flip})$  [s]")
    ax.set_ylabel("approaches")
    ax.set_title(f"{len(margins)} approaches · {frac:.0%} with margin $> L$", loc="left")
    ax.set_xlim(0, bins[-1])
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    figstyle.save(fig, out_stem, arrays={"margins_s": margins,
                                         "bags": np.array(bags),
                                         "latency": np.array([L])})
    print(f"wrote {out_stem}.pdf/.png  (n={len(margins)}, median={med:.2f}s, "
          f"min={margins.min():.2f}s, frac>L={frac:.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="figs/ar_margin")
    ap.add_argument("--latency", type=float, default=None, help="measured VLM latency L in seconds (E3)")
    ap.add_argument("--latency-sd", type=float, default=None)
    a = ap.parse_args()
    make(a.csv, a.out, a.latency, a.latency_sd)
