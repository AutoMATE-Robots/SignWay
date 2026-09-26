"""
figstyle.py — one visual language for every SignWay figure.

Rules encoded here (so no figure can drift):
  * Figures are designed AT FINAL SIZE: COL_W = 3.4 in (IEEE column),
    PAGE_W = 7.06 in (full width).  LaTeX must never scale them.
  * 7–8 pt fonts everywhere.
  * One palette: OURS teal, BASELINE greys, and the four gate-state colors
    used identically in paper figures and the video HUD.
  * save() writes PDF (vector, for the paper) + PNG (300 dpi, for slides/video)
    AND np.savez of the raw arrays next to it, so every figure can be
    re-styled without re-running any experiment.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

COL_W = 3.4          # inches, IEEE two-column single column
PAGE_W = 7.06        # inches, full width

C = {
    "ours":     "#0f766e",   # teal
    "ours_lt":  "#5eead4",
    "baseline": "#6b7280",   # grey
    "baseline2":"#9ca3af",
    "accent":   "#b45309",   # amber-brown (junction / latency markers)
    "bad":      "#b91c1c",
    # gate states — MUST match the video HUD
    "NO_SIGN":  "#d1d5db",
    "ARMED":    "#facc15",
    "PENDING":  "#fb923c",
    "DECIDED":  "#4ade80",
    # per-plate colors for timelines
    "plate0":   "#0f766e",
    "plate1":   "#7c3aed",
    "plate2":   "#db2777",
}

RC = {
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType (IEEE requirement)
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
}


def use() -> None:
    matplotlib.rcParams.update(RC)


def save(fig, out_stem: str | Path, arrays: dict[str, np.ndarray] | None = None) -> None:
    """Write <stem>.pdf + <stem>.png (+ <stem>_data.npz with the raw arrays)."""
    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.01)
    if arrays:
        np.savez(str(out) + "_data.npz", **arrays)
    plt.close(fig)
