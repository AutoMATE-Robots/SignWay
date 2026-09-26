"""
timeline.py — Fig. "ar-timeline": one approach through the gate's eyes.

Rows (top to bottom):
  1. per-plate evidence R·ℓ over time + q_t (bold) + τ (dashed)
  2. gate state as a colour band (NO_SIGN / ARMED / PENDING / DECIDED)
  3. fire ticks: ours vs. the reactive necessity-only gate

The figure is produced by RUNNING THE REAL adaptive_reasoning.gate over the
per-frame evidence — not by drawing what we hope the gate does.  When E1's
feature dumps exist, feed `--npz ar_replay/<bag>.npz`; until then `--demo`
synthesizes a realistic approach (directory with rising legibility, an
irrelevant notice, a temporary sign consumed as not-applicable) so the visual
design is locked and reviewed now.

Usage:
  python -m adaptive_reasoning.replay.timeline --demo --tau 0.55 --out figs/ar_timeline_demo
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ..evidence.resolve import Resolution
from ..gate import EvidenceGate, GateConfig, GateState, PlateEvidence, State, VLMResponse, new_goal
from ..memory import Memory
from . import figstyle
from .figstyle import C, COL_W


# ----------------------------------------------------------------------------
# Evidence input (one approach)
# ----------------------------------------------------------------------------

@dataclass
class PlateTrack:
    plate_id: str
    label: str
    R: np.ndarray          # (T,)
    ell: np.ndarray        # (T,)
    resolvable: bool = False
    direction: str | None = None


def demo_tracks(T: int = 200, fps: float = 10.0, seed: int = 7):
    """A believable approach: directory legible late & rising; compost notice
    sharp but irrelevant; temporary sign mildly relevant (fires once, VLM says
    not-applicable).  Junction at 0.9·T."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    rise = lambda t0, k: 1 / (1 + np.exp(-(t - t0) / k))
    n = lambda s: rng.normal(0, s, T)

    directory = PlateTrack(
        "directory", "directory  6-201 to 6-250 →",
        R=np.clip(0.98 + n(0.01), 0, 1),
        ell=np.clip(rise(T * 0.45, T * 0.06) * 0.95 + n(0.03), 0, 1))
    notice = PlateTrack(
        "notice", "notice  COMPOSTO",
        R=np.clip(0.02 + n(0.01), 0, 1),
        ell=np.clip(0.85 + 0.1 * rise(T * 0.3, T * 0.1) + n(0.03), 0, 1))
    temp = PlateTrack(
        "temp", "temporary  A4 printout",
        R=np.clip(0.5 + n(0.02), 0, 1),
        ell=np.clip(rise(T * 0.25, T * 0.05) * 0.9 + n(0.03), 0, 1))
    # visibility windows
    directory.ell[: int(T * 0.25)] = 0
    temp.ell[: int(T * 0.10)] = 0
    temp.ell[int(T * 0.75):] = 0          # passes out of view
    notice.ell[: int(T * 0.15)] = 0
    junction = int(T * 0.9)
    vlm_latency_frames = int(1.2 * fps)
    return [temp, notice, directory], junction, vlm_latency_frames


def load_tracks(npz_path: str):
    """E1 output → PlateTracks.  Expected keys per plate i:
    plate{i}_id, plate{i}_label, plate{i}_R, plate{i}_ell, plate{i}_resolvable;
    plus 'junction_frame', 'fps', 'vlm_latency_frames'."""
    d = np.load(npz_path, allow_pickle=True)
    tracks, i = [], 0
    while f"plate{i}_id" in d:
        resv = np.asarray(d[f"plate{i}_resolvable"])
        tracks.append(PlateTrack(str(d[f"plate{i}_id"]), str(d[f"plate{i}_label"]),
                                 d[f"plate{i}_R"], d[f"plate{i}_ell"],
                                 bool(resv.any())))
        i += 1
    return tracks, int(d["junction_frame"]), int(d["vlm_latency_frames"])


# ----------------------------------------------------------------------------
# Run the REAL gate over the tracks
# ----------------------------------------------------------------------------

def run_gates(tracks, T, tau, vlm_latency_frames, temp_not_applicable=True):
    """Returns (q, states, our_fires, decided_at, reactive_fires)."""
    gate = EvidenceGate(GateConfig(tau=tau, fast_path=False))
    mem, s = Memory("demo-goal"), new_goal()
    q = np.zeros(T)
    states: list[str] = []
    fires: list[int] = []
    pending_return: tuple[int, VLMResponse] | None = None

    for t in range(T):
        if pending_return and t >= pending_return[0]:
            s, upds = gate.on_response(s, pending_return[1], t)
            for u in upds:
                mem.apply(u)
            pending_return = None
        plates = [PlateEvidence(tr.plate_id, float(tr.R[t]), float(tr.ell[t]))
                  for tr in tracks if tr.ell[t] > 0]
        r = gate.tick(s, plates, mem, t)
        for u in r.memory_updates:
            mem.apply(u)
        s, q[t] = r.state, r.q
        states.append(s.state.value)
        if r.fire:
            fires.append(t)
            best = r.best.plate_id
            resp = (VLMResponse(best, False) if (best == "temp" and temp_not_applicable)
                    else VLMResponse(best, True, "turn_right"))
            pending_return = (t + vlm_latency_frames, resp)

    # reactive necessity-only: fires on the first frame with any readable text,
    # then re-fires when new (unconsumed) text appears — τ→0 of Eq. (gate).
    reactive: list[int] = []
    seen: set[str] = set()
    decided_r = None
    for t in range(T):
        vis = {tr.plate_id for tr in tracks if tr.ell[t] > 0.05}
        new = vis - seen
        if new and (decided_r is None):
            reactive.append(t)
            seen |= new
            if "directory" in new:
                decided_r = t + vlm_latency_frames
    decided_at = next((t for t, st in enumerate(states) if st == "DECIDED"), None)
    return q, states, fires, decided_at, reactive


# ----------------------------------------------------------------------------
# Draw
# ----------------------------------------------------------------------------

STATE_ROW = {"NO_SIGN": C["NO_SIGN"], "ARMED": C["ARMED"],
             "PENDING": C["PENDING"], "DECIDED": C["DECIDED"]}


def make(tracks, junction, vlm_latency_frames, tau, fps, out_stem, demo=False):
    figstyle.use()
    T = len(tracks[0].R)
    q, states, fires, decided_at, reactive = run_gates(tracks, T, tau, vlm_latency_frames)
    tt = np.arange(T) / fps

    fig, (ax, axs, axf) = plt.subplots(
        3, 1, figsize=(COL_W, 2.5), sharex=True,
        gridspec_kw={"height_ratios": [5, 0.9, 1.4], "hspace": 0.12})

    # -- row 1: evidence curves (top-3 tracks colored, rest thin grey) --------
    order = sorted(range(len(tracks)), key=lambda i: -float(np.max(tracks[i].R * tracks[i].ell)))
    for i in order[3:]:
        ax.plot(tt, tracks[i].R * tracks[i].ell, color=C["baseline2"], lw=0.5, alpha=0.4)
    for rank, i in enumerate(order[:3]):
        ax.plot(tt, tracks[i].R * tracks[i].ell, color=C[f"plate{rank}"], lw=0.9,
                alpha=0.9, label=tracks[i].label[:28])
    ax.plot(tt, q, color="black", lw=1.4, label=r"$q_t=\max_p R\,\ell$ (novel)")
    ax.axhline(tau, color=C["baseline"], ls="--", lw=0.8)
    ax.text(0.3, tau + 0.03, r"$\tau$", color=C["baseline"])
    ax.axvline(junction / fps, color=C["accent"], lw=1.0)
    ax.text(junction / fps - 0.15, 1.02, "junction", color=C["accent"], ha="right")
    ax.set_ylabel(r"evidence $R\cdot\ell$")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left", handlelength=1.2, labelspacing=0.25)

    # -- row 2: gate state band ----------------------------------------------
    for st, col in STATE_ROW.items():
        mask = np.array([s == st for s in states])
        axs.fill_between(tt, 0, 1, where=mask, color=col, step="mid", lw=0)
    axs.set_yticks([]); axs.set_ylabel("state", rotation=0, ha="right", va="center")
    for spine in axs.spines.values():
        spine.set_visible(False)

    # -- row 3: fire ticks ----------------------------------------------------
    axf.eventplot([np.array(reactive) / fps], lineoffsets=[1.45], linelengths=0.6,
                  colors=[C["baseline"]])
    axf.eventplot([np.array(fires) / fps], lineoffsets=[0.55], linelengths=0.6,
                  colors=[C["ours"]], linewidths=1.6)
    if decided_at is not None:
        axf.plot(decided_at / fps, 0.55, marker="o", ms=3.5, color=C["DECIDED"], zorder=5)
        axf.annotate("decided", (decided_at / fps, 0.55), textcoords="offset points",
                     xytext=(4, -1), color=C["DECIDED"], va="center")
    axf.set_yticks([0.55, 1.45])
    axf.set_yticklabels([f"ours ({len(fires)})", f"reactive ({len(reactive)})"])
    axf.set_ylim(0, 2)
    axf.set_xlabel("time [s]")
    for spine in ("left",):
        axf.spines[spine].set_visible(False)

    if demo:
        fig.text(0.99, 0.99, "DEMO DATA — replace with E1 replay",
                 ha="right", va="top", color=C["bad"], fontsize=6, alpha=0.8)

    figstyle.save(fig, out_stem, arrays={
        "t_s": tt, "q": q, "tau": np.array([tau]),
        "states": np.array(states), "fires": np.array(fires),
        "reactive_fires": np.array(reactive), "junction_s": np.array([junction / fps]),
        **{f"plate{i}_Rell": tr.R * tr.ell for i, tr in enumerate(tracks)},
    })
    print(f"wrote {out_stem}.pdf/.png  ours fired {len(fires)}x at "
          f"{[f'{f/fps:.1f}s' for f in fires]}, decided at "
          f"{decided_at/fps if decided_at else None}s; reactive fired {len(reactive)}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="E1 per-bag dump")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default="figs/ar_timeline")
    a = ap.parse_args()
    if a.demo:
        tracks, junction, lat = demo_tracks()
    elif a.npz:
        tracks, junction, lat = load_tracks(a.npz)
    else:
        ap.error("need --demo or --npz")
    make(tracks, junction, lat, a.tau, a.fps, a.out, demo=a.demo)
