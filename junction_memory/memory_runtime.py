#!/usr/bin/env python3
"""
memory_runtime.py
-----------------
ONLINE junction memory for SignWay adaptive reasoning — the runtime the four
probe scripts were measuring for. Causal: every decision at sample i uses only
samples <= i. Constants and math are kept identical to the probes wherever the
probe validated them (curvature turn detection, recall reprojection, beam
formula, lockout semantics), and the design follows README__4_ exactly:

  * store the sign's CONTENT (full plate directory), not the action      (§2.1)
  * allocentric bearings + read yaw, reproject at query time             (§2.2)
  * node anchor != decision point                                        (§2.3)
  * lateral BEAM is the universal trigger (cases 1-3)                    (§4.1)
  * edge polyline matching is a CONFIRMING channel, matched UNDIRECTED
    with perpendicular distance; direction inferred from matched-index
    drift; heading-vs-tangent only rejects ~90 deg crossings             (§4.2)
  * association must resolve BEFORE the AR gate arms -> lead distance    (§2.5)
  * misses, not wrong turns: rival suppression + create-don't-merge      (§9)
  * junction proximity is kept as BASELINE ONLY                          (§4.3)

Dependencies: numpy. Nothing else.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# Thresholds — README §10 starting points. Re-derive on new platforms.
# ----------------------------------------------------------------------------
LATERAL_M          = 1.5    # beam half-width: corridor half-width
MAX_RANGE_M        = 30.0   # beyond this odom confidence is the limit
EDGE_RADIUS_M      = 3.0    # perpendicular match radius (absorbs reverse offset)
EDGE_RADIUS_TIGHT  = 1.5    # same-direction confidence tier
TURN_CURV_THRESH   = 0.10   # rad/m  (turning radius < 10 m)
TURN_RATE_THRESH   = 0.25   # rad/s  fallback for in-place rotation
MIN_TURN_DEG       = 45.0
TURN_MERGE_S       = 1.0
SMOOTH_S           = 0.3
HEADING_TOL_DEG    = 50.0   # aligned / reversed bands; middle band = crossing
LOCKOUT_M          = 10.0   # must exceed the largest radius swept
SUSTAIN_M          = 2.0    # sustained-run requirement (rejects lucky frames)
DECREASE_BASE_M    = 1.0    # range must be closing over this much travel
NODE_ACCEPT_R      = 3.0    # anchor association radius (generous — §7.2 arcs)
VISIT_R            = 2.5    # counts as physically visiting the node
BIND_MAX_M         = 25.0   # a VLM read binds to the next node within this
PROXIMITY_R        = 2.0    # baseline-only gate, kept for comparison (§4.3)
GAP_M              = 0.6    # max hole inside a sustained run
SIGN_STALE_M       = 1.5    # a fired plate unseen for this much travel means
                            # we have drawn abeam of it and are past it
BRANCH_MERGE_DEG   = 40.0   # bearings closer than this are the same branch

REL_OFFSETS_DEG = {"straight": 0.0, "left": 90.0, "right": -90.0, "back": 180.0}
ACTION_BINS_DEG = {"straight": 0.0, "left": 90.0, "right": -90.0, "back": 180.0}
_REL_SYNONYMS = {
    "straight": "straight", "forward": "straight", "ahead": "straight",
    "left": "left", "turn_left": "left", "l": "left",
    "right": "right", "turn_right": "right", "r": "right",
    "back": "back", "behind": "back", "u_turn": "back", "uturn": "back",
}


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def norm_rel(rel: str) -> str:
    k = str(rel).strip().lower().replace("-", "_").replace(" ", "_")
    if k not in _REL_SYNONYMS:
        raise ValueError(f"unknown relative direction {rel!r}")
    return _REL_SYNONYMS[k]


def norm_goal(g: str) -> str:
    return " ".join(str(g).strip().casefold().split())



def directory_from_vlm(directory, drop=("stop",)):
    """Convert reasoner.VLMAnswer.directory into the {label: rel} form
    provide_directory() wants. Drops entries memory cannot reproject
    (`stop` is not a bearing) and unparseable rows, rather than raising."""
    out = {}
    for item in directory or []:
        if isinstance(item, dict):
            lab, dirn = item.get("label"), item.get("direction")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            lab, dirn = item
        else:
            continue
        if not lab or not dirn or str(dirn).strip().lower() in drop:
            continue
        try:
            out[norm_goal(lab)] = norm_rel(dirn)
        except ValueError:
            continue
    return out

# ----------------------------------------------------------------------------
# events — the append-only observation log (README §8: keep the log separate
# from the derived graph; the graph is rebuildable from it)
# ----------------------------------------------------------------------------
@dataclass
class Event:
    t: float
    s: float
    frame: int
    kind: str
    msg: str
    data: dict = field(default_factory=dict)
    severity: str = "info"      # info | major | warn

    def jsonable(self):
        d = {"t": self.t, "s": self.s, "frame": self.frame, "kind": self.kind,
             "msg": self.msg, "severity": self.severity}
        d.update({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in self.data.items()})
        return d


# ----------------------------------------------------------------------------
# graph — README §8 schema (JSON, schema_version 1)
# ----------------------------------------------------------------------------
@dataclass
class Branch:
    branch_id: str
    bearing_abs: float                 # allocentric, radians, float (§2.2)
    edge: Optional[str] = None


@dataclass
class Node:
    node_id: str
    x: float
    y: float
    branches: dict = field(default_factory=dict)     # bid -> Branch
    routing: dict = field(default_factory=dict)      # goal -> {branch, src, conf, frame}
    read_yaw: Optional[float] = None
    sign_visible_from: list = field(default_factory=list)
    read_count: int = 0
    last_read: Optional[str] = None
    visits: int = 1

    # -- branches ------------------------------------------------------------
    def nearest_branch(self, bearing: float):
        best, err = None, 1e9
        for b in self.branches.values():
            e = abs(math.degrees(wrap_pi(bearing - b.bearing_abs)))
            if e < err:
                best, err = b, e
        return best, err

    def register_branch(self, bearing: float, edge: Optional[str] = None) -> "Branch":
        b, err = self.nearest_branch(bearing)
        if b is not None and err < BRANCH_MERGE_DEG:
            if edge is not None and b.edge is None:
                b.edge = edge
            return b
        bid = f"b{len(self.branches)}"
        nb = Branch(bid, float(wrap_pi(bearing)), edge)
        self.branches[bid] = nb
        return nb

    # -- recall (identical math to odom_memory_probe.JunctionNode.recall) ----
    def recall(self, goal: str, yaw_now: float, matcher=None):
        """matcher(goal, [stored_labels]) -> label | None. Sign labels are not
        goal strings ("Rooms 340-360" vs "6-214"), so exact key match only
        works when the goal was the thing asked about. Pass the AR gate's
        goal-affinity scorer to resolve the general case."""
        g = norm_goal(goal)
        entry = self.routing.get(g)
        matched = None
        if entry is None and matcher is not None:
            try:
                lab = matcher(goal, list(self.routing))
            except Exception:
                lab = None
            if lab:
                matched = norm_goal(lab)
                entry = self.routing.get(matched)
        if entry is None:
            return None, None
        br = self.branches.get(entry["branch"])
        if br is None:
            return None, None
        rel = math.degrees(float(wrap_pi(br.bearing_abs - yaw_now)))
        action = min(ACTION_BINS_DEG,
                     key=lambda k: abs(float(wrap_deg(rel - ACTION_BINS_DEG[k]))))
        margin = abs(float(wrap_deg(rel - ACTION_BINS_DEG[action])))
        return action, {"rel_deg": rel, "bin_error_deg": margin,
                        "flip_margin_deg": 45.0 - margin,
                        "branch": br.branch_id, "src": entry.get("src", "vlm"),
                        "conf": entry.get("conf"), "matched_label": matched}


@dataclass
class EdgeRec:
    edge_id: str
    endpoints: list                     # [node_id | None, node_id | None]
    pts: np.ndarray                     # (N,2) decimated polyline, travel order
    cum: np.ndarray                     # (N,) arc length along polyline
    tan: np.ndarray                     # (N,) tangent bearing per vertex (rad)

    @property
    def length(self) -> float:
        return float(self.cum[-1]) if len(self.cum) else 0.0

    @staticmethod
    def from_xy(edge_id, endpoints, xs, ys, step=0.25):
        pts = [(float(xs[0]), float(ys[0]))]
        for xi, yi in zip(xs[1:], ys[1:]):
            px, py = pts[-1]
            if math.hypot(xi - px, yi - py) >= step:
                pts.append((float(xi), float(yi)))
        if len(pts) < 2 and len(xs) >= 2:
            pts.append((float(xs[-1]), float(ys[-1])))
        P = np.asarray(pts, dtype=float)
        seg = np.diff(P, axis=0)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        tanv = np.arctan2(seg[:, 1], seg[:, 0])
        tan = np.concatenate([tanv, tanv[-1:]]) if len(tanv) else np.zeros(1)
        return EdgeRec(edge_id, list(endpoints), P, cum, tan)

    def perp_query(self, px, py):
        """Perpendicular distance to the polyline (README §4.2 — absorbs the
        reverse-traversal lateral offset that nearest-SAMPLE distance cannot).
        Returns (dist, arc_u, tangent_at_match)."""
        if len(self.pts) < 2:
            d = math.hypot(px - self.pts[0, 0], py - self.pts[0, 1])
            return d, 0.0, float(self.tan[0])
        A = self.pts[:-1]
        B = self.pts[1:]
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-12)
        AP = np.array([px, py]) - A
        tpar = np.clip((AP * AB).sum(1) / L2, 0.0, 1.0)
        proj = A + tpar[:, None] * AB
        d = np.hypot(proj[:, 0] - px, proj[:, 1] - py)
        j = int(np.argmin(d))
        u = float(self.cum[j] + tpar[j] * (self.cum[j + 1] - self.cum[j]))
        return float(d[j]), u, float(self.tan[j])


# ----------------------------------------------------------------------------
# per-candidate trackers (all causal)
# ----------------------------------------------------------------------------
class _Run:
    """Sustained-run bookkeeping: 'first' can be a lucky frame; trust the run."""
    __slots__ = ("start_s", "last_s", "fired")

    def __init__(self):
        self.start_s = None
        self.last_s = None
        self.fired = False

    def update(self, ok: bool, s: float) -> bool:
        if ok:
            if self.start_s is None or (s - self.last_s) > GAP_M:
                self.start_s, self.fired = s, False
            self.last_s = s
            if not self.fired and (s - self.start_s) >= SUSTAIN_M:
                self.fired = True
                return True
        else:
            if self.last_s is not None and (s - self.last_s) > GAP_M:
                self.start_s = None
                self.fired = False
        return False

    def active(self) -> bool:
        return self.fired


class BeamTrack:
    def __init__(self):
        self.run = _Run()
        self.armed = False           # must first get LOCKOUT_M away (probe semantics)
        self.locked = False
        self.min_range = math.inf
        self.rival_flagged = False

    def reset_episode(self):
        self.run = _Run()
        self.locked = False
        self.min_range = math.inf
        self.rival_flagged = False


class EdgeTrack:
    def __init__(self):
        self.run = _Run()
        self.us: list = []           # (s_now, u) inside current run
        self.tight = 0
        self.total = 0
        self.announced = False
        self.cross_hits = 0
        self.cross_announced = False

    def reset(self):
        self.__init__()


# ----------------------------------------------------------------------------
# online turn detection — causal port of odom_memory_probe.derive/detect_turns
# (curvature over a trailing 1 m baseline; rate channel kept for in-place)
# ----------------------------------------------------------------------------
class OnlineTurnDetector:
    def __init__(self, curv_thresh=TURN_CURV_THRESH, rate_thresh=TURN_RATE_THRESH,
                 min_turn_deg=MIN_TURN_DEG, merge_s=TURN_MERGE_S,
                 baseline_m=1.0, smooth_s=SMOOTH_S):
        self.kt, self.rt = curv_thresh, rate_thresh
        self.min_turn = min_turn_deg
        self.merge_s = merge_s
        self.base_m = baseline_m
        self.smooth_s = smooth_s
        self.in_turn = False
        self.i0 = None
        self.last_active_i = None
        self.last_active_t = None

    def step(self, i, t, s, yaw_u, rate_sm):
        """Returns a finalized (i0, i1) span or None. Arrays are full history."""
        # trailing curvature over >= base_m of travel
        j = i
        while j > 0 and s[i] - s[j] < self.base_m:
            j -= 1
        dsb = s[i] - s[j]
        if dsb > 1e-3:
            kappa = abs(yaw_u[i] - yaw_u[j]) / dsb
        else:
            kappa = math.inf if abs(rate_sm) > self.rt else 0.0
        active = (kappa > self.kt) or (abs(rate_sm) > self.rt)

        out = None
        if active:
            if not self.in_turn:
                if (self.last_active_t is not None
                        and t[i] - self.last_active_t < self.merge_s
                        and self.i0 is not None):
                    pass                                    # merge: keep i0
                else:
                    self.i0 = i
                self.in_turn = True
            self.last_active_i, self.last_active_t = i, t[i]
        else:
            if self.in_turn:
                self.in_turn = False                        # tentative end
            if (self.last_active_t is not None
                    and t[i] - self.last_active_t >= self.merge_s
                    and self.i0 is not None):
                i0, i1 = self.i0, self.last_active_i
                self.i0 = None
                self.last_active_t = None
                total = math.degrees(yaw_u[i1] - yaw_u[i0])
                if abs(total) >= self.min_turn:
                    out = (i0, i1, total)
        return out


# ----------------------------------------------------------------------------
# the runtime
# ----------------------------------------------------------------------------
class MemoryRuntime:
    """
    Feed odometry with step(t, x, y, yaw). Feed AR-gate reads with
    provide_directory(...). Query resolves(goal) before arming the gate:

        necessity = pending AND (resolves(goal) is None)          # README §9
    """

    def __init__(self, bag_name="live", verify_tau=None, goal_matcher=None):
        self.bag = bag_name
        self.verify_tau = verify_tau
        self.goal_matcher = goal_matcher
        # history (grow-only python lists; ~66k samples for the 35 GB bag)
        self.t: list = []; self.x: list = []; self.y: list = []
        self.yaw_u: list = []; self.s: list = []; self.rate_sm: list = []
        self._smooth_buf: list = []
        self.frame = -1
        # graph
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, EdgeRec] = {}
        self._nn = 0; self._ne = 0
        # open segment (the edge currently being driven)
        self.seg_start = 0
        self.seg_from_node: Optional[str] = None
        # trackers
        self.beams: dict[str, BeamTrack] = {}
        self.etracks: dict[str, EdgeTrack] = {}
        self.prox_armed: dict[str, bool] = {}      # baseline-only gate
        # pending VLM read waiting to bind to the next node (§2.1/§2.3)
        self.pending_read = None
        # standing prompt resolved from memory
        self.standing = None                        # {node,goal,action,meta,s,lead}
        self.expected_node: Optional[str] = None    # beam/edge consensus target
        self._recalled_for: set = set()             # (node, goal) once per approach
        self._miss_logged: set = set()
        self.events: list[Event] = []
        self._verify = None                         # post-commit check (§9)

    # -- helpers -------------------------------------------------------------
    def _emit(self, kind, msg, severity="info", **data):
        e = Event(self.t[-1] if self.t else 0.0,
                  self.s[-1] if self.s else 0.0,
                  self.frame, kind, msg, data, severity)
        self.events.append(e)
        return e

    def _disp_from(self, x0, y0):
        """Straight-line distance from a past pose. Immune to shuffling: a
        three-point turn racks up path length while going nowhere."""
        return math.hypot(self.x[-1] - x0, self.y[-1] - y0)

    def _advance_past(self, x0, y0, yaw0):
        """How far we have advanced ALONG yaw0 since that pose. Negative while
        backing up. This is what 'have I driven past the sign' means -- backing
        up 2 m is not passing it."""
        return ((self.x[-1] - x0) * math.cos(yaw0)
                + (self.y[-1] - y0) * math.sin(yaw0))

    def _new_node_id(self):
        self._nn += 1
        return f"n_{self._nn:03d}"

    def _new_edge_id(self):
        self._ne += 1
        return f"e_{self._ne:03d}"

    def set_frame(self, frame_idx: int):
        self.frame = frame_idx

    # -- AR gate interface (README §9) ---------------------------------------
    def resolves(self, goal: str):
        """Standing memory decision for the node currently being approached.
        A hit zeroes the gate's necessity term at edge entry."""
        if self.standing and norm_goal(self.standing["goal"]) == norm_goal(goal) \
                and self.standing["node"] == self.expected_node:
            return self.standing
        return None

    def legibility_prior(self):
        """Sufficiency prior: is the sign at the expected node readable from
        the branch we are arriving by? (sign.visible_from_branches, §8)."""
        nid = self.expected_node
        if nid is None or nid not in self.nodes or not self.t:
            return None
        node = self.nodes[nid]
        if not node.sign_visible_from:
            return None
        arr = self._arrival_branch(node)
        return (arr is not None) and (arr.branch_id in node.sign_visible_from)

    def observe_plates(self, plate_ids):
        """Per-frame set of plate ids the evidence layer is currently tracking.

        Used only to notice when the plate we fired on leaves the field of
        view, which is the cue that we have just passed it (see
        _maybe_seed_from_sign). Safe to never call: seeding simply stays off.
        """
        pr = self.pending_read
        if pr and pr.get("plate_id") and pr["plate_id"] in set(plate_ids or ()):
            if self.t:
                pr["last_seen_pose"] = (self.x[-1], self.y[-1], self.yaw_u[-1])

    def _maybe_seed_from_sign(self):
        """Anchor a node on the SIGN when no turn will anchor one.

        Turn detection cannot see a junction the robot drives straight
        through, so a plate read there has nothing to bind to and expires
        (correctly -- binding it to the next turn would file it under a
        junction it never described). But a directional plate is itself
        evidence of a decision point: it is mounted at the junction it
        describes. When the plate we fired on drops out of the tracker while
        we are still moving forward, we have just drawn abeam of it, so we
        anchor a node HERE.

        The anchor is the ROBOT's pose, not the sign's: the sign hangs on a
        wall a metre or two off the centreline, and on a revisit the lateral
        beam wants the node inside a 1.5 m half-width of the forward axis.
        The robot's own track is that centreline.
        """
        pr = self.pending_read
        if not pr or not pr.get("plate_id") or pr.get("seeded"):
            return
        seen = pr.get("last_seen_pose")
        if seen is None or self._advance_past(*seen) < SIGN_STALE_M:
            return
        pr["seeded"] = True
        i = len(self.t) - 1
        yaw = self.yaw_u[i]
        # a turn may already have anchored a node right here; prefer it
        near = [(nid, math.hypot(n.x - self.x[i], n.y - self.y[i]))
                for nid, n in self.nodes.items()]
        near = sorted([c for c in near if c[1] < NODE_ACCEPT_R],
                      key=lambda c: c[1])
        if near:
            self._maybe_bind_read(self.nodes[near[0][0]])
            return
        node = self._create_node(self.x[i], self.y[i])
        self._emit("NODE_FROM_SIGN",
                   f"plate {pr['plate_id']} left the view after "
                   f"{self._disp_from(pr['read_x'], pr['read_y']):.1f} m — no "
                   f"turn anchored a "
                   f"junction here, so the SIGN anchors {node.node_id} at the "
                   f"robot pose (straight-through junction)", "major",
                   node=node.node_id, plate=pr["plate_id"],
                   read_to_seed_m=self._disp_from(pr["read_x"], pr["read_y"]))
        node.register_branch(wrap_pi(yaw + math.pi))     # back the way we came
        node.register_branch(wrap_pi(yaw))               # straight ahead
        self._close_segment(node, i, arrival_bearing=wrap_pi(yaw + math.pi))
        self.seg_start = i
        self.seg_from_node = node.node_id
        self._maybe_bind_read(node)
        self._after_visit(node)

    def provide_directory(self, directory_rel: dict, goal: str, decision: str,
                          conf: float = None, frame: int = None,
                          plate_id: str = None):
        """Called when the AR gate's VLM read completes on a FIRST visit.
        Stores the whole plate (§2.1) for binding to the next node (§2.3)."""
        if not self.t:
            return
        d = {norm_goal(k): norm_rel(v) for k, v in directory_rel.items()}
        self.pending_read = {
            "plate_id": plate_id,
            "last_seen_pose": ((self.x[-1], self.y[-1], self.yaw_u[-1])
                               if self.t else None),
            "read_x": self.x[-1] if self.t else 0.0,
            "read_y": self.y[-1] if self.t else 0.0,
            "directory_rel": d, "read_yaw": self.yaw_u[-1],
            "read_s": self.s[-1], "read_frame": frame if frame is not None else self.frame,
            "conf": conf, "goal": norm_goal(goal), "decision": norm_rel(decision),
        }
        self._emit("VLM_READ", f"VLM read plate: {len(d)} entries "
                   f"{sorted(d)} (will bind to next node)", "info",
                   directory=d, conf=conf)

    # -- main entry ----------------------------------------------------------
    def step(self, t, x, y, yaw):
        prev = len(self.t) - 1
        # unwrap yaw incrementally
        if prev < 0:
            yu = float(yaw)
            self.t.append(float(t)); self.x.append(float(x)); self.y.append(float(y))
            self.yaw_u.append(yu); self.s.append(0.0); self.rate_sm.append(0.0)
            self.detector = OnlineTurnDetector()
            return []
        yu = self.yaw_u[prev] + float(wrap_pi(yaw - self.yaw_u[prev]))
        ds = math.hypot(x - self.x[prev], y - self.y[prev])
        dt = max(t - self.t[prev], 1e-6)
        self.t.append(float(t)); self.x.append(float(x)); self.y.append(float(y))
        self.yaw_u.append(yu); self.s.append(self.s[prev] + ds)
        # smoothed rate over trailing SMOOTH_S
        self._smooth_buf.append(((yu - self.yaw_u[prev]) / dt, t))
        while self._smooth_buf and t - self._smooth_buf[0][1] > SMOOTH_S:
            self._smooth_buf.pop(0)
        self.rate_sm.append(sum(v for v, _ in self._smooth_buf) / len(self._smooth_buf))

        n_ev = len(self.events)
        i = len(self.t) - 1

        # 0) a fired plate that has left the view anchors its own junction
        self._maybe_seed_from_sign()

        # 1) turn detection -> node visits
        fin = self.detector.step(i, self.t, self.s, self.yaw_u, self.rate_sm[-1])
        if fin is not None:
            self._on_turn(*fin)

        # 2) beam trigger over all known nodes (universal, cases 1-3)
        self._beam_step(i)

        # 3) edge matching over stored polylines (confirming channel)
        self._edge_step(i)

        # 4) baseline proximity gate (comparison only)
        self._proximity_step(i)

        # 5) post-commit verification (§9)
        self._verify_step(i)

        return self.events[n_ev:]

    # ------------------------------------------------------------------ turns
    def _on_turn(self, i0, i1, total_deg):
        half = self.yaw_u[i0] + 0.5 * (self.yaw_u[i1] - self.yaw_u[i0])
        span = np.asarray(self.yaw_u[i0:i1 + 1])
        ai = i0 + int(np.argmin(np.abs(span - half)))
        ax, ay = self.x[ai], self.y[ai]
        self._emit("TURN", f"turn of {total_deg:+.0f} deg completed "
                   f"(anchor s={self.s[ai]:.1f} m)", "info",
                   anchor=[ax, ay], turn_deg=total_deg,
                   entry_yaw=self.yaw_u[i0], exit_yaw=self.yaw_u[i1])
        self._visit_node(ai, entry_yaw=self.yaw_u[i0], exit_yaw=self.yaw_u[i1],
                         via_turn=True, seg_end=i0, seg_next=i1)

    def _visit_node(self, ai, entry_yaw, exit_yaw, via_turn, seg_end, seg_next):
        ax, ay = self.x[ai], self.y[ai]
        # associate or create (create-don't-merge when ambiguous, §9)
        cands = [(nid, math.hypot(n.x - ax, n.y - ay))
                 for nid, n in self.nodes.items()]
        cands = [c for c in cands if c[1] < NODE_ACCEPT_R]
        cands.sort(key=lambda c: c[1])
        if len(cands) >= 2 and cands[1][1] < 1.5 * NODE_ACCEPT_R:
            node = self._create_node(ax, ay)
            self._emit("NODE_AMBIGUOUS", f"two nodes within accept radius — "
                       f"created {node.node_id} instead of merging (safe: costs "
                       f"one VLM call, not a wrong routing table)", "warn")
        elif cands:
            node = self.nodes[cands[0][0]]
            node.visits += 1
            self._emit("NODE_REVISIT", f"revisit of {node.node_id} "
                       f"(visit #{node.visits}, anchor err {cands[0][1]:.2f} m)",
                       "major", node=node.node_id, err_m=cands[0][1])
        else:
            node = self._create_node(ax, ay)

        # close the incoming edge at turn onset, open outgoing at turn end
        self._close_segment(node, seg_end, arrival_bearing=wrap_pi(entry_yaw + math.pi))
        node.register_branch(wrap_pi(exit_yaw))          # departure branch
        self.seg_start = seg_next
        self.seg_from_node = node.node_id

        # bind a pending VLM read (§2.1: the whole plate becomes routing)
        self._maybe_bind_read(node)
        self._after_visit(node)

    def _create_node(self, ax, ay):
        nid = self._new_node_id()
        node = Node(nid, float(ax), float(ay))
        self.nodes[nid] = node
        self.beams[nid] = BeamTrack()          # starts DISARMED at own node
        self.prox_armed[nid] = False
        self._emit("NODE_CREATED", f"junction node {nid} anchored at "
                   f"({ax:.1f}, {ay:.1f})", "major", node=nid, x=ax, y=ay)
        return node

    def _close_segment(self, node, seg_end, arrival_bearing):
        if seg_end - self.seg_start >= 4:
            eid = self._new_edge_id()
            xs = self.x[self.seg_start:seg_end + 1]
            ys = self.y[self.seg_start:seg_end + 1]
            rec = EdgeRec.from_xy(eid, [self.seg_from_node, node.node_id], xs, ys)
            if rec.length >= 2.0:
                self.edges[eid] = rec
                self.etracks[eid] = EdgeTrack()
                br = node.register_branch(arrival_bearing, edge=eid)
                if self.seg_from_node and self.seg_from_node in self.nodes:
                    src = self.nodes[self.seg_from_node]
                    # departure branch of the source node now owns this edge
                    dep_bearing = rec.tan[0]
                    src.register_branch(dep_bearing, edge=eid)
                self._emit("EDGE_STORED", f"stored corridor {eid} "
                           f"({rec.length:.1f} m, {self.seg_from_node or 'start'}"
                           f" -> {node.node_id})", "info",
                           edge=eid, length_m=rec.length)
                return
        node.register_branch(arrival_bearing)

    def _maybe_bind_read(self, node):
        pr = self.pending_read
        if pr is None:
            return
        if self._disp_from(pr["read_x"], pr["read_y"]) > BIND_MAX_M:
            self._emit("READ_EXPIRED", "VLM read expired unbound "
                       f"({self._disp_from(pr['read_x'], pr['read_y']):.0f} m "
                       f"from the read pose)", "warn")
            self.pending_read = None
            return
        node.read_yaw = pr["read_yaw"]
        node.read_count += 1
        node.last_read = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # the sign was read while driving TOWARD the node: the legibility
        # branch is the one pointing back along the read heading, not the
        # (noise-scale) node->robot bearing at the anchor
        arr = node.register_branch(wrap_pi(pr["read_yaw"] + math.pi))
        if arr.branch_id not in node.sign_visible_from:
            node.sign_visible_from.append(arr.branch_id)
        for goal, rel in pr["directory_rel"].items():
            bearing = wrap_pi(pr["read_yaw"] + math.radians(REL_OFFSETS_DEG[rel]))
            br = node.register_branch(bearing)
            node.routing[goal] = {"branch": br.branch_id, "src": "vlm",
                                  "conf": pr["conf"], "bag": self.bag,
                                  "frame": pr["read_frame"]}
        self._emit("DIRECTORY_STORED",
                   f"plate directory bound to {node.node_id}: "
                   + ", ".join(f"{g}->{node.routing[g]['branch']}"
                               for g in pr["directory_rel"]),
                   "major", node=node.node_id,
                   routing={g: node.routing[g]["branch"] for g in pr["directory_rel"]},
                   read_yaw_deg=math.degrees(wrap_pi(pr["read_yaw"])))
        self.pending_read = None

    def _arrival_branch(self, node):
        # branch pointing back the way we came in: bearing node -> robot
        if not self.t:
            return None
        bearing = math.atan2(self.y[-1] - node.y, self.x[-1] - node.x)
        b, err = node.nearest_branch(bearing)
        return b if (b is not None and err < BRANCH_MERGE_DEG) else None

    def _after_visit(self, node):
        # visiting disarms this node's beam + proximity until LOCKOUT_M away
        for nid, bt in self.beams.items():
            if nid == node.node_id:
                bt.armed = False
                bt.reset_episode()
        self.prox_armed[node.node_id] = False
        if self.expected_node == node.node_id:
            self.expected_node = None
        # standing prompt for this node is consumed
        if self.standing and self.standing["node"] == node.node_id:
            self._arm_verify(node)
            self.standing = None
        self._recalled_for = {k for k in self._recalled_for if k[0] != node.node_id}
        for et in self.etracks.values():
            et.reset()

    # ------------------------------------------------------------------ beam
    def _beam_step(self, i):
        px, py, psi = self.x[i], self.y[i], self.yaw_u[i]
        s_now = self.s[i]
        # geometry for every node (README §4.1 — five numbers, one rotation)
        geo = {}
        for nid, node in self.nodes.items():
            dx, dy = node.x - px, node.y - py
            fwd = dx * math.cos(psi) + dy * math.sin(psi)
            lat = -dx * math.sin(psi) + dy * math.cos(psi)
            rng = math.hypot(dx, dy)
            geo[nid] = (fwd, lat, rng)

        for nid, node in self.nodes.items():
            bt = self.beams[nid]
            fwd, lat, rng = geo[nid]
            # arm/disarm on lockout distance travelled away from the node
            if not bt.armed:
                if rng > LOCKOUT_M:
                    bt.armed = True
                continue
            in_beam = (fwd > 0) and (abs(lat) < LATERAL_M) and (rng < MAX_RANGE_M)
            closing = self._range_closing(nid, rng, s_now)
            rival = None
            if in_beam:
                for oid, (f2, l2, r2) in geo.items():
                    if oid == nid:
                        continue
                    if f2 > 0 and abs(l2) < LATERAL_M and r2 < 1.5 * rng:
                        rival = oid
                        break
            ok = in_beam and closing and rival is None
            if in_beam and rival is not None and not bt.rival_flagged:
                bt.rival_flagged = True
                self._emit("BEAM_RIVAL", f"beam on {nid} suppressed — rival "
                           f"{rival} at comparable range (miss, not a wrong "
                           f"turn)", "warn", node=nid, rival=rival)
            if bt.run.update(ok, s_now):
                bt.locked = True
                self.expected_node = nid
                self._emit("BEAM_LOCK",
                           f"LATERAL BEAM lock on {nid} — {rng:.1f} m out, "
                           f"|lateral| {abs(lat):.2f} m < {LATERAL_M} m, "
                           f"sustained {SUSTAIN_M:.0f} m", "major",
                           node=nid, range_m=rng, lateral_m=lat,
                           lead_m=rng)
                leg = self.legibility_prior()
                if leg is not None:
                    self._emit("GATE_PRIOR",
                               ("sign IS readable from this approach"
                                if leg else
                                "sign NOT readable from this approach — "
                                "a VLM call would be futile (sufficiency prior)"),
                               "info", node=nid, legible=leg)
                self._try_recall(nid, source="beam", lead=rng)
            if bt.locked:
                bt.min_range = min(bt.min_range, rng)
                if not ok and rng > bt.min_range + 2.0:
                    # receding without a turn-visit: straight-through pass?
                    if bt.min_range < VISIT_R:
                        self._straight_through(nid)
                    bt.reset_episode()
                    bt.armed = rng > LOCKOUT_M
                    if self.expected_node == nid:
                        self.expected_node = None

    def _range_closing(self, nid, rng, s_now):
        # compare against range ~DECREASE_BASE_M of travel ago
        hist = getattr(self, "_rng_hist", None)
        if hist is None:
            hist = self._rng_hist = {}
        h = hist.setdefault(nid, [])
        h.append((s_now, rng))
        while len(h) > 2 and s_now - h[0][0] > 2.5 * DECREASE_BASE_M:
            h.pop(0)
        past = [r for s0, r in h if s_now - s0 >= DECREASE_BASE_M]
        return bool(past) and rng < past[-1] - 0.05

    def _straight_through(self, nid):
        node = self.nodes[nid]
        node.visits += 1
        self._emit("NODE_VISIT_STRAIGHT",
                   f"passed straight through {nid} (closest "
                   f"{self.beams[nid].min_range:.2f} m) — no turn event; the "
                   f"beam, not turn detection, registered this visit", "major",
                   node=nid, closest_m=self.beams[nid].min_range)
        i = len(self.t) - 1
        entry_yaw = self.yaw_u[i]
        self._close_segment(node, i, arrival_bearing=wrap_pi(entry_yaw + math.pi))
        node.register_branch(wrap_pi(entry_yaw))
        self.seg_start = i
        self.seg_from_node = nid
        self._maybe_bind_read(node)
        self._after_visit(node)

    # ------------------------------------------------------------------ edges
    def _edge_step(self, i):
        px, py, psi = self.x[i], self.y[i], self.yaw_u[i]
        s_now = self.s[i]
        if s_now - self.s[self.seg_start] < 1.0:
            return
        for eid, rec in self.edges.items():
            et = self.etracks[eid]
            d, u, tan = rec.perp_query(px, py)
            dpsi = abs(wrap_deg(math.degrees(psi - tan)))
            aligned = dpsi < HEADING_TOL_DEG
            reversed_ = dpsi > 180.0 - HEADING_TOL_DEG
            crossing = not (aligned or reversed_)
            spatial = d < EDGE_RADIUS_M
            if spatial and crossing:
                et.cross_hits += 1
                if et.cross_hits > 10 and not et.cross_announced:
                    et.cross_announced = True
                    self._emit("EDGE_CROSSING",
                               f"near {eid} but heading ~90 deg to its tangent "
                               f"— crossing corridor, rejected", "info",
                               edge=eid)
            ok = spatial and not crossing
            fired = et.run.update(ok, s_now)
            if ok:
                et.us.append((s_now, u))
                et.total += 1
                if d < EDGE_RADIUS_TIGHT:
                    et.tight += 1
                if len(et.us) > 400:
                    et.us.pop(0)
            if fired and not et.announced:
                et.announced = True
                du = et.us[-1][1] - et.us[0][1]
                dsn = max(et.us[-1][0] - et.us[0][0], 1e-6)
                drift = du / dsn
                toward = rec.endpoints[1] if drift > 0 else rec.endpoints[0]
                same_dir = aligned
                tight_frac = et.tight / max(et.total, 1)
                dist_to_go = (rec.length - u) if drift > 0 else u
                label = ("same direction" if same_dir else
                         "REVERSE traversal" if reversed_ else "ambiguous")
                self._emit("EDGE_MATCH",
                           f"corridor {eid} recognised ({label}) — "
                           f"index drift {'+' if drift > 0 else '-'} "
                           f"=> approaching {toward or 'open end'}, "
                           f"{dist_to_go:.1f} m to go "
                           f"(perp err med {np.median([0]) if False else d:.2f} m, "
                           f"tight {100*tight_frac:.0f}%)", "major",
                           edge=eid, same_direction=bool(same_dir),
                           reverse=bool(reversed_), toward=toward,
                           dist_to_go=dist_to_go, tight_frac=tight_frac,
                           lead_m=dist_to_go)
                if toward and toward in self.nodes:
                    # confirming channel: agreeing with (or setting) the target
                    if self.expected_node in (None, toward):
                        self.expected_node = toward
                        conf = "confirms beam" if self.beams[toward].locked \
                               else "edge-only (beam not yet locked)"
                        self._emit("EDGE_CONFIRM",
                                   f"{eid} {conf}: expected node {toward}",
                                   "info", edge=eid, node=toward)
                        self._try_recall(toward, source="edge",
                                         lead=dist_to_go,
                                         confirmed=self.beams[toward].locked)
            if not ok and et.announced and et.run.start_s is None:
                et.announced = False
                et.us.clear(); et.tight = 0; et.total = 0

    # ------------------------------------------------------------- proximity
    def _proximity_step(self, i):
        px, py = self.x[i], self.y[i]
        for nid, node in self.nodes.items():
            rng = math.hypot(node.x - px, node.y - py)
            if not self.prox_armed.get(nid, False):
                if rng > LOCKOUT_M:
                    self.prox_armed[nid] = True
                continue
            if rng < PROXIMITY_R:
                self.prox_armed[nid] = False
                self._emit("PROXIMITY_FIRE",
                           f"[baseline] {PROXIMITY_R:.0f} m proximity gate on "
                           f"{nid} fires only now ({rng:.2f} m out) — after "
                           f"the AR gate would already have paid for the VLM "
                           f"call", "info", node=nid, range_m=rng)

    # ----------------------------------------------------------------- recall
    def _try_recall(self, nid, source, lead, confirmed=False):
        node = self.nodes[nid]
        goal = getattr(self, "current_goal", None)
        if goal is None:
            return
        key = (nid, norm_goal(goal))
        if key in self._recalled_for:
            return
        yaw_now = self.yaw_u[-1]
        action, meta = node.recall(goal, yaw_now, matcher=self.goal_matcher)
        if action is None:
            if node.routing:
                if key not in self._miss_logged:
                    self._miss_logged.add(key)
                    self._emit("MEMORY_MISS",
                               f"'{goal}' not on {nid}'s stored plate "
                               f"({sorted(node.routing)}) — first-pass prompt "
                               f"did not capture the whole directory; falling "
                               f"through to the VLM (prompt problem, not a "
                               f"memory problem)", "warn", node=nid, goal=goal)
            return
        if self.verify_tau is not None and meta.get("conf") is not None \
                and meta["conf"] < self.verify_tau:
            leg = self.legibility_prior()
            if leg:
                self._emit("VERIFY_REREAD",
                           f"cached conf {meta['conf']:.2f} < tau "
                           f"{self.verify_tau:.2f} and sign readable — "
                           f"re-reading instead of trusting cache", "warn",
                           node=nid, goal=goal)
                return
        self._recalled_for.add(key)
        self.standing = {"node": nid, "goal": goal, "action": action,
                         "meta": meta, "source": source, "lead_m": lead,
                         "confirmed_by_edge": confirmed}
        self._emit("MEMORY_RECALL",
                   f"MEMORY RECALL at {nid}: goal '{goal}' -> {action.upper()} "
                   f"(rel {meta['rel_deg']:+.1f} deg, flips at 45, margin "
                   f"{meta['flip_margin_deg']:.0f} deg) — resolved {lead:.1f} m "
                   f"out via {source}; VLM call not needed", "major",
                   node=nid, goal=goal, action=action, source=source,
                   lead_m=lead, confirmed_by_edge=confirmed, **meta)

    # ------------------------------------------------- post-commit verify §9
    def _arm_verify(self, node):
        st = self.standing
        if not st:
            return
        g = norm_goal(st["goal"])
        entry = node.routing.get(g)
        exp_edge = None
        if entry:
            br = node.branches.get(entry["branch"])
            if br and br.edge and br.edge in self.edges:
                exp_edge = br.edge
        self._verify = {"node": node.node_id, "edge": exp_edge,
                        "start_s": self.s[-1], "action": st["action"]}

    def _verify_step(self, i):
        v = self._verify
        if not v:
            return
        travelled = self.s[i] - v["start_s"]
        if v["edge"] is None:
            if travelled > 3.0:
                self._emit("VERIFY_SKIP", "no stored corridor for the branch "
                           "taken — nothing to verify against", "info")
                self._verify = None
            return
        rec = self.edges[v["edge"]]
        d, _, tan = rec.perp_query(self.x[i], self.y[i])
        dpsi = abs(wrap_deg(math.degrees(self.yaw_u[i] - tan)))
        on_it = d < EDGE_RADIUS_M and (dpsi < HEADING_TOL_DEG
                                       or dpsi > 180 - HEADING_TOL_DEG)
        if travelled > 4.0:
            if on_it:
                self._emit("VERIFY_OK",
                           f"post-commit check: corridor entered matches "
                           f"stored {v['edge']} (perp {d:.2f} m) — recall "
                           f"verified", "info", edge=v["edge"])
            else:
                self._emit("VERIFY_MISMATCH",
                           f"post-commit check FAILED: {d:.1f} m off stored "
                           f"{v['edge']} — revoke and fall back to the VLM",
                           "warn", edge=v["edge"])
            self._verify = None

    # ---------------------------------------------------------------- export
    def to_json(self) -> dict:
        nodes = {}
        for nid, n in self.nodes.items():
            b0 = next(iter(n.branches.values())).bearing_abs if n.branches else 0.0
            nodes[nid] = {
                "x": round(n.x, 2), "y": round(n.y, 2),
                "arity": len(n.branches),
                "branches": {bid: {
                    "edge": b.edge,
                    "bearing_local_deg": round(
                        float(wrap_deg(math.degrees(b.bearing_abs - b0))), 1),
                    "bearing_abs_deg": round(
                        float(math.degrees(wrap_pi(b.bearing_abs))), 1),
                } for bid, b in n.branches.items()},
                "routing": {g: dict(r) for g, r in n.routing.items()},
                "read_yaw_rad": (round(float(wrap_pi(n.read_yaw)), 4)
                                 if n.read_yaw is not None else None),
                "sign": {"visible_from_branches": list(n.sign_visible_from),
                         "read_count": n.read_count, "last_read": n.last_read},
                "visits": n.visits,
            }
        edges = {eid: {"endpoints": e.endpoints,
                       "length_m": round(e.length, 1),
                       "polyline": [[round(float(px), 2), round(float(py), 2)]
                                    for px, py in e.pts]}
                 for eid, e in self.edges.items()}
        return {"schema_version": 1, "bag": self.bag,
                "nodes": nodes, "edges": edges}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)


# ----------------------------------------------------------------------------
# yaw-scale correction, applied ONLINE (identical result to
# calibrate_yaw.reintegrate; parity is unit-tested)
# ----------------------------------------------------------------------------
class YawScaler:
    """Feeds raw odom through the constant-multiplier rotational fix (§7.1)."""

    def __init__(self, k: float):
        self.k = float(k)
        self._raw_prev = None
        self._yaw0 = None
        self._xc = self._yc = None

    def step(self, x, y, yaw):
        if self._raw_prev is None:
            self._raw_prev = (x, y)
            self._yaw0 = yaw
            self._yaw_u = yaw
            self._xc, self._yc = x, y
            return x, y, yaw
        self._yaw_u += float(wrap_pi(yaw - self._yaw_u))
        yaw_c = self._yaw0 + self.k * (self._yaw_u - self._yaw0)
        # SIGNED step: hypot alone is a magnitude, so a reversing step would be
        # re-integrated FORWARD along the heading and every three-point turn
        # would come out as a loop. Project onto the heading to recover the
        # direction (valid for a non-holonomic base). Must stay identical to
        # calibrate_yaw.signed_step -- parity is unit-tested.
        dx = x - self._raw_prev[0]
        dy = y - self._raw_prev[1]
        mag = math.hypot(dx, dy)
        proj = dx * math.cos(yaw) + dy * math.sin(yaw)
        ds = math.copysign(mag, proj) if mag > 1e-9 else 0.0
        self._xc += ds * math.cos(yaw_c)
        self._yc += ds * math.sin(yaw_c)
        self._raw_prev = (x, y)
        return self._xc, self._yc, yaw_c