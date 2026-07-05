"""
omni_backend.py — the model seam for the probe.

Two backends behind ONE interface:
    predict_waypoints(images, goal_pose) -> np.ndarray of shape (N, 2)
      images:    list of RGB frames (HxWx3 uint8). OmniVLA expects 2
                 (num_images_in_input 2): typically [previous_frame, current_frame].
                 For a static probe, pass [frame, frame].
      goal_pose: (x, y, theta) SE(2) goal in the robot frame
                 (x forward+, y left+, theta CCW). "right" => y negative.
      returns:   (N, 2) relative waypoints in meters, robot frame (x forward, y left).

- MockBackend: no model needed. Forward-biased arc that steers toward the goal's
  lateral offset, mimicking OmniVLA's known behavior (8 waypoints, mostly forward,
  goal modulates steering). Use it to validate the harness + visualization today.
- OmniVLABackend: skeleton that wraps the real run_omnivla.py inference. Two TODO
  blocks to fill from the repo (load + forward). See README for exactly what to paste.
"""

from typing import List, Sequence, Tuple

import numpy as np

Pose = Tuple[float, float, float]


def _find_run_omnivla_dir():
    """Find the directory containing run_omnivla.py (the OmniVLA `inference/` dir),
    walking up from this file and checking each ancestor AND its `inference/` subdir,
    then put it on sys.path so `import run_omnivla` works regardless of launch dir.
    Only the real backend needs this; the mock doesn't."""
    import sys
    from pathlib import Path
    here = Path(__file__).resolve()
    for d in [here.parent, *here.parents]:
        for cand in (d, d / "inference"):
            if (cand / "run_omnivla.py").exists():
                if str(cand) not in sys.path:
                    sys.path.insert(0, str(cand))
                return cand
    print("[omni_backend] warning: run_omnivla.py not found near "
          f"{here}; set the path manually if the import fails.")
    return None


class Backend:
    name = "base"

    def predict_waypoints(self, images: List[np.ndarray], goal_pose: Pose) -> np.ndarray:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Mock — runnable now, no weights. Good enough to exercise viz + the convention check.
# ──────────────────────────────────────────────────────────────────────────────
class MockBackend(Backend):
    name = "mock"

    def __init__(self, n_waypoints: int = 8, step_m: float = 0.30,
                 turn_gain: float = 0.6, noise: float = 0.0, seed: int = 0):
        # step_m ~ OmniVLA's metric_waypoint_spacing (per-step forward distance).
        self.n = n_waypoints
        self.step = step_m
        self.turn_gain = turn_gain
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def predict_waypoints(self, images, goal_pose):
        gx, gy, _ = goal_pose
        # desired bearing to the goal (atan2(left, forward)); steer toward it gradually
        target_bearing = np.arctan2(gy, max(gx, 1e-3))
        x, y, heading = 0.0, 0.0, 0.0
        wps = []
        for _ in range(self.n):
            # turn a fraction of the remaining bearing error each step (forward-dominant)
            err = target_bearing - heading
            heading += self.turn_gain * err / self.n
            x += self.step * np.cos(heading)
            y += self.step * np.sin(heading)
            if self.noise:
                x += self.rng.normal(0, self.noise)
                y += self.rng.normal(0, self.noise)
            wps.append((x, y))
        return np.asarray(wps, float)


# ──────────────────────────────────────────────────────────────────────────────
# Real OmniVLA — reuse inference/run_omnivla.py's load + forward path (option A).
#   Conditions on POSE only (modality_id = 4): satellite/language/image goals off.
#   Returns the 8x2 position waypoints (cols [x_forward, y_left]) scaled to meters.
#   run_omnivla documents the output as "X is front, Y is left", so it already
#   matches the probe's convention — no axis flip expected.
# ──────────────────────────────────────────────────────────────────────────────
class OmniVLABackend(Backend):
    name = "omnivla"

    def __init__(self, checkpoint_dir=None, step=None,
                 metric_waypoint_spacing=None, device="cuda"):
        # run_omnivla hardcodes metric_waypoint_spacing = 0.1
        self.spacing = 0.1 if metric_waypoint_spacing is None else float(metric_waypoint_spacing)
        self.device = device
        self._load(checkpoint_dir, step)

    def _load(self, checkpoint_dir, step):
        import os
        inf_dir = _find_run_omnivla_dir()
        import run_omnivla as R
        self.R = R

        # POSE-ONLY modality (=> modality_id 4). These are module-level globals that
        # run_forward_pass reads; set them before any forward call.
        R.satellite = False
        R.lan_prompt = False
        R.pose_goal = True
        R.image_goal = False

        cfg = R.InferenceConfig()
        if checkpoint_dir is None:
            checkpoint_dir = (os.path.join(str(inf_dir), "omnivla-original")
                              if inf_dir else "./omnivla-original")
        cfg.vla_path = os.path.abspath(checkpoint_dir)  # absolute => launch-dir independent
        if step is not None:
            cfg.resume_step = step

        (self.vla, self.action_head, self.pose_projector, self.device_id,
         self.NUM_PATCHES, self.action_tokenizer, self.processor) = R.define_model(cfg)

        # one Inference instance, reused only for its batch-build + forward methods
        self.inf = R.Inference(
            save_dir=".", lan_inst_prompt="xxxx", goal_utm=(0.0, 0.0, 0.0),
            goal_compass=0.0, goal_image_PIL=None,
            action_tokenizer=self.action_tokenizer, processor=self.processor,
        )

    def predict_waypoints(self, images, goal_pose):
        import numpy as np
        from PIL import Image

        cur = images[-1]
        cur_pil = cur if isinstance(cur, Image.Image) else Image.fromarray(
            np.asarray(cur).astype(np.uint8))
        goal_img_pil = cur_pil  # pose-only ignores the goal image; the 2-image slot is still filled

        x_fwd, y_left, theta = goal_pose
        s = self.spacing
        # OmniVLA's normalized goal-pose vector: [forward/spacing, left/spacing, cos, sin].
        # If the combined plot shows RIGHT goals bending LEFT, negate the 2nd term (-y_left/s).
        goal_pose_loc_norm = np.array(
            [x_fwd / s, y_left / s, np.cos(theta), np.sin(theta)], dtype=np.float64)

        batch = self.inf.data_transformer_omnivla(
            cur_pil, "xxxx", goal_img_pil, goal_pose_loc_norm,
            prompt_builder=self.R.PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )
        actions, _ = self.inf.run_forward_pass(
            vla=self.vla.eval(), action_head=self.action_head.eval(),
            noisy_action_projector=None, pose_projector=self.pose_projector.eval(),
            batch=batch, action_tokenizer=self.action_tokenizer, device_id=self.device_id,
            use_l1_regression=True, use_diffusion=False, use_film=False,
            num_patches=self.NUM_PATCHES, mode="train", idrun=0,
        )
        wp = actions.float().cpu().numpy()[0]   # (8, 4): [dx, dy, cos h, sin h], normalized
        return wp[:, :2] * s                    # (8, 2) meters, [x_forward, y_left]


def make_backend(kind: str, **kw) -> Backend:
    kind = kind.lower()
    if kind == "mock":
        return MockBackend(**{k: v for k, v in kw.items()
                              if k in ("n_waypoints", "step_m", "turn_gain", "noise", "seed")})
    if kind in ("omnivla", "omni"):
        return OmniVLABackend(**{k: v for k, v in kw.items()
                                 if k in ("checkpoint_dir", "step",
                                          "metric_waypoint_spacing", "device")})
    raise ValueError(f"unknown backend {kind!r} (use 'mock' or 'omnivla')")
