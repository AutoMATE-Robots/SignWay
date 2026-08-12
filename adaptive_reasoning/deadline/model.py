"""Deadline estimator: frozen DINOv2-S/14 backbone + small two-head MLP.

Heads:
  * p(junction ahead within d_max)  -- BCE
  * distance quantiles (q10, q50, q90) -- pinball loss, supervised only where
    junction_ahead == 1

The gate consumes q10 (the PESSIMISTIC distance): if the model believes the
junction is 10-15 m away, the deadline plans against 10. Quantiles are sorted at
inference to enforce monotonicity. Backbone stays frozen -- the project-wide
lesson (protect pretrained vision; train small heads).

All torch imports are inside functions so the rest of the package imports
without torch installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

QUANTILES = (0.1, 0.5, 0.9)
FEAT_DIM = 384  # DINOv2 ViT-S/14 CLS


def load_backbone(device: str = "cuda"):
    """Frozen DINOv2-S/14 (downloads via torch.hub on first use; respects HF_HOME/torch cache)."""
    import torch

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def embed_frames(backbone, frames_rgb: list, device: str = "cuda", batch: int = 64) -> np.ndarray:
    """RGB uint8 HxWx3 frames -> (N, 384) CLS features. Resize/normalize per DINOv2."""
    import torch
    import torch.nn.functional as F

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for i in range(0, len(frames_rgb), batch):
            chunk = frames_rgb[i:i + batch]
            x = torch.stack([torch.from_numpy(np.ascontiguousarray(f)) for f in chunk])
            x = x.permute(0, 3, 1, 2).float().to(device) / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - mean) / std
            out.append(backbone(x).cpu().numpy())
    return np.concatenate(out, 0)


def build_head(feat_dim: int = FEAT_DIM, hidden: int = 256):
    import torch.nn as nn

    class DeadlineHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU())
            self.cls = nn.Linear(hidden, 1)                 # junction_ahead logit
            self.q = nn.Linear(hidden, len(QUANTILES))       # d quantiles (metres)

        def forward(self, x):
            h = self.trunk(x)
            return self.cls(h).squeeze(-1), self.q(h)

    return DeadlineHead()


def pinball_loss(pred_q, target, quantiles=QUANTILES):
    """Quantile (pinball) loss; pred_q (B, Q), target (B,)."""
    import torch

    t = target.unsqueeze(-1)
    qs = torch.tensor(quantiles, device=pred_q.device).view(1, -1)
    e = t - pred_q
    return torch.maximum(qs * e, (qs - 1.0) * e).mean()


class DeadlineEstimator:
    """Inference wrapper the gate uses: frame -> (p_junction, d_q10, d_q50, d_q90)."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        import torch

        self.device = device
        self.backbone = load_backbone(device)
        self.head = build_head().to(device)
        state = torch.load(ckpt_path, map_location=device)
        self.head.load_state_dict(state["head"])
        self.head.eval()
        self.meta = state.get("meta", {})

    def predict(self, frame_rgb: np.ndarray):
        import torch

        feat = embed_frames(self.backbone, [frame_rgb], self.device)
        with torch.inference_mode():
            logit, q = self.head(torch.from_numpy(feat).to(self.device))
            p = torch.sigmoid(logit).item()
            qv = np.sort(q.cpu().numpy().ravel())  # enforce q10 <= q50 <= q90
        return dict(p_junction=p, d_q10=float(qv[0]),
                    d_q50=float(qv[1]), d_q90=float(qv[2]))


def save_metrics(path: Path, metrics: dict):
    path.write_text(json.dumps(metrics, indent=2))
