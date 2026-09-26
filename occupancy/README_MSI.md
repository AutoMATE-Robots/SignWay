# MSI Occupancy Server + Jetson Thin Client

Moves the heavy part of the Phase-1 occupancy pipeline (Depth Anything V2)
from the Jetson to an MSI A40, using the same server/tunnel pattern as
`policy_server.py` / `pepper_vla_node.py`.

```text
Jetson                                   MSI A40 (agc* node)
/image_raw + /camera_info                msi_occupancy_server.py
   -> jetson_occupancy_client.py            rectify -> Depth Anything V2
   -> JPEG over SSH tunnel  ------------->  -> RANSAC floor -> occupancy
   <- PNG (or npz) reply    <-------------  grid + debug image
   -> /local_occupancy_image  (rqt_image_view exactly as in Phase 1)
```

The occupancy math is **imported from `camera_occupancy_node.py`** — this
directory must contain that file (or sit next to it in the repo). Nothing is
copied; `tests/test_occupancy_math.py` covers both runtimes.

## Files

| file | runs on | needs |
|---|---|---|
| `msi_occupancy_server.py` | MSI A40 | numpy, opencv, CUDA torch, Depth Anything V2 |
| `run_msi_server.sh` | MSI A40 | the `occ` env below |
| `jetson_occupancy_client.py` | Jetson | rclpy, cv_bridge, cv2, numpy (stdlib HTTP — no torch) |
| `camera_occupancy_node.py` | both (server imports its math) | — |

## One-time MSI setup

Do **not** install anything into the pinned `oft` env mid-training. Make a
small separate env:

```bash
export SCRATCH=/scratch.global/$USER
conda create -p $SCRATCH/conda_envs/occ python=3.10 -y
conda activate $SCRATCH/conda_envs/occ
pip install "numpy<2" opencv-python-headless torch --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/DepthAnything/Depth-Anything-V2 $SCRATCH/Depth-Anything-V2
# put the metric hypersim checkpoint(s) in:
#   $SCRATCH/Depth-Anything-V2/metric_depth/checkpoints/
#   depth_anything_v2_metric_hypersim_{vits|vitb|vitl}.pth
```

Copy this folder to MSI (e.g. `~/Trajectory_Generation_and_Control/occupancy/`),
including `camera_occupancy_node.py`.

## Run order

**1. MSI — grab an A40 and start the server** (save-and-bash, as always):

```bash
bash run_msi_server.sh --model-size small          # port 8890 by default
```

Wait for `occupancy server ready on port 8890`. Note the node name (`agcNN`).

**2. Jetson — open the tunnel** (same shape as the policy-server tunnel,
different port; adjust to however you currently reach the A40 node):

```bash
ssh -N -L 8890:agcNN:8890 munda057@<msi-login-host>
```

**3. Jetson — start camera driver, then the client:**

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /image_raw            # confirm frames
python3 jetson_occupancy_client.py  # or with --ros-args -p server_url:=http://localhost:8890
```

**4. View live**, unchanged from Phase 1:

```bash
rqt_image_view   # select /local_occupancy_image
```

Sanity checks along the way:

```bash
curl http://localhost:8890/health                          # from the Jetson, via tunnel
ros2 topic hz /local_occupancy_image                       # end-to-end rate
```

## Protocol (for Phase 2 / trajectory refinement)

- `POST /calib` — JSON `{width,height,K,D,R,P}` (flat lists, CameraInfo order).
  The client sends this automatically whenever CameraInfo changes and re-sends
  on HTTP 409 (server restart).
- `POST /occupancy` — body = JPEG bytes.
  - default reply: PNG of the debug image (white free / black obstacle / gray
    unknown, robot at bottom), headers `X-Depth-Ms`, `X-Total-Ms`.
  - `POST /occupancy?fmt=npz` — reply is an npz with `grid` (int16, ROS
    semantics −1/0/100, rows increase forward, cols increase camera-right),
    `debug`, `meta` (resolution, extents), `depth_ms`, `total_ms`. **This is
    the payload the refinement layer should consume** — raw grid, not the
    scaled debug PNG.
- `GET /health` — status JSON.

## Notes / gotchas

- **Latency:** expect the A40 to cut depth-inference time substantially vs the
  Jetson; the new costs are JPEG encode/decode (a few ms) and the tunnel
  round trip (network-dependent — the client logs `round-trip` vs `server`
  ms so you can see the split immediately). If round-trip is dominated by
  network, drop `jpeg_quality` to 80.
- **Sharing the A40 with the VLA policy server:** Depth Anything small/base is
  ~0.1–0.4 GB of weights, so VRAM is fine next to the 7B policy. It does
  contend for compute — measure policy-server latency with the occupancy
  server under load before a live run. Do NOT co-locate with a v10/v11
  *training* job (standing rule: training owns the A40).
- **Client behavior:** newest-frame-only (single slot, stale frames dropped),
  identical to the local node. If the tunnel dies the client logs and retries;
  nothing crashes.
- The server keeps calibration in memory only. Restart server → client gets a
  409 on the next frame and re-sends calib automatically.
