# Hand tracking

`egohand-live run` turns a camera stream into 3D hand poses with NVIDIA EgoHand and publishes
them over ZeroMQ. This page covers sources, camera placement and pose, the tracker's behaviour,
the output format and performance.

## How it differs from the offline pipeline

| EgoPost (offline) | egohand-live (online) |
| --- | --- |
| Whole-clip RT-DETR tracker with bidirectional gap filling | Causal tracker: frame-to-frame continuity, left/right from image position, confidence filter |
| Bidirectional adaptive smoothing | One Euro filter on wrist translation (separate depth cutoff) and hand shape |
| Batches of 32 frames | One frame, both hands in one batch of 2 |
| npz / mp4 output | msgpack over ZeroMQ: SOMA-25 and OpenXR-26 joint poses |

`scripts/verify_against_upstream.py` checks the per-frame predictor against EgoPost's
`EgoHandPredictor.run`: left hands match to < 0.01 mm, right hands to < 0.6 mm (geometric mirror
instead of re-skinning), while running ~1.5x faster.

## Sources

```bash
egohand-live run --source realsense --show --compile      # RealSense, factory intrinsics; --compile: ~5 s start-up, ~3 ms/frame faster
egohand-live run --source webcam:0 --intrinsics cam.json --show
egohand-live run --source video:clip.mp4 --intrinsics cam.json --scale 0.5 --record overlay.mp4
```

- `--intrinsics` takes an EgoPost camera JSON: `{"model": "PINHOLE", "image_width": W,
  "image_height": H, "camera_matrix": {"matrix_3x3": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}}`.
  Metric depth depends on correct intrinsics. For video files, distorted calibrations
  (RATIONAL_POLYNOMIAL, KB4, FISHEYE624) are undistorted on the fly.
- Video files replay in real time: frames are dropped when tracking falls behind, as with a camera.
- `--publish [ENDPOINT]` (default `tcp://*:5555`) publishes hand poses; `--publish-frames` adds JPEG
  frames for remote viewers. `egohand-live view --connect tcp://HOST:5555` shows them with an overlay.
- `--rerun` streams 3D meshes to [Rerun](https://rerun.io) (local, or `rerun+http://HOST:9876/proxy`).

## Camera and GPU on different machines

```bash
egohand-live camera-server --source realsense --bind tcp://*:5556      # camera machine
egohand-live run --source tcp://CAMERA_HOST:5556 --publish --publish-frames   # GPU machine
egohand-live view --connect tcp://GPU_HOST:5555                        # camera machine
```

If the camera machine's firewall blocks incoming connections (common on managed laptops), let the
GPU machine listen and the camera connect out:

```bash
egohand-live run --source "tcp://*:5556" --publish --publish-frames    # GPU machine
egohand-live camera-server --source realsense --connect tcp://GPU_HOST:5556   # camera machine
```

Either side can start first or restart; the GPU side waits for frames. A machine that only hosts
the camera or the viewer needs no GPU, torch or model:

```bash
uv venv .venv --python 3.12 && uv pip install -e ".[realsense]"
```

**RealSense USB troubleshooting.** Repeated disconnects, `Failed to set UVC probe control`, or
`device not accepting address ..., error -71` in `journalctl -k` are USB link problems: use a short
USB 3 cable directly into the computer and check that `lsusb -t` shows the camera at 5000M. A
dropped RealSense is restarted automatically for up to 30 s.

## Camera placement

EgoHand and its detector are trained on head-mounted video, and the detector's left/right classes
are from the wearer's point of view. Put the camera at about head height behind or above the
operator, looking down at the hands on the desk; the [desk stand](../hardware/d405_desk_mount)
is built for this. A camera facing the operator from across the desk is outside the training
distribution.

A good starting point, similar to a seated person's eyes: camera 45–55 cm above the desk, pitched
about 60° down, hands 50–65 cm from the camera.

## Camera pose

Simulators need `T_world_camera`, a 4x4 transform from the camera (OpenCV axes: x right, y down,
z forward) to a desk frame (surface at z = 0, +x right, +y away from the operator, +z up). Three
ways to get it:

- **From the desk stand settings** (recommended):
  `python3 hardware/d405_desk_mount/mount_geometry.py --collar-height 520 --tilt 60 > T_world_camera.txt`.
  The tilt is the scale reading, measured below horizontal.
- **From RealSense depth**: `egohand-live calibrate-desk --d405-mount --out T_world_camera.txt`
  fits the desk plane (height, tilt, roll; `--live` shows a setup view while you adjust the stand).
  The D405 has no IR projector, so lay a textured sheet (newspaper, printed page) on a plain desk.
- **Default**: the simulator falls back to `--camera-height` / `--camera-tilt`.

Pass the file to the simulator (`--extrinsics`) or to `run --extrinsics`, which sends it with every
message.

## Tracker behaviour

- **Left and right** come from image position by default (`--handedness position`): with the camera
  behind the operator, the left hand is the one further left. The detector's own label is
  unreliable for palm-up hands (a left palm looks like a right hand's back), which otherwise
  produces mirrored, upside-down fits. `--handedness detector` uses the label.
- **Confidence**: a hand is published once its track confidence (a running average of detector
  scores) reaches `--min-confidence` (0.45) over 3 frames, and dropped when it falls below 0.25.
  This hides false detections on desk clutter while tolerating brief low-score frames of real hands.
- `--det-threshold` (0.3) is the score that starts a track; `--box-scale` (1.3) the crop padding.

## Output

`egohand-live/hands/v1` messages on topic `hands` (see
[egohand_live/protocol.py](../egohand_live/protocol.py)) carry, per hand: SOMA-25 joint positions,
OpenXR-26 joint poses (positions, XYZW orientations, radii) in the OpenCV camera frame (meters),
2D keypoints, track confidence, timings and an optional `T_world_camera`. Orientations follow the
OpenXR convention (+Y out of the back of the hand, −Z toward the fingertips), built from the joint
positions, so headset retargeters can consume them.

## Performance

RTX 5090, bf16, `--compile`, 1014x760 frames (NVIDIA sample clip downscaled with `--scale 0.5`),
~1.7 hands per frame, on a host also running unrelated CPU-heavy jobs:

| Stage | mean | p95 |
| --- | ---: | ---: |
| RT-DETR detection (incl. resize) | 5.3 ms | 6.4 ms |
| EgoHand (crop, forward, both hands in one batch) | 34.7 ms | 43.0 ms |
| Smoothing + OpenXR conversion | 0.5 ms | 0.7 ms |
| **Throughput** | **~22–23 FPS** | |
| Capture-to-result latency at 20 FPS input | 50 ms (p50) | 84 ms |

The DINOv3 ViT-H+ backbone takes ~10 ms (compute-bound; GeForce cards run bf16 matmuls with FP32
accumulation at half rate), SOMA skinning ~10 ms and the pose head ~5 ms (launch-bound, sensitive
to CPU load).

On an RTX 3080 Ti laptop that also runs Isaac Sim, tracking reached ~10 FPS with ~230 ms from
capture to poses; tracking alone and further tuning have not been measured yet. The model uses
~6 GB of GPU memory; the slim checkpoint (`egohand-live slim-ckpt`) avoids a ~17 GB RAM peak when
loading the original 10.7 GB file.
