# Hand tracking

`egohand-live run` turns a camera stream into 3D hand poses with NVIDIA EgoHand and publishes
them over ZeroMQ. This page covers sources, camera placement and pose, the tracker's behaviour,
the output format and performance.

## How it differs from the offline pipeline

| EgoPost (offline) | egohand-live (online) |
| --- | --- |
| Whole-clip RT-DETR tracker with bidirectional gap filling | Causal tracker: frame-to-frame continuity, left/right from image position, confidence filter |
| Bidirectional adaptive smoothing | One Euro filter on wrist translation (separate depth cutoff) and hand shape |
| Batches of 32 frames | One frame; the hands in view in one batch (CUDA graph per hand count) |
| npz / mp4 output | msgpack over ZeroMQ: SOMA-25 and OpenXR-26 joint poses |

`scripts/verify_against_upstream.py` checks the per-frame predictor against EgoPost's
`EgoHandPredictor.run`: left hands match to < 0.01 mm, right hands to < 0.6 mm (geometric mirror
instead of re-skinning), while running ~1.5x faster.

## Sources

```bash
egohand-live run --source realsense --show --compile      # RealSense, factory intrinsics; --compile: ~10 s start-up, ~7 ms/frame faster
egohand-live run --source webcam:0 --intrinsics cam.json --show
egohand-live run --source video:clip.mp4 --intrinsics cam.json --scale 0.5 --record overlay.mp4
```

- `--intrinsics` takes an EgoPost camera JSON: `{"model": "PINHOLE", "image_width": W,
  "image_height": H, "camera_matrix": {"matrix_3x3": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}}`.
  Metric depth depends on correct intrinsics. Distorted calibrations (RATIONAL_POLYNOMIAL, KB4,
  FISHEYE624, as written by `calibrate-camera`) are undistorted on the fly; `--zoom` < 1 keeps more
  of a wide field of view.
- `--rotate 90|180|270` turns the image (and intrinsics) when your hands enter from another edge.
- `--depth-scale` corrects the monocular depth bias measured by `calibrate-depth`.
- `--save-session DIR` records frames and per-frame results for `render-video`.
- Video files replay in real time: frames are dropped when tracking falls behind, as with a camera.
- `--publish [ENDPOINT]` (default `tcp://*:5555`) publishes hand poses; `--publish-frames` adds JPEG
  frames for remote viewers. `egohand-live view --connect tcp://HOST:5555` shows them with an overlay.
- `--rerun` streams 3D meshes to [Rerun](https://rerun.io) (local, or `rerun+http://HOST:9876/proxy`).

## Webcam setup

A webcam needs its lens measured once, and every placement needs the camera pose; both come from a
board. A printed [ChArUco board](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)
is best: it is detected even when partly outside the image, which matters for wide lenses. A
plain checkerboard on a phone screen also works.

```bash
egohand-live preview --source webcam:0                     # live view with framing guides (r: rotate)
egohand-live calibrate-camera --source webcam:0 --board charuco:auto:27 --out cam.json
egohand-live calibrate-desk-board --source webcam:0 --intrinsics cam.json --board charuco:auto:27 \
    --out T_world_camera.txt
```

- **Lens** (`calibrate-camera`): hold the board under the camera and move it to the corners of the
  image, nearer and further, tilted different ways. Views are captured automatically when the
  board is still and new; 20 by default. Three lens models are fitted (standard, rational and
  Kannala-Brandt fisheye) and the best kept (`--model`). Boards: `charuco:auto:SQUARE_MM`
  identifies layout, dictionary and marker size from the first views and prints the full spec;
  `checker:9x6` is OpenCV's standard checkerboard (`--show-pattern` shows it full-screen,
  `--write-pattern` saves it for a phone).
- **Camera pose** (`calibrate-desk-board`): lay the board flat in the middle of your work area.
  It becomes the workspace centre. The board's size sets the scale, so give the measured square
  size; `--board-height-mm 8` for a phone lying on the desk.
- **Depth scale** (`calibrate-depth`, while `run --publish --extrinsics ...` runs): rest a flat
  hand on the bare desk at six spots for two seconds each. The known desk plane gives each spot's
  true distance; the command prints the `--depth-scale` to use, and warns when the error varies
  too much to correct (camera too far from the hands).

`calibrate-desk-hands` estimates the camera pose from hand touches alone, without a board. It
relies on the monocular depth, so it only suits cameras close to the hands; prefer the board.

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
about 60° down, hands 50–65 cm from the camera. Keep the hands within ~0.9 m of the camera:
further away the monocular depth error becomes random and cannot be calibrated out.

## Camera pose

Simulators need `T_world_camera`, a 4x4 transform from the camera (OpenCV axes: x right, y down,
z forward) to a desk frame (surface at z = 0, +x right, +y away from the operator, +z up). Three
ways to get it:

- **From a board on the desk** (any camera; see [Webcam setup](#webcam-setup)):
  `egohand-live calibrate-desk-board ... --out T_world_camera.txt`.
- **From the desk stand settings** (D405 on the stand):
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

The per-frame path is tuned for consumer GPUs that share their time with a simulator:

- fp16 with fp16 accumulation (`--dtype float16`, the default): twice the matrix throughput of
  bf16 on GeForce cards, and closer to an fp32 reference (0.3–0.4 mm mean joint difference against
  1.3 mm for bf16). `--dtype bfloat16` restores the earlier behaviour.
- The backbone, pose head and keypoint head run as CUDA graphs, one per hand count; the detector
  runs through ONNX Runtime IO binding with its own CUDA graph. This removes most per-frame CPU
  work, so tracking stays fast when the CPU is busy (worst 1 % of frames under a heavy CPU load:
  725 → ~120 ms).
- Backbone weights are stored in the compute dtype once instead of being cast every frame;
  SOMA's shape regressor is precomputed; crops are normalised on the GPU.
- `--compile` adds per-block `torch.compile` inside the graphs (Linux; ~10 s first start).

RTX 3080 Ti laptop, 1280x720 webcam frames, 1.7 hands per frame on average, tracking alone:

| | default | `--compile` |
| --- | ---: | ---: |
| RT-DETR detection (incl. resize) | 10–12 ms | 10–12 ms |
| EgoHand, two hands / one hand | 55–59 / 37–40 ms | 48–53 / 32–37 ms |
| **Per frame** | **~70 ms (~14 FPS)** | **58–63 ms (~16 FPS)** |
| GPU memory | 4.6 GB | 4.6 GB |

Running Isaac Sim on the same GPU (Sharpa hands, `--render minimal`): 40–50 ms per frame with one
hand, ~65 ms from capture to the robot hand moving, the simulation at 1.0× real time. With the
simulator's default RTX rendering, the GPU is saturated and tracking halves.

Earlier measurement on an RTX 5090 (bf16, `--compile`, before these optimisations): ~22–23 FPS
for two hands, 50 ms capture-to-result latency. The slim checkpoint (`egohand-live slim-ckpt`)
avoids a ~17 GB RAM peak when loading the original 10.7 GB file.
