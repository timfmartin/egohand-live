# Robot hands in Isaac Sim

`isaac/sharpa_hands.py` drives two [Sharpa Wave](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)
hands (22 DoF each) in Isaac Lab 3 from the live tracking:

```
egohand-live run --publish ──hands──► egohand-live retarget ──joint angles + wrist pose──► Isaac Lab
     (GPU machine)                    (DexPilot, child process)                        (Sharpa hands, FR3 Duo)
```

## Setup

On the machine with Isaac Sim (tested: Isaac Sim 6.2, Isaac Lab 3.0, Ubuntu 24.04):

```bash
bash scripts/setup_isaac.sh          # Sharpa models (pinned) + pyzmq/msgpack for Isaac Sim's Python
python3 scripts/fetch_simready.py    # desk props + Franka FR3 Duo (~375 MB; --only props|franka)
```

Isaac Sim's bundled Python already ships Isaac Lab, torch, pinocchio and dex-retargeting; the
setup script installs nothing into it (extra packages go to `third_party/isaac_pydeps`).
`ISAACSIM_PATH` defaults to `~/isaacsim`. The first start with a viewport compiles shaders and can
take a few minutes.

## Run

```bash
~/isaacsim/python.sh isaac/sharpa_hands.py --connect tcp://TRACKING_HOST:5555 --extrinsics T_world_camera.txt
~/isaacsim/python.sh isaac/sharpa_hands.py --connect tcp://TRACKING_HOST:5555 --extrinsics T_world_camera.txt --rig franka
```

Without a camera, start `egohand-live synthetic --publish "tcp://*:5555"` (any Python with the
repo installed) and connect to `tcp://localhost:5555` without `--extrinsics`: the synthetic
hands carry their own camera pose.

Press **R** in the viewport to reset the objects; anything that falls off the desk respawns.

## Rigs

**Free-floating hands** (`--rig floating`, default). Each hand base is a gravity-less body servoed
toward the tracked wrist with a velocity command every control step (proportional,
acceleration-limited from the simulated velocity), so contacts are resolved by the physics engine:
hands push and grasp objects with bounded force and stop at the desk instead of passing through.

**Franka FR3 Duo** (`--rig franka`, see [isaac/franka_rig.py](../isaac/franka_rig.py)). Franka
Robotics' official FR3 Duo asset (two FR3 v2.1 arms on the Duo mount) with a Sharpa hand fixed to
each flange at −45°, as Franka mounts its own gripper. Both assets are used unmodified and composed
in USD: the hand's world anchor is deactivated and a fixed joint connects its flange to the arm's
link7, so arms and hands form one articulation with each vendor's drive tuning. Each tracked wrist
pose drives its arm through damped-least-squares IK; joint steps are limited to the FR3 speed
limits, which also bounds the force when a hand is blocked. The Duo stands 0.20 m above the desk,
0.21 m in front of it: an IK reachability sweep over the desk workspace (palm down, ±30° yaw)
reached all poses there, against ~75 % with the mount at desk height.

In both rigs the fingers are position-driven with Sharpa's gains and the hand colliders get a
high-friction material for grasping.

## Retargeting

[egohand_live/retarget.py](../egohand_live/retarget.py) uses dex-retargeting's DexPilot with the
Sharpa Wave configs and the preprocessing of NVIDIA Isaac Teleop's `DexHandRetargeter`, so it
behaves as with headset hand tracking. Fingers are retargeted in the wrist frame, which makes them
insensitive to monocular depth errors. It runs as a child process so the optimizer does not compete
with the render loop for Python's GIL.

`egohand-live retarget` also works on its own (any Python with `pip install -e '.[retarget]'`): it
publishes `egohand-live/robot-hands/v1` messages (joint names and angles, base pose, tracked joints)
on topic `robot`, for other simulators or a ROS bridge.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--rig` | `floating` | `floating` hands or `franka` (FR3 Duo) |
| `--scene` | `simready` | `simready` desk props, `cubes`, or `empty` |
| `--simready-root` | `third_party/simready` | folder with `nvidia/…` and `frankarobotics/…` (or `$SIMREADY_ROOT`) |
| `--extrinsics` | – | `T_world_camera` file; else the one in the messages, else `--camera-height` / `--camera-tilt` |
| `--view` | per rig | `camera` (along the real camera) or `overview` (behind the robot) |
| `--follow` | 0.5 | per-step blend toward the tracked wrist (1 = no smoothing) |
| `--park-after` | 1.5 s | how long a lost hand holds still before it parks |
| `--max-hand-speed` | 1.2 m/s | free-floating hands only |
| `--grip-friction` | 1.5 | friction of the hand colliders |
| `--render-every`, `--control-every` | 4, 2 | physics runs at 120 Hz; render at 30 fps, control at 60 Hz |
| `--device` | `cpu` | PhysX device; CPU is faster than GPU for this small scene |
| `--no-keypoints` | – | hide the yellow tracked-joint markers |
| `--retarget-connect` | – | use an `egohand-live retarget` already running elsewhere |
| `--debug` | – | print commanded vs. simulated poses and a per-step time breakdown |

## Performance notes

On an RTX 3080 Ti laptop the FR3 Duo scene runs at ~0.75x real time with CPU physics. Two things
matter in Isaac Lab 3 when stepping a simulation by hand: `sim.step()` renders on every call unless
told otherwise, and GPU physics pays a GPU-to-Python synchronisation for every state read, which
dominates for a few articulations.
