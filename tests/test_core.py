"""Unit tests for the model-free parts: skeleton conversion, filters, tracker."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from egohand_live.filters import HandSmoother, OneEuroFilter
from egohand_live.skeleton import hand_frames, soma25_to_openxr26
from egohand_live.tracker import HandBoxTracker, TrackerConfig


def _flat_hand(side: str, curl: float = 0.0) -> np.ndarray:
    """Palm-down hand, fingers along +Y, in a Z-up right-handed frame (meters).

    Right hand: thumb/index toward -X, pinky toward +X. The left hand is its mirror.
    ``curl`` bends each non-thumb finger by ``curl * 60 deg`` at MCP, PIP and DIP (toward
    the palm, i.e. -Z).
    """
    lateral = {"thumb": -0.045, "index": -0.025, "middle": 0.0, "ring": 0.02, "pinky": 0.038}
    j = np.zeros((25, 3))
    j[1:5] = [[-0.02, 0.02, 0], [-0.04, 0.045, 0], [-0.055, 0.07, 0], [-0.065, 0.09, 0]]
    for f, finger in enumerate(("index", "middle", "ring", "pinky")):
        x = lateral[finger]
        base = 5 + 5 * f
        j[base : base + 5] = [[x * 0.4, 0.02, 0], [x, 0.09, 0], [x, 0.135, 0], [x, 0.16, 0], [x, 0.18, 0]]
        c, s_ = np.cos(curl * np.pi / 3), np.sin(curl * np.pi / 3)
        for k in (1, 2, 3):  # MCP, PIP, DIP: rotate the distal part about X, +Y toward -Z
            pivot = j[base + k].copy()
            v = j[base + k + 1 : base + 5] - pivot
            j[base + k + 1 : base + 5] = pivot + np.stack([v[:, 0], c * v[:, 1] + s_ * v[:, 2],
                                                          -s_ * v[:, 1] + c * v[:, 2]], axis=1)  # fmt: skip
    if side == "left":
        j[:, 0] *= -1
    return j


def test_openxr_axes_flat_hand():
    for side in ("left", "right"):
        R = hand_frames(_flat_hand(side), side)
        assert np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-6)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)
        for joint in (0, 6, 7, 11, 21):  # wrist + long-finger joints
            x, y, z = R[joint].T
            assert z @ [0, -1, 0] > 0.95, (side, joint)  # -Z toward fingertips (+Y)
            assert y @ [0, 0, 1] > 0.95, (side, joint)  # +Y out of the back of the hand
        # X = Y x Z points to world +X for both hands in this pose.
        assert R[0][:, 0] @ [1, 0, 0] > 0.95


def test_openxr_conversion_shapes_and_rotation_equivariance():
    j = _flat_hand("right")
    pos, quat, rad = soma25_to_openxr26(j, "right")
    assert pos.shape == (26, 3) and quat.shape == (26, 4) and rad.shape == (26,)
    assert np.allclose(pos[1:], j, atol=1e-6)
    assert np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-5)
    # Rigidly rotating the hand rotates every joint frame by the same rotation.
    Q = Rotation.from_euler("xyz", [30, -50, 110], degrees=True)
    pos2, quat2, _ = soma25_to_openxr26(Q.apply(j), "right")
    expected = (Q * Rotation.from_quat(quat)).as_quat()
    dots = np.abs(np.sum(expected * quat2, axis=1))
    assert np.all(dots > 1 - 1e-5)
    assert np.allclose(pos2, Q.apply(pos), atol=1e-5)


def test_curled_finger_frames_stay_valid():
    j = _flat_hand("right")
    j[8] = j[7] + [0, 0, -0.03]  # index DIP straight down (90 deg PIP flexion)
    j[9] = j[8] + [0, -0.02, 0]  # tip curled back toward the palm
    R = hand_frames(j, "right")
    assert np.isfinite(R).all()
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)
    # Lateral (X) axis of the flexed index stays along the knuckle line.
    assert R[7][:, 0] @ [1, 0, 0] > 0.95


def test_one_euro_converges_and_reduces_noise():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    xs = [f(np.array([[1.0 + rng.normal(0, 0.01)]]), i / 30) for i in range(300)]
    tail = np.array(xs[150:]).ravel()
    assert abs(tail.mean() - 1.0) < 0.005
    assert tail.std() < 0.005


def test_hand_smoother_resets_after_gap():
    s = HandSmoother()
    a = np.zeros((25, 3)) + [0, 0, 0.5]
    b = a + [0.2, 0, 0]
    s(a, 0.0)
    assert np.allclose(s(b, 1.0), b)  # 1 s gap > reset_after_s -> no lag


def _det(label_id, score, cx, cy, s):
    return SimpleNamespace(label_id=label_id, score=score,
                           xyxy=(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2))  # fmt: skip


def test_tracker_identity_survives_label_flip_and_holds():
    trk = HandBoxTracker(TrackerConfig(hold_s=0.1, confirm_frames=1))
    out = trk.update([_det(1, 0.9, 300, 400, 100), _det(2, 0.9, 900, 400, 100)], 0.0, 1280)
    assert set(out) == {"left", "right"}
    # Detector swaps the labels for one frame; continuity keeps the identities.
    out = trk.update([_det(2, 0.6, 305, 400, 100), _det(1, 0.6, 905, 400, 100)], 1 / 30, 1280)
    assert out["left"].bbox[0] < 400 and out["right"].bbox[0] > 800
    # A missed detection is held briefly, then dropped.
    out = trk.update([_det(1, 0.9, 310, 400, 100)], 2 / 30, 1280)
    assert out["right"].held and not out["left"].held
    out = trk.update([_det(1, 0.9, 310, 400, 100)], 0.3, 1280)
    assert "right" not in out
    assert out["left"].bbox[2] == np.float32(100 * 1.3)


def test_tracker_position_handedness_beats_wrong_labels():
    # Palms toward the camera: the detector calls the left hand "right" and vice versa.
    swapped = [_det(2, 0.9, 300, 400, 100), _det(1, 0.9, 900, 400, 100)]
    out = HandBoxTracker(TrackerConfig(confirm_frames=1)).update(swapped, 0.0, 1280)
    assert out["left"].bbox[0] < 400 and out["right"].bbox[0] > 800
    assert out["left"].new and out["right"].new
    out = HandBoxTracker(TrackerConfig(handedness="detector", confirm_frames=1)).update(swapped, 0.0, 1280)
    assert out["left"].bbox[0] > 800  # EgoPost behaviour: trust the label


def test_tracker_position_lone_hand_and_crossed_tracks():
    trk = HandBoxTracker(TrackerConfig(confirm_frames=1))
    # A lone hand clearly on one side takes that side; near the centre the label decides.
    assert set(trk.update([_det(1, 0.9, 1000, 400, 100)], 0.0, 1280)) == {"right"}
    trk.reset()
    assert set(trk.update([_det(1, 0.9, 680, 400, 100)], 0.0, 1280)) == {"left"}
    # Tracks that end up on the wrong sides of each other are swapped (and flagged new).
    trk.reset()
    trk.update([_det(1, 0.9, 500, 400, 120), _det(2, 0.9, 780, 400, 120)], 0.0, 1280)
    trk.tracks["left"], trk.tracks["right"] = trk.tracks["right"], trk.tracks["left"]
    out = trk.update([_det(1, 0.9, 505, 400, 120), _det(2, 0.9, 785, 400, 120)], 1 / 30, 1280)
    assert out["left"].bbox[0] < 600 and out["right"].bbox[0] > 700
    assert out["left"].new and out["right"].new
    out = trk.update([_det(1, 0.9, 510, 400, 120), _det(2, 0.9, 790, 400, 120)], 2 / 30, 1280)
    assert not out["left"].new and not out["right"].new


def test_tracker_confidence_hides_false_positives_keeps_real_hands():
    # A desk object scoring 0.15-0.5 (as seen live) never becomes a published hand.
    trk = HandBoxTracker()
    fp_scores = [0.49, 0.2, 0.31, 0.18, 0.25, 0.2, 0.15, 0.22, 0.12, 0.2]
    for k, sc in enumerate(fp_scores):
        assert trk.update([_det(1, sc, 80, 150, 120)], k / 30, 1280) == {}
    assert trk.tracks["left"] is None  # and its track is forgotten
    # A real hand is published after confirm_frames and survives a short low-score dip.
    trk = HandBoxTracker()
    outs = [trk.update([_det(1, sc, 300, 400, 100)], k / 30, 1280)
            for k, sc in enumerate([0.85, 0.88, 0.86, 0.15, 0.18, 0.87])]  # fmt: skip
    assert [("left" in o) for o in outs] == [False, False, True, True, True, True]
    assert outs[2]["left"].new and not outs[3]["left"].new


def test_tracker_ignores_low_score_new_tracks():
    trk = HandBoxTracker()
    assert trk.update([_det(1, 0.2, 300, 400, 100)], 0.0, 1280) == {}


def test_protocol_roundtrip():
    from egohand_live.protocol import decode_hands, encode_hands

    j = _flat_hand("left") + [0, 0, 0.5]
    pos, quat, rad = soma25_to_openxr26(j, "left")
    hand = SimpleNamespace(side="left", score=0.9, held=False, bbox=np.array([1.0, 2.0, 3.0]),
                           joints_cam=j, keypoints_2d=np.zeros((25, 2)), openxr_positions=pos,
                           openxr_orientations=quat, openxr_radii=rad)  # fmt: skip
    result = SimpleNamespace(index=7, t_capture=123.5, image_size=(1280, 720), K=np.eye(3),
                             timings_ms={"total": 12.0}, hands={"left": hand})  # fmt: skip
    msg = decode_hands(encode_hands(result, T_world_camera=np.eye(4)))
    assert msg["seq"] == 7 and msg["image_size"] == [1280, 720]
    (h,) = msg["hands"]
    assert h["side"] == "left"
    assert np.allclose(h["soma25_positions"], j, atol=1e-6)
    assert np.allclose(h["openxr26_orientations"], quat, atol=1e-6)
    assert np.allclose(msg["T_world_camera"], np.eye(4))


def test_camera_link_listening_gpu_side():
    """GPU side listens (tcp://*:PORT), camera side connects: frames + K arrive intact."""
    import socket
    import time

    import zmq

    from egohand_live.protocol import bind_or_connect, encode_frame
    from egohand_live.sources import ZmqSource

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    src = ZmqSource(f"tcp://*:{port}")
    pub = zmq.Context.instance().socket(zmq.PUB)
    assert bind_or_connect(pub, f"tcp://127.0.0.1:{port}") == "connect"
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], np.float32)
    img = np.full((48, 64, 3), 128, np.uint8)
    try:
        deadline = time.time() + 5
        frame = None
        while frame is None and time.time() < deadline:  # PUB drops until the link is up
            pub.send_multipart(encode_frame(img, K, time.time(), 7, 90))
            with src._reader._cond:
                frame = src._reader._frame
            time.sleep(0.05)
        assert frame is not None
        assert frame.bgr.shape == img.shape and frame.index == 7
        assert np.allclose(frame.K, K)
    finally:
        pub.close(linger=0)
        src.close()


def test_robot_hands_roundtrip():
    from egohand_live.protocol import decode_robot_hands, encode_robot_hands

    names = [f"right_j{i}" for i in range(22)]
    q = np.linspace(-1, 1, 22)
    src = {"seq": 3, "t_capture": 12.5, "T_world_camera": np.eye(4)}
    hand = {"side": "right", "joint_names": names, "joint_positions": q, "base_position": [0.1, 0.2, 0.5],
            "base_orientation": [0, 0, 0, 1], "openxr26_positions": np.zeros((26, 3)), "retarget_ms": 9.87654}
    msg = decode_robot_hands(encode_robot_hands(src, "sharpa_wave", [hand]))
    (h,) = msg["hands"]
    assert msg["robot"] == "sharpa_wave" and msg["seq"] == 3 and msg["t_capture"] == 12.5
    assert h["joint_names"] == names and np.allclose(h["joint_positions"], q, atol=1e-6)
    assert np.allclose(h["base_position"], [0.1, 0.2, 0.5]) and h["openxr26_positions"].shape == (26, 3)
    assert np.allclose(msg["T_world_camera"], np.eye(4))


def test_sharpa_retarget_matches_human_finger_directions():
    """Flat and curled synthetic hands -> Sharpa joints; robot fingertips (FK) point like the human's."""
    import pytest

    pytest.importorskip("dex_retargeting")
    pin = pytest.importorskip("pinocchio")
    from scipy.spatial.transform import Rotation

    from egohand_live.retarget import ROBOT_HANDS, HandRetargeter
    from egohand_live.skeleton import soma25_to_openxr26

    if not ROBOT_HANDS["sharpa_wave"].path("urdf", "right").is_file():
        pytest.skip("Sharpa assets not fetched (scripts/setup_isaac.sh)")
    for side in ("left", "right"):
        rt = HandRetargeter("sharpa_wave", side, low_pass_alpha=1.0)
        model = pin.buildModelFromUrdf(str(rt.spec.path("urdf", side)))
        data = model.createData()
        q_idx = {model.names[j]: model.joints[j].idx_q for j in range(1, model.njoints)}
        for curl in (0.0, 0.6):
            joints = _flat_hand(side, curl=curl) + [0.05, 0.0, 0.45]
            pos, quat, _ = soma25_to_openxr26(joints, side)
            q = pin.neutral(model)
            for _ in range(5):  # let the optimizer settle from its warm start
                qr = rt(pos, quat)
            for name, val in zip(rt.joint_names, qr):
                q[q_idx[name]] = val
            pin.framesForwardKinematics(model, data, q)
            oMb = data.oMf[model.getFrameId(rt.wrist_link)]
            _, base_q = rt.base_pose(pos, quat)
            R_cam_base = Rotation.from_quat(base_q).as_matrix()
            for finger, xr in (("index", 10), ("middle", 15), ("ring", 20)):
                robot = oMb.rotation.T @ (data.oMf[model.getFrameId(f"{side}_{finger}_fingertip")].translation
                                          - oMb.translation)  # fmt: skip
                human = R_cam_base.T @ (pos[xr] - pos[1])
                cos = robot @ human / np.linalg.norm(robot) / np.linalg.norm(human)
                assert cos > np.cos(np.radians(15)), (side, curl, finger, np.degrees(np.arccos(cos)))


def test_desk_calibration_recovers_camera_pose():
    from egohand_live.calibrate import d405_mount_settings, extrinsics_from_plane, fit_desk, tilt_roll_deg

    import importlib.util
    from egohand_live.calibrate import MOUNT_GEOMETRY

    spec = importlib.util.spec_from_file_location("mg", MOUNT_GEOMETRY)
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)
    # Ground truth: the mount at collar 520 mm, tilt 58 deg, plus 2 deg of roll.
    T = np.array(mg.t_world_camera(520, 58))
    roll = np.radians(2.0)
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    T[:3, :3] = T[:3, :3] @ Rz  # roll about the optical axis
    rng = np.random.default_rng(0)
    desk = np.c_[rng.uniform(-0.5, 0.5, 6000), rng.uniform(0.1, 0.7, 6000), np.zeros(6000)]
    box = np.c_[rng.uniform(0.0, 0.1, 800), rng.uniform(0.3, 0.4, 800), rng.uniform(0.0, 0.08, 800)]
    pw = np.vstack([desk, box])
    pc = (pw - T[:3, 3]) @ T[:3, :3] + rng.normal(0, 0.001, pw.shape)
    K = np.array([[650, 0, 640], [0, 650, 360], [0, 0, 1.0]])
    uv = pc[:, :2] / pc[:, 2:] * 650 + [640, 360]
    keep = (pc[:, 2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 720)
    fit = fit_desk(pc[keep], uv[keep], (1280, 720))
    tilt, r = tilt_roll_deg(fit.n)
    assert fit.reliable
    assert abs(fit.d - T[2, 3]) < 0.003 and abs(tilt - 58) < 0.5 and abs(r + 2.0) < 0.3  # right side down
    collar, xy = d405_mount_settings(fit.d, tilt)
    assert abs(collar - 520) < 5 and np.allclose(xy, T[:2, 3], atol=0.005)
    T_est = extrinsics_from_plane(fit.n, fit.d, xy)
    assert np.allclose(T_est[:3, :3], T[:3, :3], atol=0.02) and np.allclose(T_est[:3, 3], T[:3, 3], atol=0.006)


def test_synthetic_hands_grasp_and_roundtrip():
    from egohand_live.protocol import decode_hands, encode_hands
    from egohand_live.synthetic import DEFAULT_T_WORLD_CAMERA, make_result, synthetic_hand

    for side in ("left", "right"):
        open_, closed = synthetic_hand(side, 0.0), synthetic_hand(side, 1.0)
        assert np.linalg.norm(open_[4] - open_[9]) > 0.08  # thumb tip far from index tip
        assert np.linalg.norm(closed[4] - closed[9]) < 0.02  # ... and meets it in a grasp
        assert (closed[[9, 14, 19, 24], 2] < -0.04).all()  # fingertips fold under the palm
        R = hand_frames(open_, side)[0]
        assert R[:, 1] @ [0, 0, 1] > 0.99 and -R[:, 2] @ [0, 1, 0] > 0.99  # back up, fingers +Y
    T = DEFAULT_T_WORLD_CAMERA
    msg = decode_hands(encode_hands(make_result(2.0, 7, T), T))
    assert [h["side"] for h in msg["hands"]] == ["left", "right"] and msg["seq"] == 7
    wrist_w = msg["hands"][0]["openxr26_positions"][1] @ T[:3, :3].T + T[:3, 3]
    assert -0.3 < wrist_w[0] < 0 and 0.15 < wrist_w[1] < 0.45 and 0.05 < wrist_w[2] < 0.2  # left, over the desk
