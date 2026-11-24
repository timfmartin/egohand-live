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


def test_camera_calibration_recovers_lens():
    import cv2

    from egohand_live.camcal import board_points, calibrate, egopost_json, view_descriptor
    from egohand_live.targets import Detection

    K = np.array([[910.0, 0, 655.0], [0, 905.0, 352.0], [0, 0, 1]])
    dist = np.array([-0.32, 0.12, 0.001, -0.0005, -0.02])  # cheap-webcam barrel distortion
    cols, rows, size = 9, 6, (1280, 720)
    obj = board_points(cols, rows) * 0.025
    rng = np.random.default_rng(1)

    def random_views(n, noise=0.2):
        out = []
        while len(out) < n:
            rvec = rng.uniform([-0.6, -0.6, -0.4], [0.6, 0.6, 0.4])
            tvec = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.1, 0.02), rng.uniform(0.35, 0.7)])
            px, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            px = px.reshape(-1, 2) + rng.normal(0, noise, (len(obj), 2))
            if (px.min(0) > 10).all() and (px.max(0) < np.array(size) - 10).all():
                out.append(Detection(np.arange(len(obj)), obj.astype(np.float32), px.astype(np.float32)))
        return out

    views = random_views(15)
    K_est, d_est, rms, per_view = calibrate(views, size, "standard")
    assert rms < 0.5 and per_view.max() < 1.0
    assert np.allclose(np.diag(K_est)[:2], np.diag(K)[:2], rtol=0.01)
    assert np.allclose(K_est[:2, 2], K[:2, 2], atol=6)
    # Held-out views (noise-free) reproject with the estimated lens to well under a pixel.
    for v in random_views(5, noise=0.0):
        ok, rvec, tvec = cv2.solvePnP(obj, v.img, K_est, d_est)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K_est, d_est)
        assert ok and np.abs(proj.reshape(-1, 2) - v.img).max() < 1.0
    cal = egopost_json(K_est, d_est, size)
    assert cal["model"] == "RATIONAL_POLYNOMIAL" and len(cal["distortion_coefficients"]) == 14
    assert len(view_descriptor(views[0], size, 0.2)) == 5


def test_fisheye_lens_from_partial_charuco_views():
    """A wide lens calibrated from ChArUco views that leave the image: auto picks the fisheye model."""
    import cv2

    from egohand_live.camcal import calibrate_best, egopost_json
    from egohand_live.targets import Charuco, Detection

    K = np.array([[420.0, 0, 642.0], [0, 421.0, 357.0], [0, 0, 1]])  # ~140 deg wide at 1280 px
    D = np.array([0.05, -0.02, 0.004, -0.001])
    size = (1280, 720)
    board = Charuco(9, 7, 0.027, 0.020, "5X5_100")
    rng = np.random.default_rng(3)
    views = []
    while len(views) < 18:
        rvec = rng.uniform([-0.7, -0.7, -0.5], [0.7, 0.7, 0.5])
        tvec = np.array([rng.uniform(-0.25, 0.1), rng.uniform(-0.15, 0.05), rng.uniform(0.18, 0.45)])
        cam = board.points @ cv2.Rodrigues(rvec)[0].T + tvec
        if (cam[:, 2] < 0.05).any():
            continue
        px = cv2.fisheye.projectPoints(board.points.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, K, D)[0]
        px = px.reshape(-1, 2) + rng.normal(0, 0.15, (len(board.points), 2))
        inside = ((px > 5) & (px < np.array(size) - 5)).all(1)
        if inside.sum() >= 12:  # partial views are fine
            ids = np.where(inside)[0]
            views.append(Detection(ids, board.points[ids], px[ids].astype(np.float32)))
    model, K_est, d_est, rms, _ = calibrate_best(views, size, "auto")
    assert model == "kb4" and rms < 0.4
    assert np.allclose(np.diag(K_est)[:2], np.diag(K)[:2], rtol=0.01)
    assert egopost_json(K_est, d_est, size, model)["model"] == "KB4"


def test_charuco_board_identified_from_image():
    import cv2

    from egohand_live.targets import Charuco, identify_charuco, parse_board

    truth = Charuco(9, 7, 0.027, 0.020, "5X5_100")
    img = truth.board.generateImage((9 * 60, 7 * 60), marginSize=40)
    H = np.array([[0.9, 0.12, 200], [-0.05, 0.85, 80], [1e-4, 2e-4, 1]])  # board lying on a desk, seen at an angle
    img = cv2.warpPerspective(img, H, (1280, 720), borderValue=255)
    found = identify_charuco(img, 0.027)
    assert (found.squares_x, found.squares_y, found.legacy) == (9, 7, False)
    assert found.dictionary.startswith("5X5") and abs(found.marker - 0.020) < 0.0025
    det = parse_board(found.spec).detect(img)
    assert det is not None and len(det.ids) == 48


def test_rotated_intrinsics_match_rotated_image():
    import cv2

    from egohand_live.sources import rotate_K

    K = np.array([[900, 0, 610], [0, 880, 350], [0, 0, 1]], np.float32)
    W, H, P = 1280, 720, np.array([0.12, -0.05, 0.6])
    u, v = (K @ P)[:2] / P[2]
    img = np.zeros((H, W), np.uint8)
    img[int(round(v)), int(round(u))] = 255
    for deg, code, R in ((90, cv2.ROTATE_90_CLOCKWISE, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                         (180, cv2.ROTATE_180, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                         (270, cv2.ROTATE_90_COUNTERCLOCKWISE, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])):  # fmt: skip
        vv, uu = np.argwhere(cv2.rotate(img, code) == 255)[0]
        P2 = np.array(R) @ P
        u2, v2 = (rotate_K(K, W, H, deg) @ P2)[:2] / P2[2]
        assert abs(u2 - uu) < 0.6 and abs(v2 - vv) < 0.6


def test_desk_frame_from_hand_touches():
    from egohand_live.calibrate import desk_frame_from_points

    # Camera 0.7 m above the desk, looking down 60 deg, operator's hands at the image bottom.
    t = np.radians(60)
    R_wc = np.array([[1, 0, 0], [0, -np.sin(t), np.cos(t)], [0, -np.cos(t), -np.sin(t)]])  # columns: cam axes
    cam = np.array([0.05, -0.1, 0.7])
    touches_w = np.array([[-0.2, 0.2, 0], [0.2, 0.25, 0], [0.0, 0.45, 0], [-0.15, 0.4, 0], [0.18, 0.5, 0.0]])
    touches_c = (touches_w - cam) @ R_wc + np.random.default_rng(0).normal(0, 0.002, touches_w.shape)
    T, rms = desk_frame_from_points(touches_c)
    assert rms < 0.004
    assert np.allclose(T[:3, :3], R_wc, atol=0.02)  # same axes as the true camera pose
    assert abs(T[2, 3] - 0.7) < 0.01  # camera height above the desk
    centre = touches_c.mean(0) @ T[:3, :3].T + T[:3, 3]
    assert np.allclose(centre, [0.0, 0.3, 0.0], atol=1e-6)

    from egohand_live.calibrate import spread_hint

    assert spread_hint(touches_c) is None  # an area: plane determined
    row_w = np.array([[x, 0.3, 0.0] for x in (-0.25, -0.1, 0.05, 0.2, 0.3)])  # one row across the desk
    row_c = (row_w - cam) @ R_wc
    assert spread_hint(row_c) == "nearer to you or further away"
    col_c = (np.array([[0.0, y, 0.0] for y in (0.15, 0.25, 0.35, 0.45, 0.55)]) - cam) @ R_wc
    assert spread_hint(col_c) == "further left or right"


def test_desk_pose_from_board():
    from egohand_live.calibrate import board_pose, desk_frame_from_points

    # Camera 0.9 m above the desk, looking down 50 deg; a 9x6 board (12 mm squares) lying on it.
    t = np.radians(50)
    R_wc = np.array([[1, 0, 0], [0, -np.sin(t), np.cos(t)], [0, -np.cos(t), -np.sin(t)]])
    cam = np.array([0.0, -0.4, 0.9])
    K = np.array([[1174.0, 0, 640], [0, 1174, 360], [0, 0, 1]])
    yaw = np.radians(20)  # board turned on the desk; must not matter
    sq = 0.012
    board_w = np.array([[c * sq, r * sq, 0.0] for r in range(6) for c in range(9)])
    board_w = board_w @ np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    board_w += [0.0, 0.3, 0.0]
    pc = (board_w - cam) @ R_wc
    uv = (pc @ K.T)[:, :2] / pc[:, 2:3] + np.random.default_rng(1).normal(0, 0.1, (54, 2))
    obj = np.array([[c * sq, r * sq, 0.0] for r in range(6) for c in range(9)])
    P = board_pose(obj, uv.astype(np.float32), K)
    assert np.abs(P - pc).max() < 0.005
    T, rms = desk_frame_from_points(P)
    assert rms < 1e-3
    assert np.allclose(T[:3, :3], R_wc, atol=0.01)
    assert abs(T[2, 3] - 0.9) < 0.005


def test_depth_scale_from_resting_hands():
    """Hands resting on a known desk, tracked 13 % too close: calibrate-depth recovers 1/0.87."""
    import tempfile
    import threading
    import time as _time
    from pathlib import Path
    from types import SimpleNamespace

    from egohand_live.calibrate import PALM_JOINTS, PALM_THICKNESS, depth_scale_from_hands
    from egohand_live.protocol import Publisher
    from egohand_live.synthetic import DEFAULT_T_WORLD_CAMERA as T, IMAGE_SIZE, K, synthetic_hand
    from egohand_live.skeleton import soma25_to_openxr26

    R_wc, t_wc = T[:3, :3], T[:3, 3]
    spots = [("left", -0.12, 0.25), ("right", 0.10, 0.30), ("left", 0.0, 0.42), ("right", 0.15, 0.40)]
    port = 5600 + int(_time.time() * 1000) % 300
    stop = threading.Event()
    ext = Path(tempfile.mkdtemp()) / "T.txt"
    np.savetxt(ext, T)

    def hand_msg(side, x, y, t, seq):
        j = synthetic_hand(side, 0.0)
        palm = soma25_to_openxr26(j, side)[0][PALM_JOINTS].mean(0)
        j = j + np.array([x, y, PALM_THICKNESS]) - palm  # palm underside on the desk (z = 0)
        jc = (j - t_wc) @ R_wc
        uv = jc @ K.T
        p, q, r = soma25_to_openxr26(jc * 0.87, side)  # monocular depth 13 % short, same rays
        h = SimpleNamespace(side=side, score=0.9, held=False, bbox=np.zeros(3, np.float32),
                            joints_cam=(jc * 0.87).astype(np.float32),
                            keypoints_2d=(uv[:, :2] / uv[:, 2:]).astype(np.float32),
                            openxr_positions=p, openxr_orientations=q, openxr_radii=r)  # fmt: skip
        return SimpleNamespace(index=seq, t_capture=t, image_size=IMAGE_SIZE, K=K.astype(np.float32),
                               timings_ms={}, hands={side: h}, depth_scale=1.0)  # fmt: skip

    def publish():
        pub = Publisher(f"tcp://127.0.0.1:{port}")
        t, seq = 0.0, 0
        while not stop.is_set():
            for side, x, y in spots:
                for _ in range(60):  # 2 s still at each spot (synthetic clock, 30 fps)
                    if stop.is_set():
                        break
                    pub.publish(hand_msg(side, x, y, t, seq))
                    t, seq = t + 1 / 30, seq + 1
                    _time.sleep(0.002)
        pub.close()

    th = threading.Thread(target=publish, daemon=True)
    th.start()
    try:
        scale = depth_scale_from_hands(f"tcp://127.0.0.1:{port}", str(ext), spots=3)
    finally:
        stop.set()
        th.join(2)
    assert abs(scale - 1 / 0.87) < 0.01


def test_gripper_teleop_pose_and_pinch():
    from scipy.spatial.transform import Rotation as R_

    from egohand_live.gripper_teleop import PinchGripper, desk_to_task, ee_target, transform_hand
    from egohand_live.skeleton import soma25_to_openxr26
    from egohand_live.synthetic import synthetic_hand

    T = desk_to_task(task_centre=(0.5, 0.0, 0.0))
    assert np.allclose(T[:3, :3] @ [0, 0.3, 0] + T[:3, 3], [0.5, 0, 0])  # desk centre -> task centre
    for curl, side in ((0.0, "right"), (1.0, "right"), (0.0, "left")):
        j = synthetic_hand(side, curl) + [0.05, 0.3, 0.06]  # palm down, fingers away from the operator
        p, q, _ = soma25_to_openxr26(j, side)
        pt, qt = transform_hand(T, p, q)
        for mode in ("yaw", "full"):
            pos, quat_xyzw, width = ee_target(pt, qt, rotation=mode)
            Rm = R_.from_quat(quat_xyzw).as_matrix()
            # Franka top grasp: x forward (task +x), approach z down
            assert Rm[:, 0] @ [1, 0, 0] > 0.97 and Rm[:, 2] @ [0, 0, -1] > 0.97, (side, curl, mode)
            tcp = 0.5 * (pt[5] + pt[10])
            assert np.allclose(pos, [tcp[0], tcp[1], max(tcp[2], 0.005) + 0.1034], atol=0.02)
        assert (width < 0.02) == (curl == 1.0)
    g = PinchGripper()
    assert [g(w) for w in (0.08, 0.04, 0.02, 0.04, 0.06)] == [1.0, 1.0, -1.0, -1.0, 1.0]  # hysteresis
    c = PinchGripper("continuous")
    assert c(0.0) == 0.0 and c(0.2) == 1.0 and 0.4 < c(0.0475) < 0.6


def test_session_roundtrip_and_video_timeline():
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from egohand_live.session import SessionWriter, load_session
    from egohand_live.video import frame_indices, timeline

    d = tempfile.mkdtemp()
    hand = SimpleNamespace(side="left", score=0.9, held=False, bbox=np.ones(3), joints_cam=np.ones((25, 3)),
                           keypoints_2d=np.ones((25, 2)), openxr_positions=np.ones((26, 3)),
                           openxr_orientations=np.ones((26, 4)), vertices_cam=np.ones((10, 3)))
    w = SessionWriter(d, {"image_size": [64, 36]}, faces={"left": np.zeros((4, 3), int)})
    for i in range(5):
        res = SimpleNamespace(index=i, t_capture=10 + i / 15, K=np.eye(3), timings_ms={"total": 1.0},
                              hands={"left": hand})
        w.add(np.zeros((36, 64, 3), np.uint8), res)
    w.close()
    meta, faces, recs = load_session(d)
    assert meta["image_size"] == [64, 36] and faces["left"].shape == (4, 3) and len(recs) == 5
    assert recs[2]["hands"][0]["verts"].shape == (10, 3) and Path(recs[2]["frame"]).exists()
    idx = frame_indices(np.array([r["t"] for r in recs]), fps=30)  # 15 fps recording -> 30 fps video
    assert list(idx[:4]) == [0, 0, 1, 1]
    plan = timeline(300, 30.0, [("a",), ("b",), ("c",)], step_s=4.0)  # 10 s, 3 steps
    assert plan[0] == (0, None, 1.0) and plan[-1][0] == 2
    assert any(p[1] == 0 and 0 < p[2] < 1 for p in plan)  # a wipe from step 0 to step 1
