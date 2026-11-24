"""egohand-live command line.

    egohand-live run --source realsense --show --publish tcp://*:5555
    egohand-live run --source video:clip.mp4 --record out.mp4
    egohand-live camera-server --source realsense --bind tcp://*:5556   (on the camera host)
    egohand-live view --connect tcp://GPU_HOST:5555                      (remote viewer)
    egohand-live retarget --connect tcp://GPU_HOST:5555 --robot sharpa_wave (robot joints)
    egohand-live slim-ckpt --out checkpoints/egohand_slim.ckpt
"""

import argparse
import logging
import os
import signal
import time

import numpy as np

log = logging.getLogger("egohand_live")


def _add_source_args(p):
    p.add_argument("--source", default="realsense",
                   help="realsense[:SERIAL] | webcam:N | video:PATH | tcp://HOST:PORT (camera-server "
                        "binds) | tcp://*:PORT (listen; camera-server --connect)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--intrinsics", help="EgoPost PINHOLE camera JSON (overrides source K)")
    p.add_argument("--no-realtime", action="store_true",
                   help="video files: process every frame as fast as possible")
    p.add_argument("--loop", action="store_true", help="video files: loop forever")
    p.add_argument("--scale", type=float, default=1.0, help="downscale frames (e.g. 0.5) before tracking")
    p.add_argument("--zoom", type=float, default=1.0,
                   help="undistortion only: output focal multiplier (<1 keeps more of a wide FOV)")  # fmt: skip
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="rotate frames clockwise, e.g. 180 when your hands enter from the top of the image")  # fmt: skip


def _percentiles(values):
    a = np.asarray(values)
    return f"mean {a.mean():6.1f}  p50 {np.percentile(a, 50):6.1f}  p95 {np.percentile(a, 95):6.1f}"


def cmd_run(args):
    from .engine import EgoHandEngine, EngineConfig
    from .filters import SmoothingConfig
    from .sources import open_source
    from .tracker import TrackerConfig
    from .viz import RateMeter, draw_overlay

    hands = ("left", "right") if args.hands == "both" else (args.hands,)
    cfg = EngineConfig(
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        hands=hands,
        with_mesh=bool(args.rerun or args.save_session),
        tracker=TrackerConfig(box_scale=args.box_scale, score_new=args.det_threshold,
                              handedness=args.handedness, min_confidence=args.min_confidence),
        smoothing=SmoothingConfig(enabled=not args.no_smooth),
        depth_scale=args.depth_scale,
    )
    engine = EgoHandEngine(cfg)
    source = open_source(args.source, width=args.width, height=args.height, fps=args.fps,
                         intrinsics=args.intrinsics, realtime=not args.no_realtime,
                         loop=args.loop, scale=args.scale, zoom=args.zoom,
                         rotate=args.rotate)  # fmt: skip
    first = source.read()
    if first is None:
        raise SystemExit("source produced no frames")
    log.info("Warming up (%dx%d)...", first.bgr.shape[1], first.bgr.shape[0])
    engine.warmup(first.bgr.shape[1], first.bgr.shape[0])
    source.rebase_clock()

    publisher = rr_logger = writer = None
    if args.publish:
        from .protocol import Publisher

        T = np.loadtxt(args.extrinsics).reshape(4, 4) if args.extrinsics else None
        publisher = Publisher(args.publish, T_world_camera=T)
        log.info("Publishing hands on %s%s", args.publish, " (+frames)" if args.publish_frames else "")
    if args.rerun:
        from .viz import RerunLogger

        rr_logger = RerunLogger(args.rerun, faces=engine.faces)
    session = None
    if args.save_session:
        from .session import SessionWriter

        session = SessionWriter(args.save_session, {
            "source": args.source,
            "image_size": [int(first.bgr.shape[1]), int(first.bgr.shape[0])],
            "T_world_camera": np.loadtxt(args.extrinsics).reshape(4, 4).tolist() if args.extrinsics else None,
            "depth_scale": args.depth_scale,
            "intrinsics": args.intrinsics,
            "rotate": args.rotate,
        }, faces=engine.faces)  # fmt: skip
        log.info("Saving the session to %s", args.save_session)
    if args.show:
        import cv2

        cv2.namedWindow("egohand-live", cv2.WINDOW_NORMAL)

    stop = False

    def _sigint(*_):
        nonlocal stop
        if stop:  # second Ctrl+C: abort even if blocked waiting for a frame
            raise KeyboardInterrupt
        stop = True

    signal.signal(signal.SIGINT, _sigint)
    rate = RateMeter()
    stats = {"detect": [], "egohand": [], "post": [], "total": [], "latency": [], "hands": []}
    n = 0
    last_print = time.time()
    frame = first
    while not stop and frame is not None:
        result = engine.process(frame.bgr, frame.K, frame.t)
        now = time.time()
        latency = (now - frame.t) * 1e3 if args.source_is_live else None
        rate.tick(now)
        if n >= 5:  # skip warm-up transients in the summary
            for k in ("detect", "egohand", "post", "total"):
                stats[k].append(result.timings_ms[k])
            if latency is not None:
                stats["latency"].append(latency)
            stats["hands"].append(len(result.hands))
        if publisher:
            publisher.publish(result, frame.bgr if args.publish_frames else None)
        if rr_logger:
            rr_logger.log(frame.bgr, result)
        if session:
            session.add(frame.bgr, result)
        if args.show or writer is not None or args.record:
            vis = draw_overlay(frame.bgr, result.hands, rate.fps, latency,
                               {k: result.timings_ms[k] for k in ("detect", "egohand")})  # fmt: skip
            if args.record:
                if writer is None:
                    import cv2

                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.record_fps, (vis.shape[1], vis.shape[0]))  # fmt: skip
                writer.write(vis)
            if args.show:
                import cv2

                cv2.imshow("egohand-live", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        if now - last_print > 2.0:
            last_print = now
            log.info("%5.1f FPS | %d hand(s) | detect %.1f ms, egohand %.1f ms, total %.1f ms%s",
                     rate.fps, len(result.hands), result.timings_ms["detect"],
                     result.timings_ms["egohand"], result.timings_ms["total"],
                     f", capture->result {latency:.1f} ms" if latency is not None else "")  # fmt: skip
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        frame = source.read()

    source.close()
    if session:
        session.close()
        print(f"session saved to {args.save_session} ({session.written} frames, {session.dropped} dropped); "
              f"make a video with: egohand-live render-video {args.save_session}")
    if writer is not None:
        writer.release()
    if publisher:
        publisher.close()
    if stats["total"]:
        print(f"\n=== {n} frames, {rate.fps:.1f} FPS (last 60), "
              f"mean hands/frame {np.mean(stats['hands']):.2f} ===")  # fmt: skip
        for k in ("detect", "egohand", "post", "total", "latency"):
            if stats[k]:
                print(f"{k:>8} ms: {_percentiles(stats[k])}")
        skipped = getattr(source, "skipped", None)
        if skipped:
            print(f"dropped source frames (consumer slower than camera): {skipped}")


def cmd_camera_server(args):
    import zmq

    from .protocol import bind_or_connect, encode_frame
    from .sources import open_source

    source = open_source(args.source, width=args.width, height=args.height, fps=args.fps,
                         intrinsics=args.intrinsics, realtime=not args.no_realtime,
                         loop=args.loop, scale=args.scale, zoom=args.zoom,
                         rotate=args.rotate)  # fmt: skip
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)
    endpoint = args.connect or args.bind
    mode = bind_or_connect(sock, endpoint)
    log.info("Streaming %s %s %s (JPEG q=%d)", args.source,
             "on" if mode == "bind" else "to", endpoint, args.quality)  # fmt: skip
    n, t0 = 0, time.time()
    try:
        while (frame := source.read()) is not None:
            sock.send_multipart(encode_frame(frame.bgr, frame.K, frame.t, frame.index, args.quality))
            n += 1
            if time.time() - t0 > 5:
                log.info("%.1f FPS", n / (time.time() - t0))
                n, t0 = 0, time.time()
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        sock.close(linger=0)


def cmd_view(args):
    import cv2

    from .protocol import Subscriber
    from .viz import RateMeter, draw_overlay

    sub = Subscriber(args.connect, frames=True)
    cv2.namedWindow("egohand-live view", cv2.WINDOW_NORMAL)
    rate = RateMeter()
    hands_msg = None
    log.info("Viewing %s (start the GPU side with --publish ... --publish-frames)", args.connect)
    while True:
        msg, frame = sub.latest(100)
        if msg is not None:
            hands_msg = msg
            rate.tick(time.time())
        if frame is not None:
            meta, bgr = frame
            hands = {h["side"]: h for h in hands_msg["hands"]} if hands_msg else {}
            latency = (time.time() - hands_msg["t_capture"]) * 1e3 if hands_msg else None
            timings = {k: v for k, v in (hands_msg or {}).get("latency_ms", {}).items()
                       if k in ("detect", "egohand")}  # fmt: skip
            cv2.imshow("egohand-live view", draw_overlay(bgr, hands, rate.fps, latency, timings))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    sub.close()


def cmd_retarget(args):
    try:
        import dex_retargeting  # noqa: F401
        import torch  # noqa: F401
        import yaml  # noqa: F401

        from .retarget import serve
    except ImportError as exc:
        raise SystemExit(f"{exc}: retargeting needs dex-retargeting + torch (pip install -e '.[retarget]'), "
                         "or run with Isaac Sim's python.sh, which bundles them") from exc  # fmt: skip
    serve(args.connect, args.publish, robot=args.robot, low_pass_alpha=args.lowpass)


def cmd_slim_ckpt(args):
    from .engine import slim_checkpoint

    out = slim_checkpoint(args.checkpoint, args.out)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


def cmd_calibrate_desk(args):
    from .calibrate import live_view, run

    if args.live:
        live_view(args.serial, d405_mount=args.d405_mount, out=args.out)
    else:
        run(args.serial, args.frames, d405_mount=args.d405_mount, out=args.out, preview=args.preview)


def cmd_calibrate_desk_hands(args):
    from .calibrate import desk_from_hands

    desk_from_hands(args.connect, args.out, args.spots)


def cmd_calibrate_desk_board(args):
    from .calibrate import desk_from_board

    desk_from_board(args.source, args.out, args.board, args.intrinsics, args.rotate, args.board_height_mm,
                    args.frames, args.width, args.height)  # fmt: skip


def cmd_calibrate_depth(args):
    from .calibrate import depth_scale_from_hands

    depth_scale_from_hands(args.connect, args.extrinsics, args.spots)


def cmd_calibrate_camera(args):
    from .camcal import run

    run(args.source, args.out, args.width, args.height, args.fps, args.board, args.views,
        show_pattern=args.show_pattern, write_pattern=args.write_pattern, model=args.model)  # fmt: skip


def cmd_preview(args):
    """Live camera view with framing guides, for aiming a camera before tracking."""
    import cv2

    from .sources import open_source

    rotations = [0, 90, 180, 270]
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    rot = args.rotate
    src = open_source(args.source, width=args.width, height=args.height, fps=args.fps,
                      intrinsics=args.intrinsics, zoom=args.zoom)  # fmt: skip
    win = "egohand-live preview (r: rotate, s: snapshot, q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    stamps: list[float] = []
    try:
        while True:
            f = src.read()
            if f is None:
                continue
            img = cv2.rotate(f.bgr, codes[rot]) if rot else f.bgr.copy()
            h, w = img.shape[:2]
            stamps = [t for t in stamps if f.t - t < 1.0] + [f.t]
            for i in (1, 2):  # thirds
                cv2.line(img, (w * i // 3, 0), (w * i // 3, h), (200, 200, 200), 1, cv2.LINE_AA)
                cv2.line(img, (0, h * i // 3), (w, h * i // 3), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.drawMarker(img, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
            cv2.rectangle(img, (0, h - 12), (w, h), (0, 200, 0), -1)
            lines = [
                f"{w}x{h}  {len(stamps)} fps  --rotate {rot}",
                "aim: desk work area in the middle, your forearms entering at the green (bottom) edge",
            ]
            for i, text in enumerate(lines):
                y = 36 + 34 * i
                cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                rot = rotations[(rotations.index(rot) + 1) % 4]
            elif key == ord("s"):
                path = f"outputs/preview_{time.strftime('%H%M%S')}.jpg"
                os.makedirs("outputs", exist_ok=True)
                cv2.imwrite(path, cv2.rotate(f.bgr, codes[rot]) if rot else f.bgr)
                print(f"saved {path}")
            elif key in (ord("q"), 27):
                break
    finally:
        src.close()
        cv2.destroyAllWindows()
    print(f"use --rotate {rot} with egohand-live run")


def cmd_render_video(args):
    from .video import render

    render(args.session, args.out, args.size, args.fps, args.layers, args.step_seconds, args.aspect,
           not args.no_captions, args.start, args.end)  # fmt: skip


def cmd_synthetic(args):
    import numpy as np

    from .synthetic import serve

    T = np.loadtxt(args.extrinsics).reshape(4, 4) if args.extrinsics else None
    sides = ("left", "right") if args.hands == "both" else (args.hands,)
    serve(args.publish, T, args.fps, sides, args.seconds)


def main(argv=None):
    p = argparse.ArgumentParser(prog="egohand-live", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="track hands from a camera/video")
    _add_source_args(r)
    r.add_argument("--checkpoint", help="EgoHand .ckpt (default: HF nvidia/EgoHand-v1)")
    r.add_argument("--device", default="cuda")
    r.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    r.add_argument("--compile", action="store_true", help="torch.compile DINOv3 blocks (slow first start)")
    r.add_argument("--hands", default="both", choices=["both", "left", "right"])
    r.add_argument("--box-scale", type=float, default=1.3, help="crop side / detector box side")
    r.add_argument("--det-threshold", type=float, default=0.3, help="score to start a hand track")
    r.add_argument("--min-confidence", type=float, default=0.45,
                   help="publish a hand only while its track confidence (EMA of detector scores) is "
                        "at least this; lower it for clips where real hands score low")
    r.add_argument("--handedness", choices=("position", "detector"), default="position",
                   help="left/right from image position (camera behind/above the operator; robust "
                        "to palm-facing hands) or from the detector's class label")
    r.add_argument("--depth-scale", type=float, default=1.0,
                   help="move hands along their viewing ray to this x their estimated distance "
                        "(monocular depth bias; measure with calibrate-depth)")
    r.add_argument("--no-smooth", action="store_true", help="disable One Euro smoothing")
    r.add_argument("--show", action="store_true", help="OpenCV window with overlay")
    r.add_argument("--record", help="write overlay video to this .mp4")
    r.add_argument("--record-fps", type=float, default=30.0)
    r.add_argument("--save-session", metavar="DIR",
                   help="save frames + per-frame results (joints, hand mesh) to render videos later "
                        "(egohand-live render-video DIR)")
    r.add_argument("--rerun", nargs="?", const="spawn",
                   help="Rerun: 'spawn', rerun+http://HOST:9876/proxy, or file.rrd")
    r.add_argument("--publish", nargs="?", const="tcp://*:5555", help="ZMQ PUB endpoint")
    r.add_argument("--publish-frames", action="store_true", help="also publish JPEG frames")
    r.add_argument("--extrinsics", help="4x4 T_world_camera text file, sent with each message")
    r.add_argument("--max-frames", type=int, default=0)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("camera-server", help="stream a local camera to a remote GPU host")
    _add_source_args(c)
    c.add_argument("--bind", default="tcp://*:5556", help="listen here; GPU host uses --source tcp://THIS_HOST:5556")
    c.add_argument("--connect", help="instead connect to a GPU host listening with --source tcp://*:5556 "
                                     "(use when this machine's firewall blocks incoming connections)")
    c.add_argument("--quality", type=int, default=85)
    c.set_defaults(func=cmd_camera_server)

    v = sub.add_parser("view", help="view results published by a remote `run --publish-frames`")
    v.add_argument("--connect", default="tcp://localhost:5555")
    v.set_defaults(func=cmd_view)

    rt = sub.add_parser("retarget", help="retarget published hands to robot hand joints (e.g. Sharpa Wave)")
    rt.add_argument("--connect", default="tcp://localhost:5555", help="hands publisher (run --publish)")
    rt.add_argument("--publish", default="tcp://*:5557", help="bind address for robot-hands messages")
    rt.add_argument("--robot", default="sharpa_wave")
    rt.add_argument("--lowpass", type=float, default=0.5, help="retargeter low-pass alpha (1 = off)")
    rt.set_defaults(func=cmd_retarget)

    cd = sub.add_parser("calibrate-desk", help="camera pose over the desk from RealSense depth (T_world_camera)")
    cd.add_argument("--serial", help="RealSense serial (default: first camera)")
    cd.add_argument("--frames", type=int, default=30, help="depth frames to average")
    cd.add_argument("--d405-mount", action="store_true",
                    help="camera on hardware/d405_desk_mount: origin under the bar, prints collar height + tilt")
    cd.add_argument("--out", help="write the 4x4 T_world_camera here (for run/sharpa_hands --extrinsics)")
    cd.add_argument("--preview", help="save the colour image with desk-plane points marked")
    cd.add_argument("--live", action="store_true",
                    help="live view for setting up the camera: pose readout, sim desk outline, s saves --out")
    cd.set_defaults(func=cmd_calibrate_desk)

    ch = sub.add_parser("calibrate-desk-hands",
                        help="camera pose over the desk from hand touches (any camera, no target needed)")
    ch.add_argument("--connect", default="tcp://localhost:5555", help="running `egohand-live run --publish`")
    ch.add_argument("--out", default="outputs/T_world_camera.txt")
    ch.add_argument("--spots", type=int, default=6, help="minimum spots (more are asked for until they cover an area)")
    ch.set_defaults(func=cmd_calibrate_desk_hands)

    cb = sub.add_parser("calibrate-desk-board",
                        help="camera pose over the desk from a checkerboard lying on it (phone or print)")
    _add_source_args(cb)
    cb.add_argument("--board", required=True,
                    help="board with its size: charuco:COLSxROWS:SQUARE_MM:MARKER_MM:DICT, charuco:auto:SQUARE_MM, "
                         "or checker:COLSxROWS:SQUARE_MM (inner corners)")
    cb.add_argument("--board-height-mm", type=float, default=0.0,
                    help="board surface above the desk (printed board 0, phone lying on its back ~8)")
    cb.add_argument("--frames", type=int, default=30, help="views to average")
    cb.add_argument("--out", default="outputs/T_world_camera.txt")
    cb.set_defaults(func=cmd_calibrate_desk_board)

    cz = sub.add_parser("calibrate-depth",
                        help="monocular depth scale (run --depth-scale) from hands resting on the desk")
    cz.add_argument("--connect", default="tcp://localhost:5555", help="running `egohand-live run --publish`")
    cz.add_argument("--extrinsics", required=True, help="T_world_camera from calibrate-desk-board")
    cz.add_argument("--spots", type=int, default=6)
    cz.set_defaults(func=cmd_calibrate_depth)

    cc = sub.add_parser("calibrate-camera", help="lens intrinsics from a checkerboard or ChArUco board -> --intrinsics JSON")
    cc.add_argument("--source", default="webcam:0", help="webcam:N | realsense | video:PATH")
    cc.add_argument("--width", type=int, default=1280)
    cc.add_argument("--height", type=int, default=720)
    cc.add_argument("--fps", type=int, default=30)
    cc.add_argument("--out", default="outputs/camera_intrinsics.json")
    cc.add_argument("--board", default="checker:9x6",
                    help="checker:COLSxROWS (inner corners; phone or print), charuco:COLSxROWS:SQUARE_MM:MARKER_MM:DICT "
                         "or charuco:auto (printed ChArUco; covers the image corners of wide lenses)")
    cc.add_argument("--model", default="auto", choices=("auto", "standard", "rational", "kb4"),
                    help="lens model; auto fits all and keeps the best (kb4 = fisheye)")
    cc.add_argument("--views", type=int, default=20, help="views to capture")
    cc.add_argument("--show-pattern", action="store_true",
                    help="show the checkerboard full-screen here (hold or tilt this screen under the camera)")
    cc.add_argument("--write-pattern", help="also save the checkerboard as an image (for a phone or printer)")
    cc.set_defaults(func=cmd_calibrate_camera)

    pv = sub.add_parser("preview", help="live camera view with framing guides (for aiming the camera)")
    _add_source_args(pv)
    pv.set_defaults(func=cmd_preview)

    rv = sub.add_parser("render-video", help="render a saved session (run --save-session) as a step-by-step video")
    rv.add_argument("session", help="session folder")
    rv.add_argument("--out", help="output .mp4 (default: SESSION/video.mp4)")
    rv.add_argument("--size", default="1920x1080")
    rv.add_argument("--fps", type=float, default=30.0)
    rv.add_argument("--layers", help="fixed layer stack instead of the timeline, e.g. camera,mesh "
                                     "(camera, skeleton, pose3d, mesh)")
    rv.add_argument("--step-seconds", type=float, default=4.0, help="duration of each timeline step")
    rv.add_argument("--aspect", default="16:9", choices=["16:9", "1:1", "9:16"], help="crop around the hands")
    rv.add_argument("--no-captions", action="store_true")
    rv.add_argument("--start", type=float, default=0.0, help="skip this many seconds")
    rv.add_argument("--end", type=float, help="stop at this many seconds")
    rv.set_defaults(func=cmd_render_video)

    sy = sub.add_parser("synthetic", help="publish synthetic moving/grasping hands (test the sim without a camera)")
    sy.add_argument("--publish", default="tcp://*:5555", help="ZMQ endpoint (bind with * or connect)")
    sy.add_argument("--extrinsics", help="4x4 T_world_camera to express the hands in (default: built-in)")
    sy.add_argument("--fps", type=float, default=30.0)
    sy.add_argument("--hands", choices=("both", "left", "right"), default="both")
    sy.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until Ctrl+C)")
    sy.set_defaults(func=cmd_synthetic)

    s = sub.add_parser("slim-ckpt", help="write a weights-only checkpoint (faster loading)")
    s.add_argument("--checkpoint", help="source .ckpt (default: HF)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_slim_ckpt)

    args = p.parse_args(argv)
    # opencv-python's bundled Qt has no fonts, and importing cv2 points QT_QPA_FONTDIR at that
    # empty folder. Import it first, then point Qt at the system fonts (or the user's choice).
    fontdir = os.environ.get("QT_QPA_FONTDIR")
    if not fontdir and os.path.isdir("/usr/share/fonts/truetype/dejavu"):
        fontdir = "/usr/share/fonts/truetype/dejavu"
    if fontdir:
        try:
            import cv2  # noqa: F401
        except ImportError:
            pass
        os.environ["QT_QPA_FONTDIR"] = fontdir
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip
    if hasattr(args, "source"):
        args.source_is_live = not (
            getattr(args, "no_realtime", False) and not args.source.startswith(("realsense", "webcam", "tcp://"))
        )
    args.func(args)


if __name__ == "__main__":
    main()
