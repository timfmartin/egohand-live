"""Frame sources: UVC webcam, video file (replayed like a live camera), RealSense, ZMQ.

Every source yields :class:`Frame` objects with pinhole intrinsics. Live sources run a
capture thread that keeps only the newest frame, so a slow consumer drops frames
instead of accumulating latency.

Source specs (``open_source``):
    realsense[:SERIAL]        Intel RealSense color stream, factory intrinsics
    webcam:N | N              OpenCV/V4L2 device N
    video:PATH | PATH.mp4     video file, paced to its frame rate (``--no-realtime`` to disable)
    tcp://HOST:PORT           frames from ``egohand-live camera-server``
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("egohand_live.sources")


@dataclass
class Frame:
    bgr: np.ndarray
    K: np.ndarray  # (3, 3) float32 pinhole intrinsics for this image
    t: float  # capture time, seconds (time.time() clock; replayed files use replay time)
    index: int
    depth_m: np.ndarray | None = None  # optional depth aligned to color, meters


def estimate_K(width: int, height: int) -> np.ndarray:
    """EgoPost's fallback when no calibration is given: f = max(width, height)."""
    f = float(max(width, height))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float32)


def load_intrinsics(path: str | Path, width: int, height: int) -> np.ndarray:
    """Load an EgoPost-format PINHOLE camera JSON, rescaled to (width, height)."""
    cal = json.loads(Path(path).read_text())
    model = cal.get("model", "PINHOLE").upper()
    if model != "PINHOLE" or any(abs(c) > 1e-9 for c in cal.get("distortion_coefficients", [])):
        log.warning("Calibration %s is %s with distortion; only K is used here.", path, model)
    K = np.asarray(cal["camera_matrix"]["matrix_3x3"], np.float32).copy()
    sx, sy = width / cal["image_width"], height / cal["image_height"]
    K[0] *= sx
    K[1] *= sy
    return K


class _LatestFrameThread:
    """Run ``grab()`` in a thread; ``read()`` returns the newest unseen frame."""

    def __init__(self, grab, name):
        self._grab = grab
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._last_index = -1
        self._running = True
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            while self._running:
                frame = self._grab()
                with self._cond:
                    self._frame = frame
                    self._cond.notify_all()
                if frame is None:
                    return
        except BaseException as exc:  # surfaced to the consumer in read()
            with self._cond:
                self._error = exc
                self._frame = None
                self._cond.notify_all()

    def read(self, timeout: float = 5.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._running:
                if self._error is not None:
                    raise RuntimeError("frame source failed") from self._error
                f = self._frame
                if f is None and not self._thread.is_alive():
                    return None
                if f is not None and f.index != self._last_index:
                    self._last_index = f.index
                    return f
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no frame from source within %.1fs" % timeout)
                self._cond.wait(remaining)
        return None

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)


class Source:
    def read(self) -> Frame | None:
        raise NotImplementedError

    def rebase_clock(self):
        """Replayed files: restart pacing from now (call after a slow warm-up)."""

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class WebcamSource(Source):
    def __init__(self, device: int, width=1280, height=720, fps=30, intrinsics=None):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open webcam {device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.K = load_intrinsics(intrinsics, w, h) if intrinsics else estimate_K(w, h)
        log.info("Webcam %d: %dx%d @ %.0f fps", device, w, h, self.cap.get(cv2.CAP_PROP_FPS))
        self._i = 0
        self._reader = _LatestFrameThread(self._grab, "webcam")

    def _grab(self):
        ok, img = self.cap.read()
        if not ok:
            return None
        self._i += 1
        return Frame(img, self.K, time.time(), self._i)

    def read(self):
        return self._reader.read()

    def close(self):
        self._reader.stop()
        self.cap.release()


class VideoFileSource(Source):
    """Replay a file. With ``realtime`` the stream is paced to the file's fps and frames
    the consumer is too slow for are skipped, exactly like a live camera would."""

    def __init__(self, path, intrinsics=None, realtime=True, loop=False):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.K = load_intrinsics(intrinsics, w, h) if intrinsics else estimate_K(w, h)
        self.realtime, self.loop = realtime, loop
        self._pos = 0  # index of the next frame the decoder will return
        self._start = None
        self.skipped = 0
        log.info("Video %s: %dx%d @ %.2f fps (realtime=%s)", path, w, h, self.fps, realtime)

    def _next(self):
        ok, img = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._pos, self._start = 0, time.time()
            ok, img = self.cap.read()
        if not ok:
            return None
        self._pos += 1
        return img

    def read(self):
        if not self.realtime:
            img = self._next()
            return None if img is None else Frame(img, self.K, self._pos / self.fps, self._pos)
        now = time.time()
        if self._start is None:
            self._start = now
        due = int((now - self._start) * self.fps)  # newest frame that has "arrived"
        while self._pos < due:  # drop frames we were too slow for
            if not self.cap.grab():
                if not self.loop:
                    return None
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._pos, self._start = 0, time.time()
                break
            self._pos += 1
            self.skipped += 1
        t_frame = self._start + self._pos / self.fps
        if t_frame > now:
            time.sleep(t_frame - now)  # wait for the frame to "arrive"
        img = self._next()
        if img is None:
            return None
        return Frame(img, self.K, self._start + (self._pos - 1) / self.fps, self._pos)

    def rebase_clock(self):
        if self._start is not None:
            self._start = time.time() - self._pos / self.fps

    def close(self):
        self.cap.release()


class RealSenseSource(Source):
    """RealSense color stream (factory intrinsics), optionally with aligned depth.

    Waits up to ``reconnect_s`` seconds for the camera at start-up, and if it drops
    off the bus later (loose cable, USB link reset) restarts the pipeline on the same
    serial with the same budget before giving up.
    """

    def __init__(self, serial=None, width=1280, height=720, fps=30, depth=False, reconnect_s=30.0):
        import pyrealsense2 as rs

        self.rs = rs
        self.serial, self.width, self.height, self.fps, self.depth = serial, width, height, fps, depth
        self.reconnect_s = reconnect_s
        dev, intr = self._connect()
        log.info(
            "RealSense %s (%s): %dx%d @ %d fps, fx=%.1f fy=%.1f, distortion=%s %s",
            dev.get_info(rs.camera_info.name), self.serial,
            intr.width, intr.height, fps, intr.fx, intr.fy, intr.model, list(intr.coeffs),
        )  # fmt: skip
        self._i = 0
        self._reader = _LatestFrameThread(self._grab, "realsense")

    def _start(self):
        rs = self.rs
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.depth:
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.pipe = rs.pipeline()
        profile = self.pipe.start(cfg)
        dev = profile.get_device()
        self.serial = dev.get_info(rs.camera_info.serial_number)  # reconnect to this camera
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], np.float32
        )
        self.align = rs.align(rs.stream.color) if self.depth else None
        self.depth_scale = dev.first_depth_sensor().get_depth_scale() if self.depth else None
        return dev, intr

    def _connect(self):
        deadline = time.monotonic() + self.reconnect_s
        warned = False
        while True:
            try:
                return self._start()
            except RuntimeError as e:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"RealSense {self.serial or '(any)'} not available after "
                                       f"{self.reconnect_s:.0f} s: {e}") from e  # fmt: skip
                if not warned:
                    log.warning("RealSense not ready (%s); retrying for %.0f s...", e, self.reconnect_s)
                    warned = True
                time.sleep(1.0)

    def _restart(self, exc):
        log.warning("RealSense stream lost (%s); reconnecting...", exc)
        try:
            self.pipe.stop()
        except RuntimeError:
            pass
        time.sleep(1.0)
        self._connect()
        log.info("RealSense %s reconnected", self.serial)

    def _grab(self):
        while True:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms=2000)
                break
            except RuntimeError as exc:
                self._restart(exc)
        if self.align is not None:
            frames = self.align.process(frames)
        color = frames.get_color_frame()
        depth = None
        if self.align is not None:
            d = frames.get_depth_frame()
            depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
        self._i += 1
        return Frame(np.asanyarray(color.get_data()).copy(), self.K, time.time(), self._i, depth)

    def read(self):
        return self._reader.read(timeout=self.reconnect_s + 5.0)

    def close(self):
        self._reader.stop()
        self.pipe.stop()


class ZmqSource(Source):
    """Receive JPEG frames + intrinsics from ``egohand-live camera-server``.

    ``tcp://HOST:PORT`` connects to a camera server that binds; ``tcp://*:PORT`` listens
    for a camera server started with ``--connect`` (use this when the camera machine
    blocks incoming connections).
    """

    def __init__(self, url: str):
        import zmq

        from .protocol import TOPIC_FRAME, bind_or_connect, decode_frame

        self._decode = decode_frame
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 2)
        self.sock.setsockopt(zmq.SUBSCRIBE, TOPIC_FRAME)
        mode = bind_or_connect(self.sock, url)
        self.url = url
        log.info("%s camera stream %s", "Listening for" if mode == "bind" else "Subscribed to", url)
        self._closing = threading.Event()
        self._reader = _LatestFrameThread(self._grab, "zmq-camera")

    def _grab(self):
        # Poll so close() can stop this thread before the socket is closed (ZMQ sockets
        # must not be closed while another thread is blocked on them).
        while not self._closing.is_set():
            if self.sock.poll(200):
                meta, bgr = self._decode(self.sock.recv_multipart())
                K = np.asarray(meta["K"], np.float32).reshape(3, 3)
                return Frame(bgr, K, float(meta["t"]), int(meta["index"]))
        return None

    def read(self):
        # The camera server may start later or restart; ZMQ reconnects by itself.
        while True:
            try:
                return self._reader.read(timeout=5.0)
            except TimeoutError:
                log.warning("No frames from %s for 5 s; waiting (is camera-server running?)", self.url)

    def close(self):
        self._closing.set()
        self._reader.stop()
        self.sock.close(linger=0)


def _is_distorted(calibration: dict) -> bool:
    return calibration.get("model", "PINHOLE").upper() != "PINHOLE" or any(
        abs(c) > 1e-9 for c in calibration.get("distortion_coefficients", [])
    )


class RectifiedSource(Source):
    """Remap frames to a pinhole image, optionally downscaled.

    For distorted calibrations this is EgoPost's full-image undistortion (output K =
    input K, so wide-angle content outside the pinhole view is cropped; ``zoom`` < 1
    widens the output view). Projection uses EgoPost's own camera models, so every
    model it supports (RATIONAL_POLYNOMIAL, KB4, FISHEYE624) works here.
    """

    def __init__(self, base: Source, width: int, height: int, calibration: dict | None,
                 scale: float = 1.0, zoom: float = 1.0, pending: Frame | None = None):  # fmt: skip
        self.base, self.pending = base, pending
        out_w, out_h = round(width * scale), round(height * scale)
        self.maps = None
        if calibration is not None and _is_distorted(calibration):
            from inference.camera.camera_models import load_camera_calibration

            cal = dict(calibration)
            sx, sy = width / cal["image_width"], height / cal["image_height"]
            K_src = np.asarray(cal["camera_matrix"]["matrix_3x3"], np.float64) * [[sx], [sy], [1]]
            cal.update(image_width=width, image_height=height,
                       camera_matrix={"matrix_3x3": K_src.tolist()})  # fmt: skip
            camera = load_camera_calibration(cal)
            K = np.asarray(camera.K, np.float64).copy()
            K[:2] *= scale
            K[0, 0] *= zoom
            K[1, 1] *= zoom
            map_xy = np.empty((out_h, out_w, 2), np.float32)
            for top in range(0, out_h, 128):
                yy, xx = np.mgrid[top : min(top + 128, out_h), :out_w]
                rays = np.stack(((xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1],
                                 np.ones_like(xx, dtype=np.float64)), axis=-1)  # fmt: skip
                px = camera.project(rays.reshape(-1, 3)).reshape(*rays.shape[:2], 2)
                map_xy[top : top + 128] = np.where(np.isfinite(px).all(-1, keepdims=True), px, -1)
            self.maps = cv2.convertMaps(map_xy, None, cv2.CV_16SC2)
            log.info("Undistorting %s -> PINHOLE %dx%d (fx=%.1f)", calibration["model"], out_w, out_h, K[0, 0])
            self.K = K.astype(np.float32)
        else:
            self.K = None  # pinhole: scale the source's own K per frame
        self.size, self.scale = (out_w, out_h), scale

    def read(self):
        if self.pending is not None:
            f, self.pending = self.pending, None
        else:
            f = self.base.read()
        if f is None:
            return None
        if self.maps is not None:
            f.bgr = cv2.remap(f.bgr, *self.maps, cv2.INTER_LINEAR)
            f.K = self.K
        elif self.scale != 1.0:
            f.bgr = cv2.resize(f.bgr, self.size, interpolation=cv2.INTER_AREA)
            f.K = (f.K * np.array([[self.scale], [self.scale], [1]], np.float32)).astype(np.float32)
        return f

    def rebase_clock(self):
        self.base.rebase_clock()

    def close(self):
        self.base.close()

    def __getattr__(self, name):  # e.g. VideoFileSource.skipped
        return getattr(self.base, name)


_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_K(K: np.ndarray, width: int, height: int, degrees: int) -> np.ndarray:
    """Intrinsics of the image rotated clockwise by ``degrees`` (a rotation about the optical
    axis, so the result is still an exact pinhole camera)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if degrees == 180:  # (u, v) -> (W-1-u, H-1-v)
        new = [[fx, 0, width - 1 - cx], [0, fy, height - 1 - cy]]
    elif degrees == 90:  # (u, v) -> (H-1-v, u)
        new = [[fy, 0, height - 1 - cy], [0, fx, cx]]
    elif degrees == 270:  # (u, v) -> (v, W-1-u)
        new = [[fy, 0, cy], [0, fx, width - 1 - cx]]
    else:
        return K
    return np.array(new + [[0, 0, 1]], np.float32)


class RotatedSource(Source):
    """Rotate frames (and their intrinsics) clockwise by 90, 180 or 270 degrees, e.g. for a
    camera mounted so that the operator's hands enter from the top of the image."""

    def __init__(self, base: Source, degrees: int):
        self.base, self.degrees, self.code = base, degrees, _ROTATIONS[degrees]

    def read(self):
        f = self.base.read()
        if f is None:
            return None
        h, w = f.bgr.shape[:2]
        f.K = rotate_K(np.asarray(f.K, np.float32), w, h, self.degrees)
        f.bgr = cv2.rotate(f.bgr, self.code)
        if f.depth_m is not None:
            f.depth_m = cv2.rotate(f.depth_m, self.code)
        return f

    def rebase_clock(self):
        self.base.rebase_clock()

    def close(self):
        self.base.close()

    def __getattr__(self, name):
        return getattr(self.base, name)


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _open_base(spec, width, height, fps, intrinsics, realtime, loop, depth):
    if spec.startswith("realsense"):
        serial = spec.split(":", 1)[1] if ":" in spec else None
        src = RealSenseSource(serial, width, height, fps, depth=depth)
        if intrinsics:
            src.K = load_intrinsics(intrinsics, width, height)
        return src
    if spec.startswith("tcp://"):
        return ZmqSource(spec)
    if spec.startswith("webcam:") or spec.isdigit():
        return WebcamSource(int(spec.split(":")[-1]), width, height, fps, intrinsics)
    path = spec[len("video:"):] if spec.startswith("video:") else spec
    if path.lower().endswith(VIDEO_EXTS) or Path(path).is_file():
        return VideoFileSource(path, intrinsics, realtime=realtime, loop=loop)
    raise ValueError(f"unrecognised source {spec!r}")


def open_source(spec: str, *, width=1280, height=720, fps=30, intrinsics=None, realtime=True,
                loop=False, depth=False, scale=1.0, zoom=1.0, rotate=0) -> Source:  # fmt: skip
    """Open a source; undistort (if the calibration has distortion), downscale, rotate."""
    src = _open_unrotated(spec, width, height, fps, intrinsics, realtime, loop, depth, scale, zoom)
    return RotatedSource(src, rotate) if rotate else src


def _open_unrotated(spec, width, height, fps, intrinsics, realtime, loop, depth, scale, zoom) -> Source:
    calibration = json.loads(Path(intrinsics).read_text()) if intrinsics else None
    if calibration is None and (spec.startswith("webcam:") or spec.isdigit()):
        log.warning("No --intrinsics for webcam; using f=max(w,h). Depth will be off "
                    "(egohand-live calibrate-camera measures the lens).")
    distorted = calibration is not None and _is_distorted(calibration)
    # Distorted calibrations are applied by RectifiedSource, not as a plain K.
    src = _open_base(spec, width, height, fps, None if distorted else intrinsics,
                     realtime, loop, depth)  # fmt: skip
    if not distorted and scale == 1.0:
        return src
    probe = src.read()  # actual stream size (cameras may not honour the request)
    if probe is None:
        raise RuntimeError(f"source {spec!r} produced no frames")
    h, w = probe.bgr.shape[:2]
    return RectifiedSource(src, w, h, calibration, scale=scale, zoom=zoom, pending=probe)
