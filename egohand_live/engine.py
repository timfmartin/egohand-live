"""Real-time EgoHand engine: RT-DETR -> causal box tracker -> EgoHand -> causal smoothing.

Wraps NVIDIA EgoPost (``inference``/``egohand`` modules from IsaacCapture) without
modifying it. EgoPost's offline-only stages (whole-clip box tracking and the
bidirectional adaptive smoother) are replaced by causal equivalents.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .filters import HandSmoother, SmoothingConfig
from .predictor import LivePredictor, mirrored_faces
from .skeleton import soma25_to_openxr26
from .tracker import HANDS, HandBoxTracker, TrackerConfig

log = logging.getLogger("egohand_live.engine")

EGOHAND_REPO, EGOHAND_FILE = "nvidia/EgoHand-v1", "egohand_s060000.ckpt"
RTDETR_REPO, RTDETR_FILE = "nvidia/RT-DETR-Hand-Detector-v1", "hand_detector.onnx"
DETECTOR_INPUT = 640  # RT-DETR ONNX graph is traced at 640x640 (plain resize, no letterbox)


@dataclass
class EngineConfig:
    checkpoint: str | None = None  # EgoHand .ckpt; default: HF nvidia/EgoHand-v1
    detector_onnx: str | None = None  # default: HF nvidia/RT-DETR-Hand-Detector-v1
    soma_assets: str | None = None  # default: soma.get_assets_dir() (HF nvidia/SOMA-X)
    device: str = "cuda"
    # fp16 lands 4x closer to fp32 than bf16 on EgoHand (0.3 vs 1.3 mm mean joint error), and
    # fp16 GEMMs with fp16 accumulation run at 2x the fp32-accumulate tensor-core rate on GeForce
    # cards (RTX 30xx laptop: DINOv3 batch 2 33 -> 28 ms compiled). Peak activation ~2.7e3 << 65504.
    dtype: str = "float16"
    fp16_accumulate: bool = True
    cuda_graphs: bool = True  # backbone / hand head / kp2d head as CUDA graphs (per hand count)
    detector_cuda_graph: bool = True  # RT-DETR via ORT IO binding + ORT CUDA graph
    compile: bool = False  # torch.compile the DINOv3 blocks (1 and 2 hands; ~10 s first start)
    torch_threads: int = 4  # CPU intra-op threads; the per-frame path has only tiny CPU ops
    hands: tuple[str, ...] = HANDS
    with_mesh: bool = True
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    # Monocular depth correction: each hand is moved along its viewing ray to depth_scale x its
    # estimated distance, keeping its size (measure with `egohand-live calibrate-depth`).
    depth_scale: float = 1.0


@dataclass
class HandResult:
    side: str
    score: float
    held: bool  # detector missed this hand; crop box carried over from earlier frames
    bbox: np.ndarray  # (3,) crop box [cx, cy, size], pixels
    joints_cam: np.ndarray  # (25, 3) SOMA joints, OpenCV camera frame, meters, smoothed
    joints_cam_raw: np.ndarray  # (25, 3) unsmoothed model output
    keypoints_2d: np.ndarray  # (25, 2) pixels, projection of joints_cam
    openxr_positions: np.ndarray  # (26, 3) camera frame
    openxr_orientations: np.ndarray  # (26, 4) XYZW, camera frame
    openxr_radii: np.ndarray  # (26,)
    vertices_cam: np.ndarray | None = None  # (V, 3) mesh, camera frame


@dataclass
class FrameResult:
    index: int
    t_capture: float
    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray
    hands: dict[str, HandResult]
    timings_ms: dict[str, float]
    depth_scale: float = 1.0


def _resolve_assets(cfg: EngineConfig) -> tuple[str, str, str]:
    from huggingface_hub import hf_hub_download

    ckpt = cfg.checkpoint or hf_hub_download(EGOHAND_REPO, EGOHAND_FILE)
    onnx = cfg.detector_onnx or hf_hub_download(RTDETR_REPO, RTDETR_FILE)
    if cfg.soma_assets:
        soma_dir = cfg.soma_assets
    else:
        from soma import get_assets_dir

        soma_dir = str(get_assets_dir())
    return ckpt, onnx, soma_dir


def _quiet_upstream_logs():
    # EgoPost logs per-call progress at WARNING; at 30+ Hz that floods the console.
    for name in ("inference", "egohand", "soma"):
        logging.getLogger(name).setLevel(logging.ERROR)


class EgoHandEngine:
    """Load once, then call :meth:`process` per frame."""

    def __init__(self, cfg: EngineConfig | None = None):
        self.cfg = cfg = cfg or EngineConfig()
        _quiet_upstream_logs()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(cfg.torch_threads)

        ckpt, onnx, soma_dir = _resolve_assets(cfg)
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            cfg.dtype
        ]
        torch.backends.cuda.matmul.allow_fp16_accumulation = bool(
            cfg.fp16_accumulate and dtype == torch.float16
        )

        from inference.hand_detector_2d.rtdetr.detector import _providers_for_device
        from inference.hand_detector_2d.rtdetr.onnx_model import HandDetector
        from inference.pipeline.predictor import load_egohand_model

        t0 = time.perf_counter()
        log.info("Loading EgoHand checkpoint %s", ckpt)
        model, _ = load_egohand_model(ckpt, soma_dir)
        model = model.eval().to(cfg.device)
        if cfg.compile:
            _compile_backbone_blocks(model.backbone)
        self.model = model
        use_graphs = cfg.cuda_graphs and cfg.device.startswith("cuda")
        self.predictor = LivePredictor(model, device=cfg.device, dtype=dtype, cuda_graphs=use_graphs)
        # The tracker needs sub-threshold candidates and does its own padding.
        self.detector = HandDetector(
            weights=onnx,
            score_thr=cfg.tracker.score_keep,
            providers=_providers_for_device(cfg.device),
            top1_per_class=False,
            pad=0.0,
            square=False,
            min_size=0.0,
            clip=True,
        )
        if cfg.device.startswith("cuda") and "CUDAExecutionProvider" not in self.detector.providers:
            log.warning(
                "RT-DETR is running on CPU: onnxruntime could not load its CUDA provider "
                "(missing CUDA 13 libraries? see scripts/setup.sh)."
            )
            self._det_io = None
        else:
            self._det_io = _DetectorIO(onnx, cfg.device, cfg.detector_cuda_graph)
        # Right hands are the mirrored left-hand mesh (see predictor.py).
        left_faces = model.soma_hand.faces("left")
        self.faces = {"left": left_faces, "right": mirrored_faces(left_faces)}
        self.tracker = HandBoxTracker(cfg.tracker)
        self.smoothers = {h: HandSmoother(cfg.smoothing) for h in HANDS}
        self._index = 0
        log.info(
            "Models loaded in %.1fs (detector providers: %s)",
            time.perf_counter() - t0,
            self.detector.providers,
        )

    # ------------------------------------------------------------------ stages

    def detect(self, frame_bgr: np.ndarray):
        """RT-DETR on one frame; returns unpadded ``Detection`` objects, score-sorted."""
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (DETECTOR_INPUT, DETECTOR_INPUT), interpolation=cv2.INTER_AREA)
        if self._det_io is not None:
            return _postprocess(self.detector, *self._det_io(small, w, h), w, h)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        batch = np.ascontiguousarray((rgb.astype(np.float32) * (1.0 / 255.0)).transpose(2, 0, 1)[None])
        # HandDetector's public API takes PIL images; _run takes the prepared tensor.
        return self.detector._run(batch, np.array([[w, h]], dtype=np.int64))[0]

    def warmup(self, width: int = 1280, height: int = 720, iterations: int = 3):
        """Run dummy frames through every stage (cuDNN autotune, torch.compile, ORT)."""
        img = np.full((height, width, 3), 127, np.uint8)
        K = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], np.float32)
        boxes = {
            "left": np.array([width * 0.3, height * 0.5, height * 0.4], np.float32),
            "right": np.array([width * 0.7, height * 0.5, height * 0.4], np.float32),
        }
        for _ in range(iterations):
            self.detect(img)
            self.predictor(img, K, boxes, with_vertices=self.cfg.with_mesh)
            self.predictor(img, K, {"left": boxes["left"]}, with_vertices=self.cfg.with_mesh)
        self.tracker.reset()
        for s in self.smoothers.values():
            s.reset()

    # ----------------------------------------------------------------- per frame

    @torch.no_grad()
    def process(self, frame_bgr: np.ndarray, K: np.ndarray, t: float) -> FrameResult:
        """Track hands in one frame.

        Args:
            frame_bgr: (H, W, 3) uint8 BGR image, undistorted (pinhole).
            K: (3, 3) pinhole intrinsics for this image.
            t: Capture timestamp in seconds (monotonic per stream).
        """
        timings = {}
        h, w = frame_bgr.shape[:2]
        K = np.asarray(K, dtype=np.float32)

        t0 = time.perf_counter()
        detections = self.detect(frame_bgr)
        t1 = time.perf_counter()
        tracked = self.tracker.update(detections, t, w)
        tracked = {k: v for k, v in tracked.items() if k in self.cfg.hands}
        timings["detect"] = (t1 - t0) * 1e3

        raw = self.predictor(
            frame_bgr, K, {k: v.bbox for k, v in tracked.items()}, with_vertices=self.cfg.with_mesh
        )
        t2 = time.perf_counter()
        timings["egohand"] = (t2 - t1) * 1e3

        hands: dict[str, HandResult] = {}
        for side, trk in tracked.items():
            r = raw.get(side)
            if r is None:
                continue
            joints_raw = np.asarray(r["joints"], dtype=np.float64)
            if not np.isfinite(joints_raw).all() or joints_raw[0, 2] <= 0.05:
                continue
            ds = self.cfg.depth_scale
            shift = (ds - 1.0) * joints_raw[0]  # root to ds x its distance, same ray
            joints_raw = joints_raw + shift
            if trk.new:  # new track or side change: don't filter from another hand's state
                self.smoothers[side].reset()
            joints = self.smoothers[side](joints_raw, t)
            # overlay: undo the depth correction so the skeleton stays on the hand in the image
            uv = (joints - (1.0 - 1.0 / ds) * joints[0]) @ K.T.astype(np.float64)
            kp2d = uv[:, :2] / np.maximum(uv[:, 2:], 1e-6)
            oxr_pos, oxr_quat, oxr_rad = soma25_to_openxr26(joints, side)
            verts = None
            if r["vertices"] is not None:  # shift the raw mesh with the smoothed wrist
                verts = (r["vertices"] + shift + (joints[0] - joints_raw[0])).astype(np.float32)
            hands[side] = HandResult(
                side=side,
                score=float(trk.score),
                held=trk.held,
                bbox=trk.bbox,
                joints_cam=joints.astype(np.float32),
                joints_cam_raw=joints_raw.astype(np.float32),
                keypoints_2d=kp2d.astype(np.float32),
                openxr_positions=oxr_pos,
                openxr_orientations=oxr_quat,
                openxr_radii=oxr_rad,
                vertices_cam=verts,
            )
        timings["post"] = (time.perf_counter() - t2) * 1e3
        timings["total"] = (time.perf_counter() - t0) * 1e3

        result = FrameResult(self._index, t, (w, h), K, hands, timings, self.cfg.depth_scale)
        self._index += 1
        return result


def _compile_backbone_blocks(backbone):
    """Like DinoV3Backbone.enable_full_batch_compile(2), but for any batch size: each block gets
    one static specialization per hand count. Runs inside the backbone's CUDA graph."""
    backbone._compiled_blocks = [
        torch.compile(b, backend="inductor", fullgraph=True, dynamic=False) for b in backbone.encoder.blocks
    ]
    backbone.forward = backbone._forward_compiled_spatial_features


class _DetectorIO:
    """RT-DETR fed through ORT IO binding: the resized uint8 frame goes up through a pinned
    buffer, BGR->RGB / scaling / HWC->CHW run on the GPU (same float32 math as detect()), and the
    whole ORT graph can be replayed as a CUDA graph (static 640x640 input, 300 queries)."""

    def __init__(self, onnx_path, device, cuda_graph=True):
        import onnxruntime as ort

        dev = torch.device(device)
        self.dev, idx = dev, dev.index or 0
        opts = {"device_id": idx}
        sess = None
        if cuda_graph:
            try:  # needs every node on the CUDA EP
                sess = ort.InferenceSession(
                    onnx_path, providers=[("CUDAExecutionProvider", {**opts, "enable_cuda_graph": "1"})]
                )
            except Exception as e:  # noqa: BLE001
                log.warning("ORT CUDA graph unavailable for RT-DETR (%s); running without", e)
        self.sess = sess or ort.InferenceSession(onnx_path, providers=[("CUDAExecutionProvider", opts)])
        s = DETECTOR_INPUT
        self.host = torch.empty((s, s, 3), dtype=torch.uint8).pin_memory()
        self.inp = torch.empty((1, 3, s, s), dtype=torch.float32, device=dev)
        self.sizes = torch.zeros((1, 2), dtype=torch.int64, device=dev)
        self.out = {
            "labels": torch.empty((1, 300), dtype=torch.int64, device=dev),
            "boxes": torch.empty((1, 300, 4), dtype=torch.float32, device=dev),
            "scores": torch.empty((1, 300), dtype=torch.float32, device=dev),
        }
        np_type = {torch.int64: np.int64, torch.float32: np.float32}
        self.io = self.sess.io_binding()
        self.io.bind_input("images", "cuda", idx, np.float32, tuple(self.inp.shape), self.inp.data_ptr())
        self.io.bind_input("orig_target_sizes", "cuda", idx, np.int64, (1, 2), self.sizes.data_ptr())
        for name, t in self.out.items():
            self.io.bind_output(name, "cuda", idx, np_type[t.dtype], tuple(t.shape), t.data_ptr())
        self._wh = None

    def __call__(self, small_bgr, w, h):
        self.host.numpy()[:] = small_bgr
        x = self.host.to(self.dev, non_blocking=True).flip(-1).permute(2, 0, 1)[None].float()
        self.inp.copy_(x.mul_(1.0 / 255.0))
        if self._wh != (w, h):
            self.sizes.copy_(torch.tensor([[w, h]]))
            self._wh = (w, h)
        torch.cuda.current_stream(self.dev).synchronize()  # ORT runs on its own stream
        self.sess.run_with_iobinding(self.io)  # returns after ORT's stream is done
        return tuple(self.out[k].cpu().numpy() for k in ("labels", "boxes", "scores"))


def _postprocess(det, labels, boxes, scores, w, h):
    """HandDetector._run's per-image post-processing (score filter, sort, refine_box)."""
    from inference.hand_detector_2d.rtdetr.onnx_model import LABELS, Detection, refine_box

    lab, box, sc = labels[0], boxes[0], scores[0]
    keep = sc > det.score_thr
    dets = [
        Detection(LABELS.get(int(l), str(int(l))), int(l), float(p), tuple(float(v) for v in b))
        for l, b, p in zip(lab[keep], box[keep], sc[keep])
    ]
    dets.sort(key=lambda d: -d.score)
    for d in dets:
        d.xyxy = refine_box(d.xyxy, w, h, det.pad, det.square, det.min_size, det.clip)
    return dets


def slim_checkpoint(src: str | None, dst: str | Path) -> Path:
    """Write a weights-only copy of the EgoHand checkpoint (drops optimizer state).

    The HF checkpoint carries training state; the slim copy loads several times
    faster and works with EgoPost's loader unchanged.
    """
    from huggingface_hub import hf_hub_download

    src = src or hf_hub_download(EGOHAND_REPO, EGOHAND_FILE)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    keys = ("state_dict", "hyper_parameters", "egohand_camera_depth_mode")
    keep = {k: ckpt[k] for k in keys if k in ckpt}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(keep, dst)
    return dst
