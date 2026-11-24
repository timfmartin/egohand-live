"""Per-frame EgoHand inference for up to two hands in one batch.

EgoPost's ``EgoHandPredictor.run`` is built for clips: a thread pool per call, a
batched SOMA re-skinning pass to turn left-canonical predictions back into right
hands, and full per-crop result dicts. For one frame that overhead dominates
(measured on an RTX 5090: ~110 ms per two-hand frame vs ~25 ms GPU time).

This module calls ``model.forward_step`` directly and replaces the re-skinning with
the equivalent geometric mirror. For ``sam3db_perspective`` checkpoints a right hand
is fed as the horizontally flipped full image with camera ``K' = F K M`` (M negates
camera X), so the model's camera-frame output is the real right hand with X negated.
``scripts/verify_against_upstream.py`` checks this against EgoPost.

Speed (no upstream edits, only wrappers):
  * matmul weights are stored in the autocast dtype, so autocast stops re-casting the
    ~840M DINOv3 weights every frame (numerically identical);
  * backbone, hand head and kp2d head run as CUDA graphs, one per batch size (1 or 2
    hands), which removes ~2000 Python-side kernel launches per frame;
  * SOMA's skeleton fit uses its precomputed RBF joint regressor (exact to <1 um)
    instead of 25 per-joint LU solves with a host sync each;
  * crops are warped straight from the BGR frame (right hands: only the crop footprint
    is flipped) and normalized on the GPU (bit-identical pixels).
"""

import cv2
import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

_FLIP_X = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
_IMG = 256  # EgoPost crop size (inference.preprocessing.crop.IMG_SIZE)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class CudaGraphCall:
    """Replays one CUDA graph per input signature (shapes/dtypes/non-tensor args).

    Outputs live in the graph's private pool and are overwritten by the next replay of
    the same graph; ``LivePredictor.__call__`` consumes them before the next frame.
    """

    def __init__(self, fn, warmup=3):
        self.fn = fn
        self.warmup = warmup
        self.graphs = {}

    def __call__(self, *args, **kwargs):
        flat, spec = tree_flatten((args, kwargs))
        sig = tuple(
            (tuple(x.shape), x.dtype, x.stride()) if isinstance(x, torch.Tensor) else x for x in flat
        )
        key = (str(spec), sig, torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        entry = self.graphs.get(key)
        if entry is None:
            entry = self.graphs[key] = self._capture(flat, spec)
        static_in, graph, out = entry
        for s, x in zip(static_in, flat):
            if s is not None:
                s.copy_(x, non_blocking=True)
        graph.replay()
        return out

    def _capture(self, flat, spec):
        static_in = [x.clone() if isinstance(x, torch.Tensor) else None for x in flat]
        args, kwargs = tree_unflatten([s if s is not None else x for s, x in zip(static_in, flat)], spec)
        # Autocast's weight-cast cache would leave the graph pointing at cast copies that are
        # freed when the caller's autocast region exits: warm up and capture without it.
        ac = dict(device_type="cuda", dtype=torch.get_autocast_dtype("cuda"),
                  enabled=torch.is_autocast_enabled("cuda"), cache_enabled=False)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.autocast(**ac):
            for _ in range(self.warmup):
                self.fn(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.autocast(**ac):
            out = self.fn(*args, **kwargs)
        return static_in, graph, out


def _cast_matmul_weights(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """Store Linear/Conv2d weights in the autocast dtype. Autocast casts them to exactly this
    (round-to-nearest) every call otherwise: ~5 GB of memory traffic per frame for DINOv3-H+."""
    for m in module.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            m.to(dtype)


def _use_precomputed_soma_regressor(soma_hand) -> None:
    """SOMA fits the skeleton with 25 per-joint RBF solves (each with a boolean-mask host
    sync). The fit is linear in the rest shape, and SOMA already supports precomputing it
    (``use_sparse_rbf_matrix``), which the hand layer just doesn't enable."""
    for hand in soma_hand._hands.values():
        st = hand.skeleton_transfer
        st.use_sparse_rbf_matrix = True
        st._precompute_regressors()
        st.sparse_rbf_matrix = st.sparse_rbf_matrix.to_dense()  # (25, V): tiny, faster than CSR


def _crop_rgb_u8(frame_bgr: np.ndarray, bbox, flip: bool = False) -> np.ndarray:
    """(256, 256, 3) uint8 RGB crop, bit-identical to EgoPost's crop_and_preprocess.

    With ``flip``, ``bbox`` is in the horizontally flipped image (u' = W-1-u), as in
    ``prepare_right_hand_model_input``, but only the crop's footprint is flipped. The
    BGR->RGB swap commutes with the warp; the affine is inverted exactly as cv2.warpAffine
    does internally and shifted to the ROI by whole pixels, so sampling is unchanged."""
    H, W = frame_bgr.shape[:2]
    cx, cy, s = float(bbox[0]), float(bbox[1]), float(bbox[2])
    T = np.array([[_IMG / s, 0, _IMG / 2 - cx * _IMG / s], [0, _IMG / s, _IMG / 2 - cy * _IMG / s]],
                 np.float32).astype(np.float64)  # == egohand.utils.crop_transform.get_transform
    half = s / 2 + 2
    x0, x1 = max(int(np.floor(cx - half)), 0), min(int(np.ceil(cx + half)) + 1, W)
    y0, y1 = max(int(np.floor(cy - half)), 0), min(int(np.ceil(cy + half)) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((_IMG, _IMG, 3), np.uint8)
    roi = frame_bgr[y0:y1, W - x1 : W - x0][:, ::-1] if flip else frame_bgr[y0:y1, x0:x1]
    d = 1.0 / (T[0, 0] * T[1, 1])
    a11, a22 = T[1, 1] * d, T[0, 0] * d
    inv = np.array([[a11, 0.0, -a11 * T[0, 2] - x0], [0.0, a22, -a22 * T[1, 2] - y0]])
    crop = cv2.warpAffine(np.ascontiguousarray(roi), inv, (_IMG, _IMG),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return crop[:, :, ::-1]


class LivePredictor:
    def __init__(self, model, device="cuda", dtype=torch.bfloat16, pad_to_two=False, cuda_graphs=True):
        from inference.preprocessing.mirroring import mirror_full_image_camera_horizontal

        camera_mode = getattr(model, "camera_mode", "weak_perspective")
        if camera_mode != "sam3db_perspective":
            raise NotImplementedError(
                f"LivePredictor supports sam3db_perspective checkpoints (EgoHand-v1), got {camera_mode!r}"
            )
        self.model = model
        self.device = device
        self.dtype = dtype
        # Padding one hand to batch 2 doubles backbone work; CUDA graphs keep one plan per size.
        self.pad_to_two = pad_to_two
        self._mirror = mirror_full_image_camera_horizontal
        if dtype != torch.float32:
            _cast_matmul_weights(model.backbone, dtype)
            torch.cuda.empty_cache()
        _use_precomputed_soma_regressor(model.soma_hand)
        if cuda_graphs:
            model.backbone.forward = CudaGraphCall(model.backbone.forward)
            model.hand_head.forward = CudaGraphCall(model.hand_head.forward)
            model._predict_direct_keypoints_2d = CudaGraphCall(model._predict_direct_keypoints_2d)
        self._mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
        self._host = {}  # batch size -> pinned (crops, meta) staging buffers
        self._img_size = {}

    def prepare(self, frame_bgr: np.ndarray, K: np.ndarray, boxes: dict[str, np.ndarray]):
        """Crop and canonicalize each hand. Returns (sides, batch dict)."""
        h, w = frame_bgr.shape[:2]
        sides = list(boxes)
        b = 2 if self.pad_to_two and len(sides) == 1 else len(sides)
        if b not in self._host:
            self._host[b] = (torch.empty((b, _IMG, _IMG, 3), dtype=torch.uint8).pin_memory(),
                             torch.empty((b, 12), dtype=torch.float32).pin_memory())
        crops_h, meta_h = self._host[b]
        crops, meta = crops_h.numpy(), meta_h.numpy()
        for i, side in enumerate(sides):
            bbox = np.asarray(boxes[side], np.float32)
            if side == "right":
                bbox, model_K = self._mirror(bbox, K, w)
            else:
                model_K = K
            crops[i] = _crop_rgb_u8(frame_bgr, bbox, flip=side == "right")
            meta[i, :3] = bbox
            meta[i, 3:] = np.asarray(model_K, np.float32).reshape(9)
        if b > len(sides):
            crops[1:], meta[1:] = crops[:1], meta[:1]
        dev = self.device
        # The staging buffers are reused: safe because __call__ syncs (.cpu()) before returning.
        img = crops_h.to(dev, non_blocking=True).permute(0, 3, 1, 2).float()
        img = img.div_(255.0).sub_(self._mean).div_(self._std)  # same float32 math as EgoPost
        meta = meta_h.to(dev, non_blocking=True)
        if (w, h, b) not in self._img_size:
            self._img_size[(w, h, b)] = torch.tensor([[w, h]] * b, dtype=torch.float32, device=dev)
        batch = {
            "img": img,
            "bbx_xys": meta[:, :3],
            "K_fullimg": meta[:, 3:].reshape(b, 3, 3),
            "img_size": self._img_size[(w, h, b)],
            "hand_type": ["left"] * b,  # right hands were flipped into the left-hand frame
        }
        return sides, batch

    @torch.no_grad()
    def __call__(self, frame_bgr, K, boxes: dict[str, np.ndarray], with_vertices=False):
        """Returns ``{side: {"joints": (25, 3), "vertices": (V, 3) | None}}`` in the
        real camera frame (OpenCV axes, meters)."""
        if not boxes:
            return {}
        sides, batch = self.prepare(frame_bgr, np.asarray(K, np.float32), boxes)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = self.model.forward_step(batch, train=False)
        n = len(sides)
        cam_t = out["pred_cam_t"][:n].float().unsqueeze(1)
        tensors = [out["pred_keypoints_3d"][:n].float() + cam_t]
        if with_vertices:
            tensors.append(out["pred_vertices"][:n].float() + cam_t)
        arrays = [t.cpu().numpy() for t in tensors]
        results = {}
        for i, side in enumerate(sides):
            sign = _FLIP_X if side == "right" else 1.0
            results[side] = {
                "joints": arrays[0][i] * sign,
                "vertices": arrays[1][i] * sign if with_vertices else None,
            }
        return results


def mirrored_faces(faces: np.ndarray) -> np.ndarray:
    """Faces for a mirrored mesh (reverse winding so normals still point outward)."""
    return np.ascontiguousarray(faces[:, ::-1])
