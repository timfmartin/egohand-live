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
"""

import numpy as np
import torch

_FLIP_X = np.array([-1.0, 1.0, 1.0], dtype=np.float32)


class LivePredictor:
    def __init__(self, model, device="cuda", dtype=torch.bfloat16, pad_to_two=True):
        from inference.preprocessing.crop import crop_and_preprocess
        from inference.preprocessing.mirroring import prepare_right_hand_model_input

        camera_mode = getattr(model, "camera_mode", "weak_perspective")
        if camera_mode != "sam3db_perspective":
            raise NotImplementedError(
                f"LivePredictor supports sam3db_perspective checkpoints (EgoHand-v1), got {camera_mode!r}"
            )
        self.model = model
        self.device = device
        self.dtype = dtype
        # A constant batch shape lets cuDNN / torch.compile / CUDA graphs reuse one plan.
        self.pad_to_two = pad_to_two
        self._crop = crop_and_preprocess
        self._crop_right = prepare_right_hand_model_input

    def prepare(self, frame_bgr: np.ndarray, K: np.ndarray, boxes: dict[str, np.ndarray]):
        """Crop and canonicalize each hand. Returns (sides, batch dict)."""
        h, w = frame_bgr.shape[:2]
        sides, crops, bbxs, Ks = [], [], [], []
        for side, bbox in boxes.items():
            bbox = np.asarray(bbox, np.float32)
            if side == "right":
                crop, model_bbox, model_K = self._crop_right(frame_bgr, bbox, K, calibrated_full_image=True)
            else:
                crop, _ = self._crop(frame_bgr, bbox)
                model_bbox, model_K = bbox, K
            sides.append(side)
            crops.append(crop)
            bbxs.append(np.asarray(model_bbox, np.float32))
            Ks.append(np.asarray(model_K, np.float32))
        n = len(sides)
        if self.pad_to_two and n == 1:
            crops.append(crops[0])
            bbxs.append(bbxs[0])
            Ks.append(Ks[0])
        b = len(crops)
        dev = self.device
        batch = {
            "img": torch.stack(crops).pin_memory().to(dev, non_blocking=True),
            "bbx_xys": torch.from_numpy(np.stack(bbxs)).to(dev, non_blocking=True),
            "K_fullimg": torch.from_numpy(np.stack(Ks)).to(dev, non_blocking=True),
            "img_size": torch.tensor([[w, h]] * b, dtype=torch.float32, device=dev),
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
