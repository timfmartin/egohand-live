"""Check egohand-live's per-frame predictor against EgoPost's EgoHandPredictor.run.

Runs both on the same frames and crop boxes (from the RT-DETR detector) and reports
the max joint/vertex difference per hand, plus per-frame timings of both paths.

    python scripts/verify_against_upstream.py --video clip.mp4 [--intrinsics cam.json]
"""

import argparse
import logging
import time

import numpy as np
import torch

from egohand_live.engine import EgoHandEngine, EngineConfig
from egohand_live.sources import open_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--intrinsics")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--dtype", default="float32", help="float32 isolates path differences from bf16 noise")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from inference.pipeline.predictor import EgoHandPredictor

    # CUDA graphs wrap the shared model's forwards (and their outputs are overwritten on replay),
    # so compare the plain eager path against upstream.
    eng = EgoHandEngine(EngineConfig(dtype=args.dtype, with_mesh=True, cuda_graphs=False))
    upstream = EgoHandPredictor(eng.model, batch_size=2, device="cuda", dtype=eng.predictor.dtype)
    src = open_source(args.video, intrinsics=args.intrinsics, realtime=False, scale=args.scale)

    diffs = {"left": [], "right": []}
    vdiffs = {"left": [], "right": []}
    t_up, t_live = [], []
    n = 0
    while n < args.frames and (f := src.read()) is not None:
        tracked = eng.tracker.update(eng.detect(f.bgr), f.t, f.bgr.shape[1])
        boxes = {k: v.bbox for k, v in tracked.items()}
        if not boxes:
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref = upstream.run(images_bgr=[f.bgr], image_ids=["f"], bboxes_per_image={"f": boxes},
                           hands=list(boxes), intrinsics_per_image={"f": f.K})  # fmt: skip
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        live = eng.predictor(f.bgr, f.K, boxes, with_vertices=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if n >= 3:
            t_up.append((t1 - t0) * 1e3)
            t_live.append((t2 - t1) * 1e3)
        for side in boxes:
            r = ref[side]["f"]
            j_ref = r["pred_keypoints_3d"] + r["transl"]
            v_ref = r["pred_vertices"] + r["transl"]
            diffs[side].append(np.abs(live[side]["joints"] - j_ref).max())
            vdiffs[side].append(np.abs(live[side]["vertices"] - v_ref).max())
        n += 1

    print(f"\n{n} frames, dtype={args.dtype}")
    for side in ("left", "right"):
        if diffs[side]:
            d, v = np.array(diffs[side]) * 1e3, np.array(vdiffs[side]) * 1e3
            print(f"{side:5s}: max |joint diff| mean {d.mean():.4f} mm, worst {d.max():.4f} mm | "
                  f"max |vertex diff| worst {v.max():.4f} mm  ({len(d)} frames)")  # fmt: skip
    print(f"upstream run(): mean {np.mean(t_up):6.1f} ms | live predictor: mean {np.mean(t_live):6.1f} ms")


if __name__ == "__main__":
    main()
