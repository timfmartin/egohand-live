"""Causal left/right hand box tracking on top of per-frame RT-DETR detections.

EgoPost's native tracker fills gaps from the *next* detection and smooths boxes in
both temporal directions, so it needs the full clip. This tracker only looks back:

- Continuity beats the class label: an existing track keeps the candidate that
  overlaps it best, even if the detector's left/right label flickers.
- New tracks start only from confident detections. With ``handedness="position"`` (the
  default) their side comes from the image position, not the class label: the detector
  is trained on head-mounted video, where you mostly see the backs of your hands, and it
  often calls a palm-facing left hand "right" (the two look alike). For a camera behind
  or above the operator the left hand is reliably the one further left. If two tracks
  end up on the wrong sides of each other they are swapped. ``handedness="detector"``
  uses the class label (EgoPost's behaviour).
- A lost track is held for ``hold_s`` seconds, then dropped.
- Each track keeps a confidence (EMA of its detection scores). It is published only after
  ``confirm_frames`` detections and while the confidence is >= ``min_confidence``, and is
  dropped below ``drop_confidence``. Low-score detections may continue a track (the
  detector scores some real hands < 0.2), but a false positive that never scores high,
  e.g. a desk object at 0.15-0.5, never shows up.
- Box centre and size are exponentially smoothed (size more strongly, since crop
  scale jitter shows up directly as depth jitter in the model output).
"""

from dataclasses import dataclass, field

import numpy as np

HANDS = ("left", "right")
_LABEL_TO_HAND = {1: "left", 2: "right"}


def iou_square(a, b) -> float:
    """IoU of two [cx, cy, size] square boxes."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[2] / 2, a[0] + a[2] / 2, a[1] + a[2] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[2] / 2, b[0] + b[2] / 2, b[1] + b[2] / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] ** 2 + b[2] ** 2 - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackerConfig:
    box_scale: float = 1.3  # crop side = longest detector side * box_scale (EgoHand padding=0.3)
    score_new: float = 0.3  # min score to start a track (RT-DETR scores some hands low)
    score_keep: float = 0.12  # min score to continue an overlapping track
    conf_alpha: float = 0.3  # EMA weight of a new detection score in the track confidence
    min_confidence: float = 0.45  # publish a track only while its confidence is at least this
    drop_confidence: float = 0.25  # forget a track whose confidence falls below this
    confirm_frames: int = 3  # detections needed before a new track is published
    iou_keep: float = 0.25  # min IoU with the previous box to continue a track
    w_score: float = 1.0
    w_iou: float = 1.5
    hold_s: float = 0.15  # keep a lost box this long (covers 1-4 dropped detections)
    center_alpha: float = 0.75  # EMA weight of the new centre (1 = no smoothing)
    size_alpha: float = 0.35  # EMA weight of the new size
    min_size_frac: float = 0.06  # minimum crop side, as a fraction of image width
    dedup_iou: float = 0.8  # same box reported under both labels
    handedness: str = "position"  # "position" (fixed camera behind the operator) or "detector"
    centre_band: float = 0.1  # position mode: a lone new hand within this of the image centre
    #                           (fraction of the width) takes the detector's label instead
    swap_margin: float = 0.05  # position mode: swap tracks whose centres are this far on the
    #                            wrong sides of each other (fraction of the width)


@dataclass
class _Track:
    box: np.ndarray  # smoothed [cx, cy, side] of the *unpadded* square box
    score: float
    last_seen: float
    age: int = 0
    conf: float = 0.0  # EMA of detection scores
    published: bool = False


@dataclass
class TrackedHand:
    side: str
    bbox: np.ndarray  # padded [cx, cy, size] crop box for EgoHand
    score: float  # track confidence (EMA of detection scores)
    held: bool  # True when no detection this frame (box carried over)
    new: bool = False  # track started or changed side this frame (reset per-hand filters)


@dataclass
class HandBoxTracker:
    cfg: TrackerConfig = field(default_factory=TrackerConfig)

    def __post_init__(self):
        self.tracks: dict[str, _Track | None] = dict.fromkeys(HANDS)

    def reset(self):
        self.tracks = dict.fromkeys(HANDS)

    def _candidates(self, detections):
        cands = []  # (hand_label, score, box)
        for det in detections:
            if det.score < self.cfg.score_keep:
                continue
            x1, y1, x2, y2 = det.xyxy
            box = np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)])
            dup = next((c for c in cands if iou_square(c[2], box) > self.cfg.dedup_iou), None)
            if dup is None:
                cands.append([_LABEL_TO_HAND.get(det.label_id), det.score, box])
            elif det.score > dup[1]:
                dup[0], dup[1] = _LABEL_TO_HAND.get(det.label_id), det.score
        return cands

    def update(self, detections, t: float, image_width: int) -> dict[str, TrackedHand]:
        """Advance all tracks by one frame.

        Args:
            detections: ``onnx_model.Detection`` list for this frame (unpadded boxes).
            t: Frame timestamp in seconds.
            image_width: Frame width in pixels (for the minimum crop size).
        """
        c = self.cfg
        cands = self._candidates(detections)
        used: set[int] = set()
        matched: dict[str, tuple[float, np.ndarray]] = {}

        # 1) Continue existing tracks: best (score + IoU) candidate, greedily by cost.
        pairs = []
        for hand, tr in self.tracks.items():
            if tr is None:
                continue
            for i, (_, score, box) in enumerate(cands):
                iou = iou_square(tr.box, box)
                if iou >= c.iou_keep:
                    pairs.append((c.w_score * score + c.w_iou * iou, hand, i))
        for _, hand, i in sorted(pairs, reverse=True):
            if hand in matched or i in used:
                continue
            matched[hand] = (cands[i][1], cands[i][2])
            used.add(i)

        # 2) Start (or re-acquire) tracks from confident candidates.
        if c.handedness == "position":
            swapped = self._start_by_position(cands, used, matched, image_width)
        else:
            swapped = False
            for hand in HANDS:
                if hand in matched:
                    continue
                best = max(
                    (
                        (score, i)
                        for i, (label, score, _) in enumerate(cands)
                        if i not in used and label == hand and score >= c.score_new
                    ),
                    default=None,
                )
                if best is not None:
                    score, i = best
                    matched[hand] = (score, cands[i][2])
                    used.add(i)

        # 3) Update / hold / drop.
        out: dict[str, TrackedHand] = {}
        min_side = c.min_size_frac * image_width
        for hand in HANDS:
            tr = self.tracks[hand]
            new = swapped
            if hand in matched:
                score, box = matched[hand]
                if tr is None or t - tr.last_seen > c.hold_s:
                    tr = _Track(box=box.copy(), score=score, last_seen=t, conf=score)
                else:
                    tr.box[:2] += c.center_alpha * (box[:2] - tr.box[:2])
                    tr.box[2] += c.size_alpha * (box[2] - tr.box[2])
                    tr.score, tr.last_seen = score, t
                    tr.conf += c.conf_alpha * (score - tr.conf)
                    tr.age += 1
                if tr.conf < c.drop_confidence:
                    self.tracks[hand] = None
                    continue
                self.tracks[hand] = tr
                held = False
            elif tr is not None and t - tr.last_seen <= c.hold_s:
                held = True
            else:
                self.tracks[hand] = None
                continue
            if tr.age + 1 < c.confirm_frames or tr.conf < c.min_confidence:
                tr.published = False
                continue
            new = new or not tr.published  # (re)appearing: start its filters fresh
            tr.published = True
            crop = tr.box.copy()
            crop[2] = max(crop[2] * c.box_scale, min_side)
            out[hand] = TrackedHand(hand, crop.astype(np.float32), tr.conf, held, new)
        return out

    def _start_by_position(self, cands, used, matched, image_width) -> bool:
        """Position-mode track starts (step 2) and the crossed-sides check. Updates
        ``matched`` / ``used`` in place; returns True if the two tracks were swapped."""
        c = self.cfg
        free = [h for h in HANDS if h not in matched]
        pool = sorted(
            (i for i, (_, score, _) in enumerate(cands) if i not in used and score >= c.score_new),
            key=lambda i: -cands[i][1],
        )[: len(free)]
        if len(pool) == 2:  # both sides free: order by x
            pool.sort(key=lambda i: cands[i][2][0])
            assign = list(zip(HANDS, pool))
        elif len(pool) == 1:
            i = pool[0]
            cx, label = cands[i][2][0], cands[i][0]
            if len(free) == 2:
                off = cx / image_width - 0.5
                hand = label if abs(off) < c.centre_band and label in HANDS else ("left" if off < 0 else "right")
                assign = [(hand, i)]
            else:
                hand = free[0]
                other = matched.get("right" if hand == "left" else "left")
                ok = other is None or (cx < other[1][0] if hand == "left" else cx > other[1][0])
                assign = [(hand, i)] if ok else []
        else:
            assign = []
        for hand, i in assign:
            matched[hand] = (cands[i][1], cands[i][2])
            used.add(i)

        if "left" in matched and "right" in matched:
            if matched["left"][1][0] - matched["right"][1][0] > c.swap_margin * image_width:
                matched["left"], matched["right"] = matched["right"], matched["left"]
                self.tracks["left"], self.tracks["right"] = self.tracks["right"], self.tracks["left"]
                return True
        return False
