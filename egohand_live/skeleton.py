"""Hand skeleton layouts and the SOMA-25 -> OpenXR-26 conversion.

SOMA-25 (EgoHand output order): WRIST, THUMB1-3, THUMB_END, then INDEX/MIDDLE/RING/PINKY
each as 1-4, _END. Joint ``1`` of a long finger is the metacarpal base (CMC),
``2`` the knuckle (MCP), ``3`` PIP, ``4`` DIP, ``_END`` the tip.

OpenXR-26 (XR_EXT_hand_tracking): PALM, WRIST, THUMB_{METACARPAL,PROXIMAL,DISTAL,TIP},
then INDEX/MIDDLE/RING/LITTLE as {METACARPAL,PROXIMAL,INTERMEDIATE,DISTAL,TIP}.
Joint axes: -Z along the bone toward the fingertip, +Y out of the back of the hand,
X = Y x Z. Poses are absolute in the reference space (not parent-relative).
"""

import numpy as np

SOMA25_NAMES = (
    "Wrist",
    "Thumb1", "Thumb2", "Thumb3", "ThumbEnd",
    "Index1", "Index2", "Index3", "Index4", "IndexEnd",
    "Middle1", "Middle2", "Middle3", "Middle4", "MiddleEnd",
    "Ring1", "Ring2", "Ring3", "Ring4", "RingEnd",
    "Pinky1", "Pinky2", "Pinky3", "Pinky4", "PinkyEnd",
)  # fmt: skip
SOMA25_PARENT = np.array(
    [-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23]
)
SOMA25_BONES = tuple((int(SOMA25_PARENT[j]), j) for j in range(1, 25))

OPENXR26_NAMES = (
    "palm", "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "little_metacarpal", "little_proximal", "little_intermediate", "little_distal", "little_tip",
)  # fmt: skip
# OpenXR index i>=1 takes SOMA joint OPENXR_FROM_SOMA[i]; PALM (0) is synthesized.
OPENXR_FROM_SOMA = np.array([-1, *range(25)])

# Bone child used to orient each SOMA joint (tips reuse their own incoming bone).
_SOMA_CHILD = np.array(
    [11, 2, 3, 4, -1, 6, 7, 8, 9, -1, 11, 12, 13, 14, -1, 16, 17, 18, 19, -1, 21, 22, 23, 24, -1]
)

# Approximate adult joint radii in meters (OpenXR order), used only for visualization
# and for consumers that require the radius field.
OPENXR26_RADIUS = np.array(
    [0.020, 0.020,
     0.017, 0.013, 0.011, 0.009,
     0.013, 0.011, 0.009, 0.008, 0.007,
     0.013, 0.011, 0.009, 0.008, 0.007,
     0.012, 0.010, 0.008, 0.007, 0.006,
     0.011, 0.009, 0.007, 0.006, 0.005],
    dtype=np.float32,
)  # fmt: skip

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP, MIDDLE_CMC = 0, 6, 11, 21, 10


def _normalize(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) -> unit quaternions (..., 4) in XYZW order."""
    from scipy.spatial.transform import Rotation

    shape = R.shape[:-2]
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(*shape, 4)


def hand_frames(joints: np.ndarray, side: str) -> np.ndarray:
    """Per-joint OpenXR-convention rotation matrices for SOMA-25 joint positions.

    Built from geometry only (bone directions + knuckle line), so the result does not
    depend on the SOMA rig's internal joint orients. Columns are the joint X, Y, Z axes
    expressed in the input frame.

    Args:
        joints: (25, 3) positions in any right-handed frame.
        side: "left" or "right".

    Returns:
        (25, 3, 3) rotation matrices.
    """
    j = np.asarray(joints, dtype=np.float64)
    # Knuckle line chosen so that X = Y x Z holds for both hands (OpenXR convention):
    # right hand -> toward the pinky, left hand -> toward the index.
    knuckles = j[PINKY_MCP] - j[INDEX_MCP]
    x_ref = _normalize(knuckles if side == "right" else -knuckles)

    child = _SOMA_CHILD.copy()
    tips = child < 0
    bone_dir = np.empty_like(j)
    bone_dir[~tips] = j[child[~tips]] - j[~tips]
    bone_dir[tips] = j[tips] - j[SOMA25_PARENT[tips]]
    bone_dir[WRIST] = j[MIDDLE_MCP] - j[WRIST]
    z = -_normalize(bone_dir)

    # Long fingers flex about the knuckle line, so it stays a good lateral reference.
    # The thumb does not; give it the palm normal instead.
    y = _normalize(np.cross(z, x_ref))
    dorsal = y[WRIST]
    thumb = slice(1, 5)
    x_thumb = _normalize(np.cross(dorsal[None], z[thumb]))
    y[thumb] = _normalize(np.cross(z[thumb], x_thumb))
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def soma25_to_openxr26(joints: np.ndarray, side: str):
    """Convert SOMA-25 joint positions to OpenXR-26 joint poses.

    Returns:
        positions (26, 3), quaternions XYZW (26, 4), radii (26,), all float32.
    """
    j = np.asarray(joints, dtype=np.float64)
    frames = hand_frames(j, side)
    pos = np.empty((26, 3))
    rot = np.empty((26, 3, 3))
    pos[1:] = j
    rot[1:] = frames
    # PALM: centre of the middle metacarpal, oriented like the wrist.
    pos[0] = 0.5 * (j[MIDDLE_CMC] + j[MIDDLE_MCP])
    rot[0] = frames[WRIST]
    return (
        pos.astype(np.float32),
        matrix_to_quat_xyzw(rot).astype(np.float32),
        OPENXR26_RADIUS.copy(),
    )
