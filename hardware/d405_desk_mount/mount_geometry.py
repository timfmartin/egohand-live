#!/usr/bin/env python3
"""Dimensions of the RealSense D405 desk mount + camera extrinsics helper.

Pure Python (no FreeCAD needed). build_mount.py takes every dimension from here,
so edit values in this file and rebuild.

Run this file to get the camera pose for `egohand-live run --extrinsics`:

    python3 mount_geometry.py --collar-height 415 --tilt 45 > T_world_camera.txt

World frame ("foot frame"): origin on the desk surface under the bar axis,
+X to the operator's right, +Y forward (away from the operator, the direction the
camera looks), +Z up. The matrix translation is in metres.
Camera frame: RealSense/OpenCV optical frame (x right, y down, z forward) with the
origin at the D405 left imager, which is the origin of both its depth and colour
streams.
All lengths below are in mm.
"""
import argparse
import math

# -- 20 x 20 aluminium extrusion (B-type, slot 6) ------------------------------
BAR = 20.0
BAR_LEN = 880.0
BAR_CLR = 0.2  # per side; the lock pads take up the play
BORE = BAR + 2 * BAR_CLR

# -- Bar locks: an M4x16 + M4 nut pushes a flexible pad (printed in one piece with
#    the wall) flat against the bar's back face; the pad spans the slot and bears
#    on both lips, the bar is pressed flat against the opposite wall ---------------
M4_CLEAR_D = 4.4
M4_NUT_AF = 7.0 + 0.3  # nut across flats + print tolerance
M4_NUT_T = 3.2 + 0.5  # nut thickness + print tolerance
LOCK_SCREW_LEN = 16.0  # M4x16
LOCK_HEAD_STANDOFF = 3.0  # head-to-boss gap when the screw first touches the pad
PAD_W = 16.0
PAD_T = 3.0
PAD_LEN = 26.0
PAD_GAP = 0.8  # slits around and behind the pad
PAD_END_SLIT = 1.5  # slit over a free end that is not an open face (room for bridge sag)
PAD_SCREW_FROM_END = 8.0  # screw distance from the pad's free end
PAD_BACK_Y = -(BORE / 2 + PAD_T)  # pad's rear face (the screw pushes here)
LOCK_BOSS_Y = PAD_BACK_Y - LOCK_SCREW_LEN + LOCK_HEAD_STANDOFF  # boss rear face
LOCK_NUT_Y0 = LOCK_BOSS_Y + 4.5  # rear face of the nut slot

# -- Tilt joint hardware: 2x M5x30, pivot with a plain nut, lock bolt with a washer
#    under the head and the printed wing knob (normal M5 nut inside) ---------------
M5_CLEAR_D = 5.5
M5_NUT_AF = 8.0
M5_NUT_T = 4.0
M5_HEAD_D = 8.5
M5_HEAD_H = 5.0
M5_WASHER_D = 10.0
M5_WASHER_T = 1.0
TILT_BOLT_LEN = 30.0

# -- Camera clamp (collar + tilt fork) ------------------------------------------
CLAMP_WALL = 5.0
CLAMP_H = 40.0
CHEEK_T = 5.0
FORK_GAP = 11.4
PIVOT_Y = 40.0  # pivot axis in front of the bar axis
PIVOT_Z = CLAMP_H / 2  # pivot above the clamp's bottom face
CHEEK_R = CLAMP_H / 2
LOCK_R = 12.0  # tilt-lock bolt distance from the pivot
SLOT_W = 5.6
TILT_MIN, TILT_MAX = -10.0, 90.0  # degrees below horizontal
CLAMP_LOCK_Z = PAD_SCREW_FROM_END  # pad hangs from z = PAD_LEN down to the free end at z = 0

# -- Camera cradle --------------------------------------------------------------
TONGUE_T = 11.0
HUB_R = 8.5
LOBE_R = 6.5
ARM_L = 32.0  # pivot axis -> camera back face
CRADLE_PLATE_T = 5.0
M3_CLEAR_D = 3.4
BRIDGE_Y1 = PIVOT_Y - LOCK_R - LOBE_R - 1.0  # solid bridge between the cheeks ends here

# -- Tilt wing knob (holds a normal M5 nut) -------------------------------------
KNOB_BASE_T = 2.5  # under the nut
KNOB_HUB_R = 6.25  # narrow part, level with the pivot bolt head it must clear
KNOB_WING_Z0 = M5_HEAD_H + 0.5  # wings start above the pivot bolt head
KNOB_H = KNOB_WING_Z0 + 5.0
KNOB_TOP_R = 8.0
KNOB_WING_HALF = 14.0
KNOB_WING_T = 5.0
KNOB_NUT_AF = M5_NUT_AF + 0.2

# -- RealSense D405 (datasheet 337029-017: Fig. 10-14, Tables 3-54, 4-17, 4-22) --
CAM_W = 42.0
CAM_H = 42.0
CAM_D = 23.0
CAM_CORNER_R = 4.0  # approximate
CAM_M3_DX = 20.0  # 2x M3x0.5 on the back, 20 mm apart, centred; max 4.0 mm deep
CAM_ORIGIN_BEHIND_GLASS = 3.7  # depth start point behind the front glass
CAM_LEFT_IMAGER_X = -9.0  # left imager is 9 mm left of the 1/4-20 centreline (seen from behind)

# -- Desk foot ------------------------------------------------------------------
FOOT_X0, FOOT_X1 = -100.0, 100.0
FOOT_Y0, FOOT_Y1 = -35.0, 55.0  # rear edge (operator side) / front edge
FOOT_PLATE_T = 8.0
FOOT_WALL = 6.0
FOOT_SOCKET_DEPTH = 70.0
FOOT_TOP = FOOT_PLATE_T + FOOT_SOCKET_DEPTH
# two pads on the back wall, both standing up from their root (prints without
# floating layers): the lower one grows from the socket floor and ends under a
# slit, the upper one ends at the open socket top; each screw sits near a free end
FOOT_PAD_LO = (FOOT_PLATE_T, FOOT_PLATE_T + PAD_LEN)
FOOT_PAD_HI = (FOOT_TOP - PAD_LEN, FOOT_TOP)
FOOT_LOCK_Z = (FOOT_PAD_LO[1] - PAD_SCREW_FROM_END, FOOT_TOP - PAD_SCREW_FROM_END)
FOOT_CLAMP_ZONE_X = 55.0  # |x| >= this is kept flat for the desk clamps


def camera_origin_local():
    """D405 left-imager origin in the cradle frame (pivot at 0, tilt 0, +Y = view)."""
    return (CAM_LEFT_IMAGER_X, ARM_L + CAM_D - CAM_ORIGIN_BEHIND_GLASS, 0.0)


def t_world_camera(collar_height, tilt_deg):
    """4x4 T_world_camera as nested lists, translation in metres.

    collar_height: desk surface -> bottom face of the camera clamp, in mm.
    tilt_deg: camera pitch below horizontal (the number on the tick scale).
    """
    t = math.radians(tilt_deg)
    s, c = math.sin(t), math.cos(t)
    x, y, z = camera_origin_local()
    px = x
    py = PIVOT_Y + y * c + z * s
    pz = collar_height + PIVOT_Z - y * s + z * c
    # columns: camera x (right), y (down), z (optical axis) expressed in world
    rot = [[1.0, 0.0, 0.0], [0.0, -s, c], [0.0, -c, -s]]
    return [
        rot[0] + [px / 1000.0],
        rot[1] + [py / 1000.0],
        rot[2] + [pz / 1000.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collar-height", type=float, required=True,
                    help="desk surface to the bottom face of the camera clamp, mm")
    ap.add_argument("--tilt", type=float, default=45.0, help="tilt below horizontal, degrees")
    args = ap.parse_args()
    if not TILT_MIN <= args.tilt <= TILT_MAX:
        ap.error(f"tilt must be within [{TILT_MIN}, {TILT_MAX}]")
    for row in t_world_camera(args.collar_height, args.tilt):
        print(" ".join(f"{v: .6f}" for v in row))


if __name__ == "__main__":
    main()
