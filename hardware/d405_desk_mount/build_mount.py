"""FreeCAD build script for the RealSense D405 desk mount (egohand-live).

Headless:  cd hardware/d405_desk_mount && freecadcmd build_mount.py
FreeCAD GUI Python console:
           exec(open("/path/to/build_mount.py").read(), {"__file__": "/path/to/build_mount.py"})

Writes next to this file:
  stl/*.stl, step/*.step   the four printable parts, already in print orientation
  d405_desk_mount.FCStd    assembly in the use pose (PREVIEW_TILT, PREVIEW_COLLAR_H)
All dimensions live in mount_geometry.py.
"""
import importlib
import math
import os
import sys

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # freecadcmd without __file__
    HERE = os.environ.get("MOUNT_DIR", os.getcwd())
sys.path.insert(0, HERE)
import mount_geometry as G  # noqa: E402

importlib.reload(G)

import FreeCAD as App  # noqa: E402
import MeshPart  # noqa: E402
import Part  # noqa: E402

V = App.Vector
O, X, Y, Z = V(0, 0, 0), V(1, 0, 0), V(0, 1, 0), V(0, 0, 1)

PREVIEW_TILT = 45.0
PREVIEW_COLLAR_H = 415.0  # desk -> bottom of camera clamp, mm (lens ~40 cm above desk)

GO = G.FORK_GAP / 2 + G.CHEEK_T  # cheek outer face |x|


# ----------------------------------------------------------------------------- helpers
def box(x0, x1, y0, y1, z0, z1):
    return Part.makeBox(x1 - x0, y1 - y0, z1 - z0, V(x0, y0, z0))


def cyl_x(r, y, z, x0, x1):
    return Part.makeCylinder(r, x1 - x0, V(x0, y, z), X)


def cyl_y(r, x, z, y0, y1):
    return Part.makeCylinder(r, y1 - y0, V(x, y0, z), Y)


def cyl_z(r, x, y, z0, z1):
    return Part.makeCylinder(r, z1 - z0, V(x, y, z0), Z)


def hex_z(af, x, y, z0, z1):
    r = af / 2 / math.cos(math.radians(30))
    pts = [V(x + r * math.cos(math.radians(60 * i)), y + r * math.sin(math.radians(60 * i)), z0) for i in range(6)]
    return Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(V(0, 0, z1 - z0))


def prism_x(pts_yz, x0, x1):
    pts = [V(x0, y, z) for y, z in pts_yz]
    return Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(V(x1 - x0, 0, 0))


def prism_y(pts_xz, y0, y1):
    pts = [V(x, y0, z) for x, z in pts_xz]
    return Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(V(0, y1 - y0, 0))


def rotated(shape, center, axis, deg):
    s = shape.copy()
    s.rotate(center, axis, deg)
    return s


def moved(shape, vec):
    s = shape.copy()
    s.translate(vec)
    return s


def fillet_parallel(shape, axis, r):
    """Fillet the straight edges of a simple solid that run parallel to axis."""
    edges = [e for e in shape.Edges
             if isinstance(e.Curve, Part.Line) and abs(abs(e.Curve.Direction.dot(axis)) - 1) < 1e-6]
    return shape.makeFillet(r, edges)


def fuse_all(shapes):
    return shapes[0].fuse(shapes[1:]).removeSplitter()


def cut_all(base, cutters):
    out = base.cut(cutters).removeSplitter()
    solids = out.Solids
    if len(solids) != 1:
        raise RuntimeError(f"expected 1 solid, got {len(solids)}")
    return solids[0]


def lead_in(half, z_face, depth, direction):
    """45 deg lead-in chamfer cutter for a square bore opening at z_face.
    direction=+1: bore goes upward from z_face, -1: downward."""
    def sq(h, z):
        p = [V(-h, -h, z), V(h, -h, z), V(h, h, z), V(-h, h, z)]
        return Part.makePolygon(p + [p[0]])
    w1 = sq(half + depth + 0.5, z_face - direction * 0.5)
    w2 = sq(half, z_face + direction * depth)
    return Part.makeLoft([w1, w2], True)


def arc_slot_x(cy, cz, r, w, a0, a1, x0, x1):
    """Arc slot (axis along X) centred on (cy, cz), radius r, width w, from angle a0 to a1
    (degrees, measured from +Y towards +Z)."""
    ann = cyl_x(r + w / 2, cy, cz, x0, x1).cut(cyl_x(r - w / 2, cy, cz, x0, x1))
    n = 16
    pts = [(cy, cz)] + [(cy + 3 * r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                         cz + 3 * r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
    s = ann.common(prism_x(pts, x0, x1))
    for a in (a0, a1):
        s = s.fuse(cyl_x(w / 2, cy + r * math.cos(math.radians(a)), cz + r * math.sin(math.radians(a)), x0, x1))
    return s.removeSplitter()


def tick_x(cy, cz, a_deg, r1, r2, width, x_face, depth, side):
    """Engraving cutter for a radial tick on the face at x_face whose normal is side*X."""
    if side > 0:
        t = box(x_face - depth, x_face + 1, r1, r2, -width / 2, width / 2)
    else:
        t = box(x_face - 1, x_face + depth, r1, r2, -width / 2, width / 2)
    t.rotate(O, X, a_deg)
    t.translate(V(0, cy, cz))
    return t


def pad_cutters(z_lo, z_hi, end_slit=None):
    """Cut a flexible pad out of the inner layer of the -Y bore wall between z_lo and
    z_hi: two side slits plus a gap behind it. The pad stays attached at its root; its
    free end is either an open face of the part or gets an end slit ('bottom'/'top')."""
    b, w, g = G.BORE / 2, G.PAD_W / 2, G.PAD_GAP
    y_back = G.PAD_BACK_Y
    cut = [box(-w - g, -w, y_back - g, -b + 0.01, z_lo, z_hi),
           box(w, w + g, y_back - g, -b + 0.01, z_lo, z_hi),
           box(-w - g, w + g, y_back - g, y_back, z_lo, z_hi)]
    e = G.PAD_END_SLIT
    if end_slit == "bottom":
        cut.append(box(-w - g, w + g, y_back - g, -b + 0.01, z_lo, z_lo + e))
    elif end_slit == "top":
        cut.append(box(-w - g, w + g, y_back - g, -b + 0.01, z_hi - e, z_hi))
    return cut


def lock_cutters(z, x_side):
    """M4 screw hole (along +Y, x = 0, height z) through the rear boss into the gap behind
    the pad, plus a side-loading M4 nut slot that opens at x = x_side."""
    y_end = G.PAD_BACK_Y - G.PAD_GAP / 2  # stop inside the gap, don't touch the pad
    hole = cyl_y(G.M4_CLEAR_D / 2, 0, z, G.LOCK_BOSS_Y - 1, y_end)
    ac = G.M4_NUT_AF / math.cos(math.radians(30))
    slot = box(-ac / 2 - 0.1, x_side + 1, G.LOCK_NUT_Y0, G.LOCK_NUT_Y0 + G.M4_NUT_T,
               z - G.M4_NUT_AF / 2, z + G.M4_NUT_AF / 2)
    return [hole, slot]


# ----------------------------------------------------------------------------- parts
def build_cam_clamp():
    """Collar + tilt fork. Frame: bar axis = Z, collar bottom at z=0, camera looks +Y."""
    b = G.BORE / 2
    o = b + G.CLAMP_WALL
    H = G.CLAMP_H
    yp, zp, R = G.PIVOT_Y, G.PIVOT_Z, G.CHEEK_R
    gi = G.FORK_GAP / 2
    y0 = o - 1.0

    solids = [fillet_parallel(box(-o, o, -o, o, 0, H), Z, 2.0)]
    rect = [(y0, 0), (yp, 0), (yp, H), (y0, H)]
    k = math.sqrt(0.5)
    # 45 deg support under the front-lower quadrant of the cheek disc (prints without support)
    chamf = [(yp, 0), (yp + R * (2 * k - 1), 0), (yp + R * k, zp - R * k), (yp, zp)]
    for xa, xb in ((gi, GO), (-GO, -gi)):
        solids += [prism_x(rect, xa, xb), prism_x(chamf, xa, xb), cyl_x(R, yp, zp, xa, xb)]
    solids.append(box(-GO, GO, y0, G.BRIDGE_Y1, 0, H))  # bridge between the cheeks
    # rear boss for the M4 lock screw, 45 deg roof so it prints without support
    yb, yw = G.LOCK_BOSS_Y, -(o - 1.0)
    roof = 16.0
    solids.append(prism_x([(yw, 0), (yb, 0), (yb, roof), (yw, roof + (yw - yb))], -o, o))
    body = fuse_all(solids)

    cutters = [box(-b, b, -b, b, -1, H + 1), lead_in(b, 0, 1.0, +1), lead_in(b, H, 1.0, -1),
               cyl_x(G.M5_CLEAR_D / 2, yp, zp, -GO - 1, GO + 1)]
    # tilt-lock slot: the lock bolt sits at angle (180 - tilt) around the pivot
    cutters.append(arc_slot_x(yp, zp, G.LOCK_R, G.SLOT_W, 180 - G.TILT_MAX, 180 - G.TILT_MIN, -GO - 1, GO + 1))
    # angle scale on the -X cheek (bolt-head side, outside the washer): every 15 deg,
    # long ticks at 0 / 45 / 90
    r1 = G.LOCK_R + G.M5_WASHER_D / 2 + 0.5
    for tilt in range(0, 91, 15):
        r2 = 19.5 if tilt in (0, 45, 90) else 18.7
        cutters.append(tick_x(yp, zp, 180 - tilt, r1, r2, 0.8, -GO, 0.6, side=-1))
    # bar lock: pad hangs from z = PAD_LEN down to the open bottom face
    cutters += pad_cutters(-1.0, G.PAD_LEN)
    cutters += lock_cutters(G.CLAMP_LOCK_Z, o)
    return cut_all(body, cutters)


def build_cradle():
    """Camera cradle. Frame: pivot axis = X through the origin, tilt 0, camera looks +Y,
    camera back face at y = ARM_L."""
    L, pt = G.ARM_L, G.CRADLE_PLATE_T
    hw = G.CAM_W / 2
    tt = G.TONGUE_T / 2
    hub = G.HUB_R
    solids = [
        fillet_parallel(box(-hw, hw, L - pt, L, -G.CAM_H / 2, G.CAM_H / 2), Y, G.CAM_CORNER_R),
        box(-tt, tt, 0, L - pt + 0.5, -hub, hub),  # arm
        cyl_x(hub, 0, 0, -tt, tt),  # pivot hub
        cyl_x(G.LOBE_R, -G.LOCK_R, 0, -tt, tt),  # lock lobe
        box(-tt, tt, -G.LOCK_R, 0, -G.LOBE_R, G.LOBE_R),
        # gussets between arm and back plate
        prism_x([(L - pt + 0.5, hub), (L - pt + 0.5, hub + 7.5), (L - pt - 8.0, hub)], -tt, tt),
        prism_x([(L - pt + 0.5, -hub), (L - pt + 0.5, -hub - 7.5), (L - pt - 8.0, -hub)], -tt, tt),
    ]
    body = fuse_all(solids)
    cutters = [cyl_x(G.M5_CLEAR_D / 2, 0, 0, -tt - 1, tt + 1),
               cyl_x(G.M5_CLEAR_D / 2, -G.LOCK_R, 0, -tt - 1, tt + 1)]
    for sx in (-1, 1):
        cutters.append(cyl_y(G.M3_CLEAR_D / 2, sx * G.CAM_M3_DX / 2, 0, L - pt - 1, L + 1))
    return cut_all(body, cutters)


def build_knob():
    """Wing knob for a normal M5 nut. Frame: axis = Z, contact face at z=0. The narrow
    hub stays clear of the pivot bolt head next to it; the wings sit above that head."""
    wh, wt = G.KNOB_WING_HALF, G.KNOB_WING_T
    solids = [cyl_z(G.KNOB_HUB_R, 0, 0, 0, G.KNOB_WING_Z0 + 0.01),
              cyl_z(G.KNOB_TOP_R, 0, 0, G.KNOB_WING_Z0, G.KNOB_H),
              box(-(wh - wt / 2), wh - wt / 2, -wt / 2, wt / 2, G.KNOB_WING_Z0, G.KNOB_H),
              cyl_z(wt / 2, wh - wt / 2, 0, G.KNOB_WING_Z0, G.KNOB_H),
              cyl_z(wt / 2, -(wh - wt / 2), 0, G.KNOB_WING_Z0, G.KNOB_H)]
    body = fuse_all(solids)
    cutters = [cyl_z(G.M5_CLEAR_D / 2, 0, 0, -1, G.KNOB_H + 1),
               hex_z(G.KNOB_NUT_AF, 0, 0, G.KNOB_BASE_T, G.KNOB_H + 1)]
    return cut_all(body, cutters)


def build_foot():
    """Desk foot. Frame: desk surface at z=0, bar axis = Z, camera looks +Y, operator at -Y."""
    b = G.BORE / 2
    o = b + G.FOOT_WALL
    T, top = G.FOOT_PLATE_T, G.FOOT_TOP
    solids = [
        fillet_parallel(box(G.FOOT_X0, G.FOOT_X1, G.FOOT_Y0, G.FOOT_Y1, 0, T), Z, 12.0),
        fillet_parallel(box(-o, o, -o, o, 0, top), Z, 2.0),
        prism_x([(o - 1, T), (G.FOOT_Y1 - 3, T), (G.FOOT_Y1 - 3, T + 3), (o - 1, T + 55)], -3, 3),
        prism_y([(o - 1, T), (50, T), (50, T + 4), (o - 1, T + 55)], -3, 3),
        prism_y([(-(o - 1), T), (-50, T), (-50, T + 4), (-(o - 1), T + 55)], -3, 3),
        # rear boss for the two M4 lock screws (also the rear gusset)
        fillet_parallel(box(-o, o, G.LOCK_BOSS_Y, -(o - 1.0), 0, top), Z, 2.0),
    ]
    body = fuse_all(solids)

    cutters = [box(-b, b, -b, b, T, top + 1), lead_in(b, top, 1.0, -1),
               cyl_z(3.0, 0, 0, -1, T + 1)]  # vent / push-out hole
    for sx in (-1, 1):  # optional wood-screw holes (countersunk, 4 mm screws)
        for y in (-20.0, 35.0):
            x = sx * 85.0
            cutters.append(cyl_z(2.25, x, y, -1, T + 1))
            cutters.append(Part.makeCone(2.25, 4.75, 2.5, V(x, y, T - 2.5), Z))
            cutters.append(cyl_z(4.75, x, y, T - 0.01, T + 1))
    # both pads stand up from their root: the lower one ends under a slit, the upper one
    # at the open top
    cutters += pad_cutters(T, G.FOOT_PAD_LO[1] + G.PAD_END_SLIT, end_slit="top")
    cutters += pad_cutters(G.FOOT_PAD_HI[0], top + 1.0)
    for z in G.FOOT_LOCK_Z:
        cutters += lock_cutters(z, o)
    return cut_all(body, cutters)


# ----------------------------------------------------------------------------- reference geometry
def build_d405():
    """D405 stand-in, in the cradle frame."""
    L = G.ARM_L
    body = fillet_parallel(box(-G.CAM_W / 2, G.CAM_W / 2, L, L + G.CAM_D, -G.CAM_H / 2, G.CAM_H / 2), Y,
                           G.CAM_CORNER_R)
    usb_bump = cyl_y(6.0, -G.CAM_W / 2, 0, L + 1.0, L + 13.0)
    plug = box(-G.CAM_W / 2 - 5 - 18, -G.CAM_W / 2 - 3, L + 6.82 - 4, L + 6.82 + 4, -5.5, 5.5)
    return fuse_all([body, usb_bump, plug])


def build_m3_heads():
    L, pt = G.ARM_L, G.CRADLE_PLATE_T
    return fuse_all([cyl_y(2.75, sx * G.CAM_M3_DX / 2, 0, L - pt - 3.0, L - pt) for sx in (-1, 1)])


def knob_on_lock_bolt(knob, spin=0.0):
    """Knob placed on the +X cheek face at the lock bolt, in the cradle frame at tilt 0."""
    s = rotated(knob, O, Z, spin)
    s.rotate(O, Y, 90)  # knob axis -> +X
    s.translate(V(GO, -G.LOCK_R, 0))
    return s


def build_pivot_hw():
    """Pivot M5x30: head on +X, plain nut on -X. Static (does not move with the tilt)."""
    L = G.TILT_BOLT_LEN
    return fuse_all([cyl_x(G.M5_HEAD_D / 2, 0, 0, GO, GO + G.M5_HEAD_H),
                     cyl_x(2.5, 0, 0, GO - L, GO),
                     cyl_x(G.M5_NUT_AF / 2 / math.cos(math.radians(30)), 0, 0, -GO - G.M5_NUT_T, -GO)])


def build_lock_bolt_hw():
    """Lock M5x30 with washer + head on -X; nut is inside the knob. Cradle frame, tilt 0."""
    y = -G.LOCK_R
    seat = -GO - G.M5_WASHER_T
    return fuse_all([cyl_x(G.M5_WASHER_D / 2, y, 0, seat, -GO),
                     cyl_x(G.M5_HEAD_D / 2, y, 0, seat - G.M5_HEAD_H, seat),
                     cyl_x(2.5, y, 0, seat, seat + G.TILT_BOLT_LEN)])


def build_lock_screws(z_list):
    """M4x16 lock screws in the rear boss, head LOCK_HEAD_STANDOFF off the boss."""
    parts = []
    for z in z_list:
        y_head = G.LOCK_BOSS_Y - G.LOCK_HEAD_STANDOFF
        parts += [cyl_y(2.0, 0, z, y_head, y_head + G.LOCK_SCREW_LEN),
                  cyl_y(3.5, 0, z, y_head - 4.0, y_head)]
    return fuse_all(parts)


def fov_frustum(collar_h, tilt):
    """Camera view volume down to the desk plane, in world coordinates."""
    T = G.t_world_camera(collar_h, tilt)
    c = V(T[0][3] * 1000, T[1][3] * 1000, T[2][3] * 1000)
    cols = [V(T[0][i], T[1][i], T[2][i]) for i in range(3)]
    hx, hy = math.tan(math.radians(84 / 2)), math.tan(math.radians(58 / 2))
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        d = cols[0] * (sx * hx) + cols[1] * (sy * hy) + cols[2]
        d.normalize()
        t = -c.z / d.z if d.z < -1e-6 else 1500.0
        corners.append(c + d * min(t, 1500.0))
    faces = [Part.Face(Part.makePolygon([c, corners[i], corners[(i + 1) % 4], c])) for i in range(4)]
    faces.append(Part.Face(Part.makePolygon(corners + [corners[0]])))
    return Part.Solid(Part.Shell(faces))


def cradle_pose(shape, tilt, clamp_z=0.0):
    """Cradle frame -> clamp frame (or world, with clamp_z = collar height)."""
    s = rotated(shape, O, X, -tilt)
    s.translate(V(0, G.PIVOT_Y, clamp_z + G.PIVOT_Z))
    return s


def pivot_pose(shape, clamp_z=0.0):
    return moved(shape, V(0, G.PIVOT_Y, clamp_z + G.PIVOT_Z))


# ----------------------------------------------------------------------------- checks
def check(clamp, cradle, foot, knob):
    print("== checks ==")
    for name, s in (("cam_clamp", clamp), ("cam_cradle", cradle), ("desk_foot", foot), ("tilt_knob", knob)):
        bb = s.optimalBoundingBox(True, False)
        print(f"{name:11s} valid={s.isValid()} solids={len(s.Solids)} vol={s.Volume / 1000:.1f} cm3 "
              f"size={bb.XLength:.1f} x {bb.YLength:.1f} x {bb.ZLength:.1f} mm")
    bar_c = box(-G.BAR / 2, G.BAR / 2, -G.BAR / 2, G.BAR / 2, -600, 600)
    print(f"bar vs clamp bore overlap: {bar_c.common(clamp).Volume:.4f} mm3")
    bar_f = box(-G.BAR / 2, G.BAR / 2, -G.BAR / 2, G.BAR / 2, G.FOOT_PLATE_T, 900)
    print(f"bar vs foot socket overlap: {bar_f.common(foot).Volume:.4f} mm3")

    static = fuse_all([clamp, pivot_pose(build_pivot_hw())])
    moving = fuse_all([cradle, build_d405(), build_m3_heads(), build_lock_bolt_hw()])
    worst, worst_bar, worst_knob = (-1.0, None), (-1.0, None), (-1.0, None)
    for tilt in range(int(G.TILT_MIN), int(G.TILT_MAX) + 1, 5):
        m = cradle_pose(moving, tilt)
        worst = max(worst, (m.common(static).Volume, tilt))
        worst_bar = max(worst_bar, (m.common(bar_c).Volume, tilt))
        for spin in range(0, 180, 30):  # wing orientation on the lock bolt
            k = cradle_pose(knob_on_lock_bolt(knob, spin), tilt)
            vk = k.common(static).Volume + k.common(m).Volume
            worst_knob = max(worst_knob, (vk, (tilt, spin)))
    print(f"tilt sweep {G.TILT_MIN:.0f}..{G.TILT_MAX:.0f} deg: max overlap cradle+camera+bolt vs clamp+pivot "
          f"{worst[0]:.4f} mm3 (tilt {worst[1]}), vs bar {worst_bar[0]:.4f} mm3 (tilt {worst_bar[1]}), "
          f"wing knob (all spins) {worst_knob[0]:.4f} mm3 {worst_knob[1]}")
    plate_cam = fuse_all([build_d405(), build_m3_heads(),
                          box(-G.CAM_W / 2, G.CAM_W / 2, G.ARM_L - G.CRADLE_PLATE_T, G.ARM_L,
                              -G.CAM_H / 2, G.CAM_H / 2)])
    gaps = [(clamp.distToShape(cradle_pose(plate_cam, t))[0], t) for t in range(-10, 91, 10)]
    print("min gap back plate/camera <-> clamp: %.2f mm (tilt %d)" % min(gaps))
    gaps_bar = [(bar_c.distToShape(cradle_pose(plate_cam, t))[0], t) for t in range(-10, 91, 10)]
    print("min gap camera <-> bar: %.2f mm (tilt %d)" % min(gaps_bar))
    hub_gap = G.LOCK_R - G.M5_HEAD_D / 2 - G.KNOB_HUB_R
    print(f"knob hub <-> pivot bolt head: {hub_gap:.2f} mm; wings start {G.KNOB_WING_Z0 - G.M5_HEAD_H:.1f} mm "
          f"above the head")

    grip = 2 * G.CHEEK_T + G.FORK_GAP
    piv = grip + G.M5_NUT_T
    lock = G.M5_WASHER_T + grip + G.KNOB_BASE_T + G.M5_NUT_T
    print(f"pivot M5x{G.TILT_BOLT_LEN:.0f}: needs {piv:.1f} mm (+{G.TILT_BOLT_LEN - piv:.1f}); "
          f"lock M5x{G.TILT_BOLT_LEN:.0f} with washer + knob: needs {lock:.1f} mm (+{G.TILT_BOLT_LEN - lock:.1f})")
    nut_front = G.LOCK_NUT_Y0 + G.M4_NUT_T
    tip = G.LOCK_BOSS_Y - G.LOCK_HEAD_STANDOFF + G.LOCK_SCREW_LEN
    print(f"lock screw M4x{G.LOCK_SCREW_LEN:.0f}: nut y {G.LOCK_NUT_Y0:.1f}..{nut_front:.1f}, tip touches pad "
          f"at y {tip:.1f} with the head {G.LOCK_HEAD_STANDOFF:.1f} mm off the boss")
    T = G.t_world_camera(PREVIEW_COLLAR_H, PREVIEW_TILT)
    print(f"preview: tilt {PREVIEW_TILT} deg, collar {PREVIEW_COLLAR_H} mm -> camera origin at "
          f"x={T[0][3] * 1000:.1f} y={T[1][3] * 1000:.1f} z={T[2][3] * 1000:.1f} mm")


# ----------------------------------------------------------------------------- export + assembly
def export(shape, name):
    os.makedirs(os.path.join(HERE, "stl"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "step"), exist_ok=True)
    bb = shape.optimalBoundingBox(True, False)
    s = moved(shape, V(-bb.Center.x, -bb.Center.y, -bb.ZMin))
    mesh = MeshPart.meshFromShape(Shape=s, LinearDeflection=0.02, AngularDeflection=0.15, Relative=False)
    mesh.write(os.path.join(HERE, "stl", name + ".stl"))
    s.exportStep(os.path.join(HERE, "step", name + ".step"))
    print(f"wrote stl/{name}.stl ({mesh.CountFacets} facets), step/{name}.step")


def main():
    clamp = build_cam_clamp()
    cradle = build_cradle()
    foot = build_foot()
    knob = build_knob()
    check(clamp, cradle, foot, knob)

    export(clamp, "cam_clamp")  # print as modelled: collar bottom on the bed
    export(moved(rotated(cradle, O, X, -90), V(0, 0, G.ARM_L)), "cam_cradle")  # camera face on the bed
    export(foot, "desk_foot")  # plate on the bed
    export(rotated(knob, O, X, 180), "tilt_knob")  # wings on the bed

    name = "d405_desk_mount"
    if name in App.listDocuments():
        App.closeDocument(name)
    doc = App.newDocument(name)
    zc = PREVIEW_COLLAR_H
    tilt = PREVIEW_TILT
    grey = (0.6, 0.6, 0.6)
    items = [
        ("DeskFoot", foot, (0.95, 0.55, 0.15)),
        ("Bar_20x20x880", box(-G.BAR / 2, G.BAR / 2, -G.BAR / 2, G.BAR / 2, G.FOOT_PLATE_T,
                              G.FOOT_PLATE_T + G.BAR_LEN), (0.75, 0.77, 0.80)),
        ("CameraClamp", moved(clamp, V(0, 0, zc)), (0.95, 0.55, 0.15)),
        ("CameraCradle", cradle_pose(cradle, tilt, zc), (0.20, 0.45, 0.85)),
        ("TiltKnob", cradle_pose(knob_on_lock_bolt(knob, 0), tilt, zc), (0.20, 0.45, 0.85)),
        ("D405", cradle_pose(build_d405(), tilt, zc), (0.12, 0.12, 0.12)),
        ("M5_PivotBolt", pivot_pose(build_pivot_hw(), zc), grey),
        ("M5_LockBolt", cradle_pose(build_lock_bolt_hw(), tilt, zc), grey),
        ("M3_CameraScrews", cradle_pose(build_m3_heads(), tilt, zc), grey),
        ("M4_LockScrew_Clamp", moved(build_lock_screws([G.CLAMP_LOCK_Z]), V(0, 0, zc)), grey),
        ("M4_LockScrews_Foot", build_lock_screws(list(G.FOOT_LOCK_Z)), grey),
        ("Desk", box(-450, 450, G.FOOT_Y0, 700, -25, 0), (0.85, 0.78, 0.65)),
        ("CameraFOV", fov_frustum(zc, tilt), (0.3, 0.8, 0.3)),
    ]
    for label, shape, color in items:
        obj = doc.addObject("Part::Feature", label)
        obj.Shape = shape
        if App.GuiUp:
            obj.ViewObject.ShapeColor = color
            if label == "CameraFOV":
                obj.ViewObject.Transparency = 80
            if label == "Desk":
                obj.ViewObject.Transparency = 30
    doc.recompute()
    path = os.path.join(HERE, name + ".FCStd")
    doc.saveAs(path)
    print("wrote", path)
    return doc


main()
