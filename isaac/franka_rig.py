"""Franka FR3 Duo with Sharpa Wave hands on the flanges, for isaac/sharpa_hands.py --rig franka.

Composed in USD from two unmodified assets:

- the FR3 Duo from Franka Robotics' SimReady assets (``frankarobotics/fr3_duo``): two FR3 v2.1
  arms on the Duo mount, one articulation, Franka's drive tuning;
- Sharpa's Isaac Lab hand (``{side}_sharpa_wave_with_flange.usda``), with its world-anchoring
  ``root_joint`` deactivated and its flange fixed to the arm's flange (link7 + 107 mm), so the
  hand's links join the Duo articulation and keep Sharpa's finger tuning.

The Duo faces +y (across the desk). The tracked wrist pose drives each hand's wrist link
(``{side}_hand_C_MC``) through damped-least-squares IK on that arm's seven joints, with a
null-space pull toward a home posture; the fingers keep their DexPilot targets.

Placement and hand angle come from an IK reachability sweep (pinocchio, FR3 kinematics) over
the desk workspace (x +-0.4 m, y 0.15-0.6 m, z 0.06-0.2 m, palm down, fingers forward, +-30 deg
yaw): with the mount 0.20 m above the desk at y = -0.26 m and the hands at -45 deg on the
flange (as Franka mounts its own gripper), both arms reach the whole grid with >= 0.5 rad
joint margin at home. At desk height the best placement reached ~75 %.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
SHARPA_DIR = REPO / "third_party" / "sharpa-urdf-usd-xml" / "wave_01"
DUO_USD = Path("frankarobotics") / "fr3_duo" / "simready_usd" / "fr3_duo.usda"  # under the SimReady root

SIDES = ("left", "right")
ARM_JOINTS = {s: [f"{s}_fr3v2_1_joint{i}" for i in range(1, 8)] for s in SIDES}
FINGER_JOINTS = {s: f"{s}_(thumb|index|middle|ring|pinky)_.*" for s in SIDES}
LINK7 = {s: f"{s}_fr3v2_1_link7" for s in SIDES}
FLANGE_OFFSET = 0.107  # link7 -> flange (FR3 link8)
HAND_MOUNT_YAW_DEG = -45.0
DUO_POS = (0.0, -0.26, 0.20)  # mount point; z above the desk surface
DUO_YAW_DEG = 90.0  # the Duo faces +x; turn it to face +y
# Home postures (deg) for the default placement: hand parked at PARK_POS, palm down.
HOME_DEG = {"left": [-57, -50, 109, -137, -136, 121, 116], "right": [52, -63, -105, -135, 133, 130, -35]}
READY_DEG = [0, -45, 0, -135, 0, 90, 45]  # Duo default posture
PARK_POS = {"left": np.array([-0.25, 0.33, 0.18]), "right": np.array([0.25, 0.33, 0.18])}
# Palm down, fingers forward (+y): hand base x (palmar) = -z_world, z (fingers) = +y_world.
PALM_DOWN_FORWARD = Rotation.from_matrix(np.column_stack([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])).as_quat()

VEL_LIMIT = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])  # FR3 joint speed limits [rad/s]
DRIVE_TIME_CONSTANT = 0.1  # damping / stiffness of the Duo's joint drives [s]


def duo_usd(simready_root: Path) -> Path:
    path = Path(simready_root) / DUO_USD
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; --simready-root must contain frankarobotics/fr3_duo (SimReady)")
    return path


def hand_usd(side: str) -> Path:
    path = SHARPA_DIR / f"{side}_sharpa_wave" / f"{side}_sharpa_wave_with_flange.usda"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run scripts/setup_isaac.sh")
    return path


def articulation_cfg(prim_path: str, simready_root: Path, home_deg=HOME_DEG):
    """Isaac Lab ArticulationCfg for the Duo (hands are added by ``attach_hands``)."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

    yaw = math.radians(DUO_YAW_DEG)
    joint_pos = {j: math.radians(a) for s in SIDES for j, a in zip(ARM_JOINTS[s], home_deg[s])}
    joint_pos[".*_(thumb|index|middle|ring|pinky)_.*"] = 0.0
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(duo_usd(simready_root)),
            rigid_props=PhysxRigidBodyCfg(disable_gravity=True, max_depenetration_velocity=1.0),
            articulation_props=PhysxArticulationCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DUO_POS, rot=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)), joint_pos=joint_pos
        ),
        # Both assets ship drive gains tuned for Isaac (None = use the USD values).
        actuators={
            "arms": ImplicitActuatorCfg(joint_names_expr=[".*_fr3v2_1_joint[1-7]"], stiffness=None, damping=None),
            "fingers": ImplicitActuatorCfg(joint_names_expr=[".*_(thumb|index|middle|ring|pinky)_.*"],
                                           stiffness=None, damping=None),  # fmt: skip
        },
        soft_joint_pos_limit_factor=1.0,
    )


def attach_hands(stage, duo_prim: str) -> dict[str, str]:
    """Reference a Sharpa hand per side under the Duo prim and fix its flange to link7.

    Must run after the Duo is spawned and before ``sim.reset()``. Returns the hand prim paths.
    """
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    duo = stage.GetPrimAtPath(duo_prim)
    link7 = {p.GetName(): p for p in Usd.PrimRange(duo) if p.GetName() in LINK7.values()}
    cache = UsdGeom.XformCache()
    T_duo = cache.GetLocalToWorldTransform(duo)
    half = math.radians(HAND_MOUNT_YAW_DEG) / 2
    mount_rot = Gf.Quatf(math.cos(half), 0.0, 0.0, math.sin(half))
    T_mount = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), HAND_MOUNT_YAW_DEG))
    T_mount.SetTranslateOnly(Gf.Vec3d(0, 0, FLANGE_OFFSET))
    paths = {}
    for side in SIDES:
        path = f"{duo_prim}/{side.capitalize()}Hand"
        hand = stage.DefinePrim(path, "Xform")
        hand.GetReferences().AddReference(str(hand_usd(side)))
        for p in Usd.PrimRange(hand):
            if p.IsA(UsdPhysics.Joint) and not UsdPhysics.Joint(p).GetBody0Rel().GetTargets():
                p.SetActive(False)  # world anchor; also carries the hand's own articulation root
            elif p.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr().Set(True)
        # Start the hand where the joint will hold it (USD row-vector convention: child * parent).
        T_world = T_mount * cache.GetLocalToWorldTransform(link7[LINK7[side]])
        xf = UsdGeom.Xformable(hand)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(T_world * T_duo.GetInverse())
        joint = UsdPhysics.FixedJoint.Define(stage, f"{duo_prim}/{side.capitalize()}HandMount")
        joint.CreateBody0Rel().SetTargets([link7[LINK7[side]].GetPath()])
        joint.CreateBody1Rel().SetTargets([f"{path}/{side}_hand_flange"])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, FLANGE_OFFSET))
        joint.CreateLocalRot0Attr().Set(mount_rot)
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        paths[side] = path
    return paths


class ArmIK:
    """Damped-least-squares IK step for the seven arm joints toward a wrist-link pose.

    Each call returns joint position targets ``q + dq`` from the measured ``q``. ``dq`` is
    scaled down to ``dq_max`` per joint, so the target never runs far ahead of the arm: with
    ``dq_max = VEL_LIMIT * DRIVE_TIME_CONSTANT`` the PD drives move at most at the joint speed
    limits, which also bounds the push when the hand is blocked. A null-space term pulls the
    redundant DoF toward ``q_rest``.
    """

    def __init__(self, lower, upper, dq_max=VEL_LIMIT * DRIVE_TIME_CONSTANT, damping=0.05,
                 rot_weight=0.4, null_gain=0.1, q_rest=None, margin=0.03):  # fmt: skip
        self.lo, self.hi = np.asarray(lower) + margin, np.asarray(upper) - margin
        self.dq_max = np.broadcast_to(np.asarray(dq_max, float), (7,))
        self.lam2, self.w, self.k0 = damping**2, rot_weight, null_gain
        self.q_rest = (self.lo + self.hi) / 2 if q_rest is None else np.asarray(q_rest)

    def step(self, q, J, p, quat, goal_p, goal_quat) -> np.ndarray:
        """q (7,), J (6, 7) world-frame [linear; angular], p / quat: current wrist-link pose
        (XYZW), goal_p / goal_quat: target pose. Returns new joint targets (7,)."""
        e = np.concatenate([goal_p - p, self.w * (Rotation.from_quat(goal_quat) * Rotation.from_quat(quat).inv()).as_rotvec()])
        Jw = J.copy()
        Jw[3:] *= self.w
        J_pinv = Jw.T @ np.linalg.solve(Jw @ Jw.T + self.lam2 * np.eye(6), np.eye(6))
        dq = J_pinv @ e + (np.eye(7) - J_pinv @ Jw) @ (self.k0 * (self.q_rest - q))
        scale = np.max(np.abs(dq) / self.dq_max)
        if scale > 1.0:
            dq /= scale  # keep the direction, respect every joint's speed limit
        return np.clip(q + dq, self.lo, self.hi)
