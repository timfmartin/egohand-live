"""Desk scenes for isaac/sharpa_hands.py.

World frame (as in hardware/d405_desk_mount): desk surface at z = 0, +x to the operator's
right, +y away from the operator, +z up. The hands work roughly at y = 0.15 .. 0.6.

The ``simready`` scene uses NVIDIA SimReady props (Hugging Face dataset
``nvidia/simready-assets``, pinned below). ``scripts/fetch_simready.py`` downloads just these
props into ``third_party/simready``; ``--simready-root`` can point at an existing copy
with the same ``nvidia/<asset>/...`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIMREADY_REPO = "nvidia/simready-assets"
SIMREADY_REV = "653d91e182011f357cf1644013cc94ddf194d349"

DESK_SIZE = (1.2, 0.8, 0.04)
DESK_CENTER = (0.0, 0.35)


@dataclass(frozen=True)
class Prop:
    asset: str  # SimReady asset folder name
    x: float
    y: float
    yaw: float = 0.0  # degrees
    mass: float = 0.1  # kg (most SimReady props carry no mass; PhysX would use water density)
    hollow: bool = False  # convex decomposition (bowls, cups) instead of a convex hull
    z: float = 0.0  # extra height, e.g. to start inside a bowl

    @property
    def usd_rel(self) -> str:
        return f"nvidia/{self.asset}/simready_usd/sm_{self.asset}_01.usd"


SIMREADY_DESK: list[Prop] = [
    # back row: containers / targets
    Prop("container_system_sterilising_d01", -0.28, 0.60, 10, 0.30, hollow=True),
    Prop("kitchenware_bowl_d01", 0.00, 0.60, 0, 0.35, hollow=True),
    Prop("dishware_plastic_bigPlate_a01", 0.27, 0.60, 0, 0.15, hollow=True),
    Prop("misc_golfBall_a01", 0.02, 0.60, 0, 0.046, z=0.02),  # starts in the bowl
    Prop("food_egg_chicken_b05", 0.27, 0.60, 30, 0.06, z=0.012),  # on the plate
    # middle row: things to pick up and pour from
    Prop("dishware_mug_coffee_b01", -0.37, 0.41, 200, 0.30, hollow=True),
    Prop("food_beverage_can_b01", -0.26, 0.41, 0, 0.35),
    Prop("kitchenware_cup_a01", -0.15, 0.41, 0, 0.02, hollow=True),
    Prop("food_condiment_ketchup_a01", -0.05, 0.42, 20, 0.30),
    Prop("food_beverage_boxedDrink_a01", 0.05, 0.42, -15, 0.20),
    Prop("dishware_glass_tumbler_a01", 0.15, 0.41, 0, 0.25, hollow=True),
    Prop("food_processed_cannedFood_j01", 0.26, 0.41, 0, 0.40),
    # front row: small things to grasp
    Prop("food_produce_apple_f01", -0.36, 0.24, 0, 0.18),
    Prop("food_produce_tangerine_d01", -0.26, 0.24, 0, 0.08),
    Prop("food_produce_banana_c01", -0.14, 0.25, 60, 0.12),
    Prop("food_produce_tomato_h01", -0.03, 0.24, 0, 0.10),
    Prop("cleaning_sponge_b01", 0.07, 0.24, 15, 0.02),
    Prop("tape_clear_a02", 0.17, 0.24, 0, 0.06),
    Prop("handTool_screwdriver_phillips_ph1_a01", 0.29, 0.25, 70, 0.05),
    Prop("handTool_socket_1_1_8in_a01", 0.38, 0.24, 0, 0.08),
    Prop("food_produce_lemon_a02", -0.09, 0.12, 40, 0.10),
]

CUBES: list[tuple[float, float, tuple[float, float, float]]] = [
    (-0.12, 0.32, (0.8, 0.2, 0.2)), (0.0, 0.32, (0.2, 0.7, 0.2)), (0.12, 0.32, (0.2, 0.3, 0.8)),
]  # fmt: skip


def default_simready_root(repo: Path) -> Path:
    return repo / "third_party" / "simready"


def missing_props(root: Path, props=SIMREADY_DESK) -> list[str]:
    return [p.asset for p in props if not (root / p.usd_rel).is_file()]


def build_desk(sim_utils):
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/GroundPlane", ground, translation=(0.0, 0.0, -0.75))
    dome = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.9, 0.9, 0.92))
    dome.func("/World/DomeLight", dome)
    sun = sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0, color=(1.0, 0.97, 0.92))
    sun.func("/World/SunLight", sun, orientation=(0.2, -0.25, 0.0, 0.947))  # xyzw, from front-left above
    desk = sim_utils.CuboidCfg(
        size=DESK_SIZE,
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.29, 0.19), roughness=0.6),
    )
    desk.func("/World/Desk", desk, translation=(DESK_CENTER[0], DESK_CENTER[1], -DESK_SIZE[2] / 2))


def build_cubes(sim_utils, RigidObject, RigidObjectCfg) -> list:
    objects = []
    for i, (x, y, color) in enumerate(CUBES):
        cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Cube{i}",
            spawn=sim_utils.CuboidCfg(
                size=(0.045, 0.045, 0.045),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                mass_props=sim_utils.MassCfg(mass=0.05),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.0235)),
        )
        objects.append(RigidObject(cfg))
    return objects


def build_simready(sim_utils, RigidObject, RigidObjectCfg, root: Path, props=SIMREADY_DESK) -> list:
    """Spawn SimReady props, then fix up physics: realistic mass, convex hull (or convex
    decomposition for hollow props) colliders, gentle depenetration."""
    from pxr import PhysxSchema, Usd, UsdPhysics

    objects = []
    for i, p in enumerate(props):
        yaw = np.radians(p.yaw)
        cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/P{i:02d}_{p.asset}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(root / p.usd_rel)),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(p.x, p.y, 0.002 + p.z), rot=(0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))
            ),
        )
        objects.append(RigidObject(cfg))

    stage = sim_utils.get_current_stage()
    for i, p in enumerate(props):
        top = stage.GetPrimAtPath(f"/World/Objects/P{i:02d}_{p.asset}")
        for q in list(Usd.PrimRange(top)):
            if q.IsInstance():
                q.SetInstanceable(False)  # needed to edit colliders inside instances
        approx = "convexDecomposition" if p.hollow else "convexHull"
        for q in Usd.PrimRange(top):
            if q.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr().Set(approx)
            if q.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.MassAPI.Apply(q).CreateMassAttr().Set(p.mass)
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(q)
                rb.CreateMaxDepenetrationVelocityAttr().Set(1.0)
                rb.CreateSolverPositionIterationCountAttr().Set(16)
    return objects
