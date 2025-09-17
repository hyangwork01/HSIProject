
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
from dataclasses import MISSING

@configclass
class MyRigidObjectCfg(RigidObjectCfg):
    prim_path = "/World/envs/env_.*/RigidObject"
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled = True, 
        kinematic_enabled = False, # False则不固定物体，True则固定物体
        angular_damping = None,
        linear_damping = None,
        max_linear_velocity = None,
        max_angular_velocity = None,
        retain_accelerations = None,
    )
    collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled = True,
        contact_offset=0.002,
        rest_offset=0.0,
    )
    mass_props = sim_utils.MassPropertiesCfg(
        density=None,
    )
    # visual_material=sim_utils.PreviewSurfaceCfg(
    #         diffuse_color=(0.2, 0.7, 0.3), metallic=0.2
    # )
    spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=MISSING,
        scale=(1.0, 1.0, 1.0),
        rigid_props=rigid_props,
        collision_props=collision_props,
        mass_props=mass_props,
    )
    init_state=RigidObjectCfg.InitialStateCfg()