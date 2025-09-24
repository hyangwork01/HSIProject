from isaaclab.app import AppLauncher

headless = False
app_launcher = AppLauncher({"headless": headless})
simulation_app = app_launcher.app
from omegaconf import OmegaConf

import torch

from core.simulator.hoi_isaaclab.config import IsaacLabSimulatorConfig, IsaacLabSimParams
from core.simulator.hoi_isaaclab.simulator import IsaacLabSimulator
from core.simulator.base_simulator.config import (
    RobotConfig,
    RobotAssetConfig,
    InitState,
    ControlConfig,
    ControlType,
)
from core.envs.base_env.env_utils.terrains.flat_terrain import FlatTerrain
from core.envs.base_env.env_utils.terrains.terrain_config import TerrainConfig
from core.scenelib.scenelib import SceneLib
import math
# Create robot asset configuration
robot_asset_config = RobotAssetConfig(
    robot_type="h1",
    usd_asset_file_name="usd/h1.usd",
    self_collisions=False,
    collapse_fixed_joints=False,
)

# Create robot configuration
robot_config = RobotConfig(
    body_names=['pelvis', 'head', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 'left_foot_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 'right_foot_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_arm_end_effector', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_arm_end_effector'],
    dof_names=['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint', 'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'],
    dof_body_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    joint_axis=['z', 'x', 'y', 'y', 'y', 'z', 'x', 'y', 'y', 'y', 'z', 'y', 'x', 'z', 'y', 'y', 'x', 'z', 'y'],
    dof_obs_size=114,  # 19 joints * 6 (pos, vel, etc.)
    number_of_actions=19,
    self_obs_max_coords_size=373,
    left_foot_name="left_foot_link",
    right_foot_name="right_foot_link",
    head_body_name="head",
    key_bodies=[ "left_foot_link", "right_foot_link", "left_arm_end_effector",  "right_arm_end_effector" ],
    non_termination_contact_bodies=[ "left_foot_link", "left_ankle_link", "right_foot_link", "right_ankle_link" ],
    dof_effort_limits=[200., 200., 200., 300., 40., 200., 200., 200., 300., 40., 200., 40., 40., 18., 18., 40., 40., 18., 18.],
    dof_vel_limits=[23., 23., 23., 14., 9., 23., 23., 23., 14., 9., 23., 9., 9., 20., 20., 9., 9., 20., 20.],
    dof_armatures=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    dof_joint_frictions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    asset=robot_asset_config,
    init_state=InitState(
        pos=[0.0, 0.0, 1.0],
        default_joint_angles={
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.4,
            "left_knee_joint": 0.8,
            "left_ankle_joint": -0.4,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.4,
            "right_knee_joint": 0.8,
            "right_ankle_joint": -0.4,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
        },
    ),
    control=ControlConfig(
        control_type=ControlType.PROPORTIONAL,
        action_scale=1.0,
        clamp_actions=100.0,
        stiffness={
            'hip_yaw': 200,
            'hip_roll': 200,
            'hip_pitch': 200,
            'knee': 300,
            'ankle': 40,
            'torso': 300,
            'shoulder': 100,
            'elbow': 100,
        },
        damping={
            'hip_yaw': 5,
            'hip_roll': 5,
            'hip_pitch': 5,
            'knee': 6,
            'ankle': 2,
            'torso': 6,
            'shoulder': 2,
            'elbow': 2,
        },
    )
)

# Create simulator configuration
simulator_config = IsaacLabSimulatorConfig(
    sim=IsaacLabSimParams(
        fps=200,
        decimation=4,
    ),
    headless=headless,  # Set to True for headless mode
    robot=robot_config,
    num_envs=4,  # Number of parallel environments
    experiment_name="scene_chair_example",
    w_last=False,  # IsaacLab uses wxyz quaternions
)

device = torch.device("cuda")

# Create a flat terrain using the default config
terrain_config = TerrainConfig()
terrain = FlatTerrain(config=terrain_config, num_envs=simulator_config.num_envs, device=device)


scene_cfg_path = "core/config/scene/object_bed.yaml"
scene_cfg = OmegaConf.load(scene_cfg_path)
# Create SceneLib instance
scene_lib = SceneLib(scene_cfg,num_envs=simulator_config.num_envs, device=device)

# Create scenes
# scene_lib.create_task_scenes("sitchair", terrain,objects_path="/home/luohy/MyRepository/MyDataSets/Data/Objects")
# scene_lib.create_task_scenes("home", terrain,objects_path="/home/luohy/MyRepository/MyDataSets/Data/Objects")
# scene_lib.create_task_scenes("sitbed", terrain,objects_path="/home/luohy/MyRepository/MyDataSets/Data/Objects")
# scene_lib.create_task_scenes("muti_chair", terrain,objects_path="/home/luohy/MyRepository/MyDataSets/Data/Objects")
scene_lib.create_scenes( terrain)


# Create and initialize the simulator
simulator = IsaacLabSimulator(config=simulator_config, terrain=terrain, scene_lib=scene_lib, visualization_markers=None, device=device, simulation_app=simulation_app)
simulator.on_environment_ready()

# Get the scene positions. This indicates the center of the scene.
scene_positions = torch.stack(simulator.get_scene_positions())

# Get robot default state
default_state = simulator.get_default_state()
# Set the robot to a new random position above the ground
root_pos = torch.zeros(simulator_config.num_envs, 3, device=device)
# Add small offset so we don't spawn ontop of the scene
root_pos[:, :2] = scene_positions[:, :2] + 0.5
root_pos[:, 2] = 1.0
default_state.root_pos[:] = root_pos

# Reset the robots
simulator.reset_envs(default_state, env_ids=torch.arange(simulator_config.num_envs, device=device))

# Run the simulation loop
try:
    while True:
        """
            Camera controls in IsaacLab and IsaacGym:
            1. L - start/stop recording. Once stopped it will save the video.
            2. ; - cancel recording and delete the video.
            3. O - toggle camera target. This will cycle through the available camera targets, such as humanoids and objects in the scene.
            4. Q - close the simulator.
        """

        for i in range(0, simulator_config.sim.fps):
            actions = torch.randn(simulator_config.num_envs, simulator_config.robot.number_of_actions, device=device)
            simulator.step(actions)
        simulator.reset_envs(default_state, env_ids=torch.arange(simulator_config.num_envs, device=device))
        simulator._initial_scene_pos
        env_ids=torch.arange(simulator_config.num_envs, device=device)
        initial_scene_pos = simulator._initial_scene_pos
        E, O, _ = initial_scene_pos.shape            
        sample_len = simulator.terrain.env_length/2
        sample_width = simulator.terrain.env_width/2
        uv = torch.rand(E, O, 2, device=device) * 2.0 - 1.0  # 形状 [E, O, 2]
        # 广播乘以各自范围
        scale = torch.tensor([sample_len, sample_width], device=device).view(1, 1, 2)
        xy = uv * scale  # 形状 [E, O, 2]
        initial_scene_pos[..., 0:2] = xy

        scene_position = torch.stack(simulator.get_scene_positions()).unsqueeze(1)
        initial_scene_pos[..., :3] += scene_position[env_ids,:,:3]


        # 假设 initial[...,3:7] 存的是原四元数 [w, x, y, z]
        orig_q = initial_scene_pos[..., 3:7]   # [E, O, 4]
        # 1. 生成绕 Z 轴的随机四元数 q_s
        theta = torch.rand(E, O, device=device) * 2 * math.pi
        qs = torch.stack([
            torch.cos(theta * 0.5),
            torch.zeros_like(theta),
            torch.zeros_like(theta),
            torch.sin(theta * 0.5),
        ], dim=-1)  # [E, O, 4]

        # 2. 定义批量四元数乘法（Hamilton积），这里用左乘 qs * orig_q
        def quat_mul(q, r):
            # q, r 都是 [..., 4]，按 (w, x, y, z)
            w1, x1, y1, z1 = q.unbind(-1)
            w2, x2, y2, z2 = r.unbind(-1)
            return torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ], dim=-1)

        # 3. 计算新四元数
        new_q = quat_mul(qs, orig_q)  # [E, O, 4]

        # 4. 写回 initial_scene_pos
        initial_scene_pos[..., 3:7] = new_q

        simulator.reset_objects(initial_scene_pos,env_ids=torch.arange(simulator_config.num_envs, device=device))



except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    simulator.close()
