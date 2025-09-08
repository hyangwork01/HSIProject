import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from easydict import EasyDict
from core.envs.base_env.env_utils.terrains.terrain import Terrain
from core.scenelib.scenelib import SceneLib
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from core.simulator.isaaclab.utils.scene import SceneCfg
from core.simulator.base_simulator.simulator import Simulator
from core.simulator.base_simulator.config import (
    MarkerState,
    ControlType,
    VisualizationMarker,
    SimBodyOrdering,
    SimulatorConfig,
)
from core.simulator.base_simulator.robot_state import RobotState


class IsaacLabSimulator(Simulator):
    # =====================================================
    # Group 1: Initialization & Configuration
    # =====================================================
    def __init__(
        self,
        config: SimulatorConfig,
        terrain: Terrain,
        device: torch.device,
        simulation_app: Any,
        scene_lib: Optional[SceneLib] = None,
        visualization_markers: Optional[Dict[str, VisualizationMarker]] = None,
    ) -> None:
        """
        Initialize the IsaacLabSimulator.

        Parameters:
            config (SimulatorConfig): The configuration dictionary.
            terrain (Terrain): Terrain data for simulation.
            device (torch.device): Device to use for computation.
            simulation_app (Any): The simulation application instance.
            scene_lib (Optional[SceneLib], optional): The scene library containing scene and object data.
            visualization_markers (Optional[Dict[str, VisualizationMarker]], optional): Configuration for visualization markers.
        """
        super().__init__(
            config=config,
            scene_lib=scene_lib,
            terrain=terrain,
            visualization_markers=visualization_markers,
            device=device,
        )

        sim_cfg = sim_utils.SimulationCfg(
            device=str(device),
            dt=1.0 / self.config.sim.fps,
            render_interval=self.config.sim.decimation,
            physx=PhysxCfg(
                solver_type=self.config.sim.physx.solver_type,
                max_position_iteration_count=self.config.sim.physx.num_position_iterations,
                max_velocity_iteration_count=self.config.sim.physx.num_velocity_iterations,
                bounce_threshold_velocity=self.config.sim.physx.bounce_threshold_velocity,
                gpu_max_rigid_contact_count=self.config.sim.physx.gpu_max_rigid_contact_count,
                gpu_found_lost_pairs_capacity=self.config.sim.physx.gpu_found_lost_pairs_capacity,
                gpu_found_lost_aggregate_pairs_capacity=self.config.sim.physx.gpu_found_lost_aggregate_pairs_capacity,
            ),
        )
        self._simulation_app = simulation_app
        self._sim = SimulationContext(sim_cfg)
        self._sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

        scene_cfg = self._get_scene_cfg()

        self._scene = InteractiveScene(scene_cfg)
        if not self.headless:
            self._setup_keyboard()
        print("[INFO]: Setup complete...")

        self._robot = self._scene["robot"]
        self._contact_sensor = self._scene["contact_sensor"]
        self._object = []
        if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
            for obj_idx in range(self.scene_lib.num_objects_per_scene):
                self._object.append(self._scene[f"object_{obj_idx}"])
        if visualization_markers:
            self._build_markers(visualization_markers)
        self._sim.reset()

    def reset_envs(self, new_states, env_ids):
 
        super().reset_envs(new_states, env_ids)
    

    # def reset_objects(self,new_states = None, env_ids = None):
    #     if env_ids is None:
    #         # Update initial object positions
    #         if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
    #             objects_start_pos = torch.zeros(
    #                 (self.num_envs, 13), device=self.device, dtype=torch.float
    #             )
    #             for obj_idx, object in enumerate(self._object):
    #                 objects_start_pos[:, :7] = self._initial_scene_pos[:, obj_idx, :]
    #                 object.write_root_state_to_sim(objects_start_pos)
    #     # Update initial object positions
    #     if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
    #         objects_start_pos = torch.zeros(
    #             (len(env_ids), 13), device=self.device, dtype=torch.float
    #         )
    #         for obj_idx, object in enumerate(self._object):
    #             objects_start_pos[:, :7] = self._initial_scene_pos[env_ids, obj_idx, :]
    #             object.write_root_state_to_sim(objects_start_pos, env_ids)

    def reset_objects(self, new_states=None, env_ids=None):
        # 1. 如果没有传入 env_ids，就默认对所有 env 进行操作
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            # 保证 env_ids 是 long 类型的索引张量
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # 2. 如果没有传入 new_states，就使用 _initial_scene_pos 中保存的初始位置
        if new_states is None:
            # _initial_scene_pos: [num_envs, num_objects, state_dim]
            new_states = self._initial_scene_pos[env_ids,:,:]


        # 2.5 长度检查：new_states 和 env_ids 的 batch 大小必须一致
        if new_states.shape[0] != env_ids.shape[0]:
            raise ValueError(
                f"Batch size mismatch: new_states has {new_states.shape[0]} entries, "
                f"but env_ids has {env_ids.shape[0]} entries"
            )

        # 3. 准备写入模拟器的状态张量
        #    假设 write_root_state_to_sim 需要形状 (B, 13) 的输入，
        #    其中前 7 个元素是位置/旋转数据，剩余可以保留为 0。
        B = env_ids.shape[0]
        objects_start_pos = torch.zeros((B, 13), device=self.device, dtype=torch.float)

        # 4. 按照每个 object 依次写入对应 env 的 new_states
        for obj_idx, obj in enumerate(self._object):
            # new_states[:, obj_idx, :] 的形状是 (B, 7)
            objects_start_pos[:, :7] = new_states[:, obj_idx, :]
            # 把这些状态写入 simulator（假设第二个参数接受 env_ids）
            obj.write_root_state_to_sim(objects_start_pos, env_ids)

    def _get_scene_cfg(self) -> SceneCfg:
        """
        Construct and return the scene configuration from the current config, scene library, and terrain.

        Returns:
            SceneCfg: The constructed scene configuration.
        """
        scene_cfgs = None
        if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
            scene_cfgs, self._initial_scene_pos,self._object_types = self._preprocess_object_playground()

            scene_cfg = SceneCfg(
                config=self.config,
                robot_config=self.robot_config,
                num_envs=self.config.num_envs,
                env_spacing=2.0,
                scene_cfgs=scene_cfgs,
                terrain=self.terrain,
                objects_type = self._object_types,
            )
        else:
            scene_cfg = SceneCfg(
                config=self.config,
                robot_config=self.robot_config,
                num_envs=self.config.num_envs,
                env_spacing=2.0,
                scene_cfgs=scene_cfgs,
                terrain=self.terrain,
            )            
        return scene_cfg

    def _preprocess_object_playground(self) -> Tuple[List[Any], torch.Tensor, List[Any]]:
        """
        Process and build the object playground from the scene library.

        Returns:
            Tuple[List[Any], torch.Tensor]: A tuple containing the object configurations and the initial object positions.
        """
        print("=========== Building object playground")

        objects_cfgs = []
        for _ in range(self.scene_lib.num_objects_per_scene):
            objects_cfgs.append([])

        objects_type = []
        for _ in range(self.scene_lib.num_objects_per_scene):
            objects_type.append([])

        initial_obj_pos = torch.zeros(
            (self.num_envs, self.scene_lib.num_objects_per_scene, 7),
            device=self.device,
            dtype=torch.float,
        )

        

        for scene_idx, scene in enumerate(self.scene_lib.scenes):
            scene_offset = self.scene_lib.scene_offsets[scene_idx]

            height_at_scene_origin = self.terrain.get_ground_heights(
                torch.tensor(
                    [[scene_offset[0], scene_offset[1]]],
                    device=self.device,
                    dtype=torch.float,
                )
            ).item()
            self._scene_position.append(
                torch.tensor(
                    [scene_offset[0], scene_offset[1], height_at_scene_origin],
                    device=self.device,
                    dtype=torch.float,
                )
            )
            self._object_dims.append([])

            for obj_idx, obj in enumerate(scene.objects):
                # TODO:根据ID去对应不同的Object
                # Get the spawn info for this object which contains the correct ID

                file_extension = obj.object_path.split("/")[-1].split(
                    "."
                )[-1]

                assert file_extension in [
                    "usd",
                    "usda",
                ], f"Home asset [{obj.object_path}] must be a USD file"
                

                objects_type[obj_idx].append(obj.object_type)


                # Calculate the global position of the object
                global_object_position = torch.tensor(
                    [
                        scene_offset[0] + obj.translation[0],
                        scene_offset[1] + obj.translation[1],
                        0 + obj.translation[2],
                    ],
                    device=self.device,
                    dtype=torch.float,
                )

                initial_obj_pos[scene_idx, obj_idx, :3] = global_object_position
                initial_obj_pos[scene_idx, obj_idx, 3:] = torch.tensor(
                    [
                        obj.rotation[3],
                        obj.rotation[0],
                        obj.rotation[1],
                        obj.rotation[2],


                    ],
                    device=self.device,
                    dtype=torch.float,
                )  # Convert xyzw to wxyz

                main_dir_path = (
                    f"{os.path.dirname(os.path.abspath(__file__))}/../../../"
                )
                asset_path = Path(
                    os.path.join(main_dir_path, obj.object_path)
                ).resolve()

                if obj.object_type == "articulation":
                    rigid_props = sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=obj.options.rigid_body_enabled,
                        kinematic_enabled=obj.options.kinematic_enabled,
                        angular_damping=obj.options.angular_damping,
                        linear_damping=obj.options.linear_damping,
                        max_linear_velocity=obj.options.max_linear_velocity,
                        max_angular_velocity=obj.options.max_angular_velocity,
                        retain_accelerations=obj.options.retain_accelerations,
                    )
                    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                        articulation_enabled = obj.options.articulation_enabled,
                        enabled_self_collisions = obj.options.enabled_self_collisions,
                        fix_root_link = obj.options.fix_root_link,
                    )
                    mass_props = sim_utils.MassPropertiesCfg(
                        density=obj.options.density,
                    )
                    if obj.options.visual_material is False:
                        spawn_cfg = sim_utils.UsdFileCfg(
                            usd_path=str(asset_path),
                            scale=obj.scale,
                            rigid_props=rigid_props,
                            articulation_props=articulation_props,
                            mass_props=mass_props,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.2, 0.7, 0.3), metallic=0.2
                            ),
                        )
                    else:
                        spawn_cfg = sim_utils.UsdFileCfg(
                            usd_path=str(asset_path),
                            scale=obj.scale,
                            rigid_props=rigid_props,
                            articulation_props=articulation_props,
                            mass_props=mass_props,
                        )
                elif obj.object_type == "rigid":
                    rigid_props = sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=obj.options.rigid_body_enabled,
                        kinematic_enabled=obj.options.kinematic_enabled,
                        angular_damping=obj.options.angular_damping,
                        linear_damping=obj.options.linear_damping,
                        max_linear_velocity=obj.options.max_linear_velocity,
                        max_angular_velocity=obj.options.max_angular_velocity,
                        retain_accelerations=obj.options.retain_accelerations,
                    )
                    collision_props = sim_utils.CollisionPropertiesCfg(
                        collision_enabled = obj.options.collision_enabled,
                        contact_offset=0.002,
                        rest_offset=0.0,
                    )

                    mass_props = sim_utils.MassPropertiesCfg(
                        density=obj.options.density,
                    )
                    if obj.options.visual_material is False:
                        spawn_cfg = sim_utils.UsdFileCfg(
                            usd_path=str(asset_path),
                            scale=obj.scale,
                            rigid_props=rigid_props,
                            collision_props=collision_props,
                            mass_props=mass_props,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.2, 0.7, 0.3), metallic=0.2
                            ),
                        )
                    else:
                        spawn_cfg = sim_utils.UsdFileCfg(
                            usd_path=str(asset_path),
                            scale=obj.scale,
                            rigid_props=rigid_props,
                            collision_props=collision_props,
                            mass_props=mass_props,
                        )   
                else:
                    raise ValueError(f"Invalid object type: {obj.object_type}")
                
                objects_cfgs[obj_idx].append(spawn_cfg)
                object_dims = torch.tensor(
                obj.object_dims, device=self.device, dtype=torch.float
                )
                self._object_dims[-1].append(object_dims)

            self._object_dims[-1] = torch.stack(self._object_dims[-1]).reshape(
                self._num_objects_per_scene, -1
            )
        for idx in range(self.scene_lib.num_objects_per_scene):
            if len(set(objects_type[idx])) != 1:
                raise ValueError(
                    f"Object {idx} has more than one object type. This is not supported."
                )
            objects_type[idx] = objects_type[idx][0]

        
        return objects_cfgs, initial_obj_pos ,objects_type

    def _setup_keyboard(self) -> None:
        """
        Set up keyboard callbacks for control using the Se2Keyboard interface.
        """
        from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard

        self.keyboard_interface = Se2Keyboard()
        self.keyboard_interface.add_callback("R", self._requested_reset)
        self.keyboard_interface.add_callback("U", self._update_inference_parameters)
        self.keyboard_interface.add_callback("L", self._toggle_video_record)
        self.keyboard_interface.add_callback(";", self._cancel_video_record)
        self.keyboard_interface.add_callback("Q", self.close)
        self.keyboard_interface.add_callback("O", self._toggle_camera_target)
        self.keyboard_interface.add_callback("J", self._push_robot)

    # =====================================================
    # Group 2: Environment Setup & Configuration
    # =====================================================
    def on_environment_ready(self) -> None:
        """
        Configure initial environment settings when the simulation is ready.
        This includes setting up joint limits and initializing state tensors.
        """
        self._isaaclab_default_state = RobotState(
            root_pos=self._robot.data.root_pos_w.clone(),
            root_rot=self._robot.data.root_quat_w.clone(),
            root_vel=torch.zeros(
                (len(self._robot.data.root_pos_w), 3), device=self.device
            ),
            root_ang_vel=torch.zeros(
                (len(self._robot.data.root_pos_w), 3), device=self.device
            ),
            dof_pos=self._robot.data.joint_pos.clone(),
            dof_vel=self._robot.data.joint_vel.clone(),
            rigid_body_pos=self._robot.data.body_pos_w.clone(),
            rigid_body_rot=self._robot.data.body_quat_w.clone(),
            rigid_body_vel=self._robot.data.body_lin_vel_w.clone(),
            rigid_body_ang_vel=self._robot.data.body_ang_vel_w.clone(),
        )

        dof_limits = self._robot.data.joint_limits.clone()
        self._dof_limits_lower_sim = dof_limits[0, :, 0].to(self.device)
        self._dof_limits_upper_sim = dof_limits[0, :, 1].to(self.device)

        super().on_environment_ready()

        # Update initial object positions
        if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
            objects_start_pos = torch.zeros(
                (self.num_envs, 13), device=self.device, dtype=torch.float
            )
            for obj_idx, object in enumerate(self._object):
                objects_start_pos[:, :7] = self._initial_scene_pos[:, obj_idx, :]
                object.write_root_state_to_sim(objects_start_pos)

    # =====================================================
    # Group 3: Simulation Steps & State Management
    # =====================================================
    def _physics_step(self) -> None:
        """
        Advance the simulation by stepping for a number of iterations equal to the decimation factor.
        Applies PD control or motor forces as required.
        """
        for idx in range(self.decimation):
            if self.control_type == ControlType.BUILT_IN_PD:
                self._apply_pd_control()
            else:
                self._apply_motor_forces()
            self._scene.write_data_to_sim()
            self._sim.step(render=False)
            if (idx + 1) % self.decimation == 0 and not self.headless:
                self._sim.render()
            self._scene.update(dt=self._sim.get_physics_dt())

    def _apply_pd_control(self) -> None:
        """
        Apply PD control by converting actions into PD targets and updating joint targets accordingly.
        """
        common_pd_tar = self._action_to_pd_targets(self._common_actions)
        isaaclab_pd_tar = common_pd_tar[:, self.data_conversion.dof_convert_to_sim]
        self._robot.set_joint_position_target(isaaclab_pd_tar, joint_ids=None)

    def _apply_motor_forces(self) -> None:
        """
        Apply motor forces to the robot.

        Raises:
            NotImplementedError: Not supported yet.
        """
        common_torques = self._compute_torques(self._common_actions)
        isaaclab_torques = common_torques[:, self.data_conversion.dof_convert_to_sim]
        self._robot.set_joint_effort_target(isaaclab_torques, joint_ids=None)

    def _set_simulator_env_state(
        self, new_states: RobotState, env_ids: Optional[torch.Tensor]
    ) -> None:
        """
        Apply the provided state to the simulation by writing root and joint states.

        Parameters:
            new_states (RobotState): The new simulation state.
            env_ids (Optional[torch.Tensor]): Specific environment IDs to update.
        """
        init_root_state = torch.cat(
            [
                new_states.root_pos,
                new_states.root_rot,
                new_states.root_vel,
                new_states.root_ang_vel,
            ],
            dim=-1,
        )
        self._robot.write_root_state_to_sim(init_root_state, env_ids)
        self._robot.write_joint_state_to_sim(
            new_states.dof_pos, new_states.dof_vel, None, env_ids
        )

    # =====================================================
    # Group 4: State Getters
    # =====================================================
    def _get_simulator_default_state(self) -> RobotState:
        """
        Retrieve the default simulation state based on the initialized values.

        Returns:
            RobotState: The default state containing positions, orientations, velocities, etc.
        """
        return self._isaaclab_default_state

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """
        Obtain the ordering of body, degree-of-freedom, and contact sensor names.

        Returns:
            SimBodyOrdering: An object containing the body names, DOF names, and contact sensor body names.
        """
        return SimBodyOrdering(
            body_names=self._robot.data.body_names,
            dof_names=self._robot.data.joint_names,
            contact_sensor_body_names=self._contact_sensor.body_names,
        )

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the state (positions, rotations, velocities) of all simulation bodies.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict state retrieval to specific environments if provided.

        Returns:
            RobotState: The state of the bodies.
        """
        isaacsim_bodies_positions = self._robot.data.body_pos_w.clone()
        isaacsim_bodies_rotations = self._robot.data.body_quat_w.clone()
        isaacsim_bodies_velocities = self._robot.data.body_lin_vel_w.clone()
        isaacsim_bodies_ang_velocities = self._robot.data.body_ang_vel_w.clone()

        isaacsim_bodies_positions = isaacsim_bodies_positions.view(
            self.num_envs, self._num_bodies, 3
        )
        isaacsim_bodies_rotations = isaacsim_bodies_rotations.view(
            self.num_envs, self._num_bodies, 4
        )
        isaacsim_bodies_velocities = isaacsim_bodies_velocities.view(
            self.num_envs, self._num_bodies, 3
        )
        isaacsim_bodies_ang_velocities = isaacsim_bodies_ang_velocities.view(
            self.num_envs, self._num_bodies, 3
        )
        if env_ids is not None:
            isaacsim_bodies_positions = isaacsim_bodies_positions[env_ids]
            isaacsim_bodies_rotations = isaacsim_bodies_rotations[env_ids]
            isaacsim_bodies_velocities = isaacsim_bodies_velocities[env_ids]
            isaacsim_bodies_ang_velocities = isaacsim_bodies_ang_velocities[env_ids]
        return RobotState(
            rigid_body_pos=isaacsim_bodies_positions,
            rigid_body_rot=isaacsim_bodies_rotations,
            rigid_body_vel=isaacsim_bodies_velocities,
            rigid_body_ang_vel=isaacsim_bodies_ang_velocities,
        )

    def _get_simulator_dof_forces(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve applied torque forces for the robot's degrees of freedom.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict query to specific environments if provided.

        Returns:
            torch.Tensor: The DOF forces.
        """
        isaacsim_dof_forces = self._robot.data.applied_torque.clone()
        if env_ids is not None:
            isaacsim_dof_forces = isaacsim_dof_forces[env_ids]
        return isaacsim_dof_forces

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the state (positions and velocities) of the robot's DOFs.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict state retrieval to specific environments if provided.

        Returns:
            RobotState: The DOF state.
        """
        isaacsim_dof_pos = self._robot.data.joint_pos.clone()
        isaacsim_dof_vel = self._robot.data.joint_vel.clone()
        if env_ids is not None:
            isaacsim_dof_pos = isaacsim_dof_pos[env_ids]
            isaacsim_dof_vel = isaacsim_dof_vel[env_ids]
        return RobotState(
            dof_pos=isaacsim_dof_pos,
            dof_vel=isaacsim_dof_vel,
        )

    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve the contact force buffer for simulation bodies.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            torch.Tensor: Tensor containing the contact forces.
        """
        if self._contact_sensor.data.force_matrix_w is not None:
            isaacsim_rb_contacts = (
                self._contact_sensor.data.force_matrix_w.clone().view(
                    self.num_envs, self._num_bodies, -1, 3
                )
            )
            isaacsim_rb_contacts = isaacsim_rb_contacts.sum(dim=2)
        else:
            isaacsim_rb_contacts = self._contact_sensor.data.net_forces_w.clone().view(
                self.num_envs, self._num_bodies, 3
            )
        if env_ids is not None:
            isaacsim_rb_contacts = isaacsim_rb_contacts[env_ids]
        return isaacsim_rb_contacts

    def _get_simulator_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve the contact buffer for simulation objects.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            torch.Tensor: The object contact buffer.
        """
        if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
            object_forces = []
            for obj_idx in range(self.scene_lib.num_objects_per_scene):
                object_forces.append(
                    self._object[obj_idx].data.net_contact_forces_w.clone()
                )
            if env_ids is not None:
                object_forces = object_forces[env_ids]
            return torch.stack(object_forces, dim=1)
        else:
            return_tensor = torch.zeros(
                self.num_envs, 1, 3, device=self.device, dtype=torch.float
            )
            if env_ids is not None:
                return_tensor = return_tensor[env_ids]
            return return_tensor

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the root state (position, rotation, velocity) of the robot.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            RobotState: The robot's root state.
        """
        isaacsim_root_pos = self._robot.data.root_pos_w.clone()
        isaacsim_root_rot = self._robot.data.root_quat_w.clone()
        isaacsim_root_vel = self._robot.data.root_lin_vel_w.clone()
        isaacsim_root_ang_vel = self._robot.data.root_ang_vel_w.clone()
        if env_ids is not None:
            isaacsim_root_pos = isaacsim_root_pos[env_ids]
            isaacsim_root_rot = isaacsim_root_rot[env_ids]
            isaacsim_root_vel = isaacsim_root_vel[env_ids]
            isaacsim_root_ang_vel = isaacsim_root_ang_vel[env_ids]
        return RobotState(
            root_pos=isaacsim_root_pos,
            root_rot=isaacsim_root_rot,
            root_vel=isaacsim_root_vel,
            root_ang_vel=isaacsim_root_ang_vel,
        )

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the combined root state for all simulation objects.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            RobotState: The objects' root state.
        """
        isaacsim_root_pos = []
        isaacsim_root_rot = []
        isaacsim_root_vel = []
        isaacsim_root_ang_vel = []
        for obj_idx in range(self.scene_lib.num_objects_per_scene):
            isaacsim_root_pos.append(self._object[obj_idx].data.root_pos_w.clone())
            isaacsim_root_rot.append(self._object[obj_idx].data.root_quat_w.clone())
            isaacsim_root_vel.append(self._object[obj_idx].data.root_lin_vel_w.clone())
            isaacsim_root_ang_vel.append(
                self._object[obj_idx].data.root_ang_vel_w.clone()
            )
        isaacsim_root_pos = torch.stack(isaacsim_root_pos, dim=1)
        isaacsim_root_rot = torch.stack(isaacsim_root_rot, dim=1)
        isaacsim_root_vel = torch.stack(isaacsim_root_vel, dim=1)
        isaacsim_root_ang_vel = torch.stack(isaacsim_root_ang_vel, dim=1)
        if env_ids is not None:
            isaacsim_root_pos = isaacsim_root_pos[env_ids]
            isaacsim_root_rot = isaacsim_root_rot[env_ids]
            isaacsim_root_vel = isaacsim_root_vel[env_ids]
            isaacsim_root_ang_vel = isaacsim_root_ang_vel[env_ids]
        return RobotState(
            root_pos=isaacsim_root_pos,
            root_rot=isaacsim_root_rot,
            root_vel=isaacsim_root_vel,
            root_ang_vel=isaacsim_root_ang_vel,
        )

    def get_num_actors_per_env(self) -> int:
        """
        Compute and return the number of actor instances per environment.

        Returns:
            int: Number of actors per environment.
        """
        root_pos = self._robot.data.root_pos_w
        return root_pos.shape[0] // self.num_envs

    # =====================================================
    # Group 5: Control & Computation Methods
    # =====================================================

    def _push_robot(self):
        vel_w = self._robot.data.root_vel_w
        self._robot.write_root_velocity_to_sim(
            vel_w + torch.ones_like(vel_w),
            env_ids=torch.arange(self.num_envs, device=self.device),
        )

    # =====================================================
    # Group 6: Rendering & Visualization
    # =====================================================
    def render(self) -> None:
        """
        Render the simulation view. Initializes or updates the camera if the simulator is not in headless mode.
        """
        if not self.headless:
            if not hasattr(self, "_perspective_view"):
                from core.simulator.isaaclab.utils.perspective_viewer import (
                    PerspectiveViewer,
                )

                self._perspective_view = PerspectiveViewer()
                self._init_camera()
            else:
                self._update_camera()
        super().render()

    def _init_camera(self) -> None:
        """
        Initialize the camera view based on the current simulation root state.
        """
        self._cam_prev_char_pos = (
            self._get_simulator_root_state(0).root_pos.cpu().numpy()
        )
        pos = self._cam_prev_char_pos + np.array([0, -5, 1])
        self._perspective_view.set_camera_view(
            pos, self._cam_prev_char_pos + np.array([0, 0, 0.2])
        )

    def _update_camera(self) -> None:
        """
        Update the camera view based on the target's position and current camera movement.
        """
        if self._camera_target["element"] == 0:
            char_root_pos = (
                self._get_simulator_root_state(self._camera_target["env"])
                .root_pos.cpu()
                .numpy()
            )
            height_offset = 0.2
        else:
            in_scene_object_id = self._camera_target["element"] - 1
            char_root_pos = (
                self._get_simulator_object_root_state(self._camera_target["env"])
                .root_pos[in_scene_object_id]
                .cpu()
                .numpy()
            )
            height_offset = 0

        cam_pos = np.array(self._perspective_view.get_camera_state())
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = np.array(
            [char_root_pos[0], char_root_pos[1], char_root_pos[2] + height_offset]
        )
        new_cam_pos = np.array(
            [
                char_root_pos[0] + cam_delta[0],
                char_root_pos[1] + cam_delta[1],
                char_root_pos[2] + cam_delta[2],
            ]
        )
        self._perspective_view.set_camera_view(new_cam_pos, new_cam_target)
        self._cam_prev_char_pos[:] = char_root_pos

    def _write_viewport_to_file(self, file_name: str) -> None:
        """
        Capture the current viewport and save it to the specified file.

        Parameters:
            file_name (str): The filename for the saved image.
        """
        from omni.kit.viewport.utility import (
            get_active_viewport,
            capture_viewport_to_file,
        )

        vp_api = get_active_viewport()
        capture_viewport_to_file(vp_api, file_name)

    def close(self) -> None:
        """
        Close the simulation application and perform cleanup.
        """
        self._simulation_app.close()

    def _build_markers(
        self, visualization_markers: Dict[str, VisualizationMarker]
    ) -> None:
        """Build and configure visualization markers.

        Args:
            visualization_markers (Dict[str, VisualizationMarker]): Dictionary mapping marker names to their configurations
        """
        self._visualization_markers = {}
        if visualization_markers is None:
            return

        for marker_name, markers_cfg in visualization_markers.items():
            if markers_cfg.type == "sphere":
                marker_obj_cfg = VisualizationMarkersCfg(
                    prim_path=f"/Visuals/{marker_name}",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=1,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(
                                    markers_cfg.color[0],
                                    markers_cfg.color[1],
                                    markers_cfg.color[2],
                                )
                            ),
                        ),
                    },
                )
            elif markers_cfg.type == "arrow":
                marker_obj_cfg = VisualizationMarkersCfg(
                    prim_path=f"/Visuals/{marker_name}",
                    markers={
                        "arrow_x": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                            scale=(1.0, 1.0, 1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(
                                    markers_cfg.color[0],
                                    markers_cfg.color[1],
                                    markers_cfg.color[2],
                                ),
                                opacity=0.5,
                            ),
                        ),
                    },
                )
            else:
                raise ValueError(f"Marker type {markers_cfg.type} not supported")

            marker_scale = []
            for i, marker in enumerate(markers_cfg.markers):
                if markers_cfg.type == "sphere":
                    if marker.size == "tiny":
                        scale = 0.007
                    elif marker.size == "small":
                        scale = 0.01
                    else:
                        scale = 0.05
                    marker_scale.append([scale, scale, scale])
                elif markers_cfg.type == "arrow":
                    if marker.size == "small":
                        scale = 0.1
                    else:
                        scale = 0.5
                    marker_scale.append([scale, 0.2 * scale, 0.2 * scale])

            self._visualization_markers[marker_name] = EasyDict(
                {
                    "marker": VisualizationMarkers(marker_obj_cfg),
                    "scale": torch.tensor(marker_scale, device=self.device).repeat(
                        self.num_envs, 1
                    ),
                }
            )

    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """Update the visualization markers with new state information.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary mapping marker names to their state (translation and orientation)
        """
        if markers_state is None:
            return

        for marker_name, markers_state_item in markers_state.items():
            marker_dict = self._visualization_markers[marker_name]
            marker_dict.marker.visualize(
                translations=markers_state_item.translation.view(-1, 3),
                orientations=markers_state_item.orientation.view(-1, 4),
                scales=marker_dict.scale,
            )
