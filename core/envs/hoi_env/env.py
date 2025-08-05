from typing import Dict, Optional

import torch
from torch import Tensor
from isaac_utils import rotations, torch_utils
from core.envs.mimic.mimic_utils import (
    dof_to_local,
    exp_tracking_reward,
)
from core.envs.base_env.env_utils.humanoid_utils import quat_diff_norm
from core.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState

from core.envs.base_env.env import BaseEnv
from core.envs.mimic.components.mimic_obs import MimicObs
from core.envs.mimic.components.mimic_motion_manager import MimicMotionManager
from core.envs.mimic.components.masked_mimic_obs import MaskedMimicObs
from core.envs.hoi_env.components.object_obs import ObjectObs

class HoiEnv(BaseEnv):
    def __init__(self, config,device:torch.device,*args, **kwargs):
        super().__init__(config,device,*args, **kwargs)
        self.obj_obs_cb = ObjectObs(self.config,self)


    def create_visualization_markers(self):
        if self.config.headless:
            return {}

        visualization_markers = super().create_visualization_markers()

        object_markers = []
        object_markers_cfg = VisualizationMarker(
            type="sphere",
            color=(1.0, 0, 0, 0.0),
            markers=object_markers
        )
        visualization_markers["object_markers"] = object_markers_cfg

        return visualization_markers

    def get_markers_state(self):
        if self.config.headless:
            return {}

        markers_state = super().get_markers_state()
        
        ## TODO: Add object markers state

        return markers_state

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if len(env_ids) > 0:
            self.reset_hoi_task(env_ids)
        return super().reset(env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        self.check_update_task()

    def check_update_task(self):
        reset_task_mask = self.progress_buf >= self._heading_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self.reset_hoi_task(rest_env_ids)

    def reset_hoi_task(self, env_ids):
        n = len(env_ids)
        # TODO: Reset Objects to new positions
        

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        super().compute_observations(env_ids)
        self.obj_obs_cb.compute_observations(env_ids)

    def get_obs(self):
        obs = super().get_obs()
        object_obs = self.obj_obs_cb.get_obs()
        obs.update(object_obs)
        return obs

    def compute_reward(self):
        rew = super().compute_reward()
        
        ## TODO: Compute reward based on object interactions, task completion, etc.
        
        return rew