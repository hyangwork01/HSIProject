from __future__ import annotations
from abc import ABC, abstractmethod  # 抽象基类
from typing import Tuple, Protocol, runtime_checkable
import torch

class Terrain(ABC):
    """所有 Terrain 的统一接口。外部代码只依赖这些方法/属性。还包含一些内部处理方法"""

    def __init__(self, config, num_envs: int, device) -> None:
        self.config = config
        self.device = device
        self.num_envs = num_envs

    
    
    def sample_valid_locations(self, envs_ids) -> torch.Tensor:
        pass

    def _init_height_points(self, num_envs) -> Tuple[int, torch.Tensor]:
        pass

    def get_ground_heights(self, locations: torch.Tensor) -> torch.Tensor:
        pass

    def get_height_maps(self, root_states, env_ids=None, return_all_dims=False):
        pass

    

    # def generate_terrain_plot(self):
    #     pass

    