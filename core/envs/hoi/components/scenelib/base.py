
from abc import ABC, abstractmethod  # 抽象基类
# from typing import Tuple, Protocol, runtime_checkable
import torch

class Base(ABC):

    def __init__(self,cfg,num_envs: int,device: str = "cpu"):
        self._rigid_objects_cfg_list = dict()
        self._articulations_cfg_list = dict()
        self._sensors_cfg_list = dict()
        self._extras_cfg_list = dict()
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.num_objects_per_scene = None
        self.num_articulations_per_scene = None
        self.num_rigidobjects_per_scene = None

    # def get_cfg(self):
