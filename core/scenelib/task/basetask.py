from abc import ABC, abstractmethod
from typing import List
from core.utils.scene_lib import Scene
from pathlib import Path
from core.envs.base_env.env_utils.terrains.terrain import Terrain
import random

class BaseTaskScene(ABC):
    def get_usd_path(self,objects_path: str) -> list[str]:
        """
        获取usd文件路径列表
        
        """
        root = Path(objects_path)
        usd_paths: list[str] = []
        if not root.is_dir():
            raise ValueError(f"Objects path '{objects_path}' 不存在或不是目录。")
        for usd_file in root.rglob("instance.usd"):
            usd_paths.append(str(usd_file.resolve()))
        return usd_paths
    
    def sort_usd_path(
        self,
        usds_list: List[str],
        assign_method: str = "sequential"
    ) -> List[str]:
        """
        对 usds_list 列表按 assign_method 排序：
        - "sequential": 按倒数第二级目录（001,002,...）升序排列
        - "random": 随机打乱顺序
        返回排序/打乱后的新列表，不修改原列表。
        """
        sorted_paths = usds_list.copy()
        if assign_method == "sequential":
            sorted_paths.sort(key=lambda p: Path(p).parts[-2])
        elif assign_method == "random":
            random.shuffle(sorted_paths)
        else:
            raise ValueError(
                f"Unknown assign_method '{assign_method}'. "
                "Use 'sequential' or 'random'."
            )
        return sorted_paths

    @abstractmethod
    def __init__(self,terrain: Terrain,num_envs: int =1, objects_path: str = None, assign_method: str = "sequential", replicate_method: str = "random",create_type:str ="single"):
        """
        
        """
        pass

    @abstractmethod
    def create_scenes(self) -> List[Scene]:
        """
        
        """
        pass

    @abstractmethod
    def set_scene(self) -> Scene:
        """

        """
        pass
    # @abstractmethod
    # def create_single_scenes(self) -> List[Scene]:
    #     pass

    # @abstractmethod
    # def create_multi_scenes(self) -> List[Scene]:
    #     pass

    # @abstractmethod
    # def set_scene(self, chair_path: str) -> Scene:
    #     pass    
