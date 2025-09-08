import numpy as np
import math
import torch
from scipy import ndimage

from core.envs.base_env.env_utils.terrains.subterrain import SubTerrain
from core.envs.base_env.env_utils.terrains.subterrain_generator import (
    discrete_obstacles_subterrain,
    poles_subterrain,
    pyramid_sloped_subterrain,
    pyramid_stairs_subterrain,
    random_uniform_subterrain,
    stepping_stones_subterrain,
)
from core.envs.base_env.env_utils.terrains.terrain_utils import (
    convert_heightfield_to_trimesh, get_heights_jit, get_height_maps_jit
)
from core.envs.base_env.env_utils.terrains.terrain_config import TerrainConfig

import matplotlib.pyplot as plt
from core.envs.base_env.env_utils.terrains.terrain import Terrain
import math

class Terrain:

    def __init__(self, config: TerrainConfig, num_envs: int, device) -> None:
        config.load_terrain = False
        config.save_terrain = False
        self.terrain_type = "Flatten" # 默认是平地，也可以修改为Trimesh

        self.config = config
        self.device = device
        self.num_scenes = num_envs

        self.spacing_between_scenes = config.spacing_between_scenes
        self.minimal_humanoid_spacing = config.minimal_humanoid_spacing

        self.num_scenes_per_row = math.ceil(
            math.sqrt(self.num_scenes)
        ) # 尽可能的将场景排列成正方形，优先满足行即可能存在第一行有三个（num_scenes_per_row）而后面某行只有一个的情况

        self.env_length = config.map_length
        self.env_width = config.map_width
        self.border_size = config.border_size



        self.env_cols = self.num_scenes_per_row 
        self.env_rows = math.ceil(self.num_scenes / self.num_scenes_per_row)
        self.num_maps = self.env_rows * self.env_cols

        assert (
            self.env_length*self.env_width >= self.minimal_humanoid_spacing
        ), "Not enough space between humanoids, create a bigger terrain or reduce the number of envs."

        # 场景宽度长度以及间距对应的像素数
        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)
        self.border = int(self.border_size / self.horizontal_scale)

        assert (
            self.width_per_env_pixels != math.ceil(self.env_width / self.horizontal_scale)
        ), "Terrain的宽度不能被horizontal_scale整除，请检查horizontal_scale"
        assert (
            self.length_per_env_pixels != math.ceil(self.env_length / self.horizontal_scale)
        ), "Terrain的长度不能被horizontal_scale整除，请检查horizontal_scale"
        assert (
            self.border != math.ceil(self.border_size / self.horizontal_scale)
        ), "Terrain的边界不能被horizontal_scale整除，请检查horizontal_scale"


        """
        原来的boder是相当于只有最外面四周的，但我希望它是每个场景的边界，并且所有操作都不能碰到这个边界。
        """
        # 总共需要用来表示的像素数
        self.tot_cols = (
            int(self.env_rows * (self.width_per_env_pixels + 2 * self.border))
            
        )   # 每列有多少个像素
        self.tot_rows = (
            int(self.env_cols * (self.length_per_env_pixels + 2 * self.border) )
        )   # 每行有多少个像素

        self.single_env_cols = self.width_per_env_pixels + 2 * self.border
        self.single_env_rows = self.length_per_env_pixels + 2 * self.border
        # 声明所有需要的用于记录场景信息的数组（每个值代表对应的像素位置的值）
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.ceiling_field_raw = np.zeros(
            (self.tot_rows, self.tot_cols), dtype=np.int16
        ) + (3 / self.vertical_scale)

        self.walkable_field_raw = np.zeros(
            (self.tot_rows, self.tot_cols), dtype=np.int16
        )
        self.flat_field_raw = np.ones((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.scene_map = torch.zeros(
            (self.tot_rows, self.tot_cols), dtype=torch.bool, device=self.device
        )

        if self.config.load_terrain:
            print("Loading a pre-generated terrain")
            params = torch.load(self.config.terrain_path)
            self.height_field_raw = params["height_field_raw"]
            self.walkable_field_raw = params["walkable_field_raw"]
        else:
            self.generate_subterrains()

        # 在高度图数组中存储的是像素化的高度值，需要将其转换为实际的高度值
        self.height_samples = torch.tensor(self.height_field_raw, device=self.device, dtype=torch.float) * self.vertical_scale

        # 利用采样的宽度和采样点数，来先生成一个默认的采样点集以及这个点集中点的数量
        self.num_height_points, self.height_points = self.init_height_points(num_envs)

        # # 这个待处理。
        # self.vertices, self.triangles = convert_heightfield_to_trimesh(
        #     self.height_field_raw,
        #     self.horizontal_scale,
        #     self.vertical_scale,
        #     self.config.slope_threshold,
        #     flat_tolerance=0.0001,
        #     max_triangle_size=50
        # )

        self.compute_walkable_coords()
        self.compute_flat_coords()


    def compute_walkable_coords(self):
        # 准备常用量
        b = self.border                              # 单侧 border 宽度（像素）
        w = self.width_per_env_pixels + 2 * b        # 含 border 后每 env 占用的总列数
        h = self.length_per_env_pixels + 2 * b       # 含 border 后每 env 占用的总行数

        # 对每个环境块内部打 local border
        for env_id in range(self.num_scenes):
            # 计算行列索引
            row_idx = env_id // self.env_cols
            col_idx = env_id %  self.env_cols

            # 计算该块在大网格中的行/列起止
            rs = row_idx * h    # 带有 border 的起始行
            re = rs + h         # 带有 border 的结束行
            cs = col_idx * w    # 带有 border 的起始列
            ce = cs + w         # 带有 border 的结束列

            # 顶边
            self.walkable_field_raw[rs: rs + b,   cs:ce] = 1
            # 底边
            self.walkable_field_raw[re - b: re,   cs:ce] = 1
            # 左边
            self.walkable_field_raw[rs:re,       cs: cs + b] = 1
            # 右边
            self.walkable_field_raw[rs:re,       ce - b: ce] = 1

        # 转为 Torch 张量并提取所有值为0（可行走点）
        self.walkable_field = torch.tensor(
            self.walkable_field_raw, device=self.device, dtype=torch.int32
        )
        walkable_x_indices, walkable_y_indices = torch.where(self.walkable_field == 0)
        self.walkable_x_coords = walkable_x_indices * self.horizontal_scale
        self.walkable_y_coords = walkable_y_indices * self.horizontal_scale

    def compute_flat_coords(self):
        # 准备常用量
        b = self.border                              # 单侧 border 宽度（像素）
        w = self.width_per_env_pixels + 2 * b        # 含 border 后每 env 占用的总列数
        h = self.length_per_env_pixels + 2 * b       # 含 border 后每 env 占用的总行数

        # 对每个环境块内部打 local border
        for env_id in range(self.num_scenes):
            # 计算行列索引
            row_idx = env_id // self.env_cols
            col_idx = env_id %  self.env_cols

            # 计算该块在大网格中的行/列起止
            rs = row_idx * h    # 带有 border 的起始行
            re = rs + h         # 带有 border 的结束行
            cs = col_idx * w    # 带有 border 的起始列
            ce = cs + w         # 带有 border 的结束列

            # 顶边
            self.flat_field_raw[rs: rs + b,   cs:ce] = 1
            # 底边
            self.flat_field_raw[re - b: re,   cs:ce] = 1
            # 左边
            self.flat_field_raw[rs:re,       cs: cs + b] = 1
            # 右边
            self.flat_field_raw[rs:re,       ce - b: ce] = 1

        self.flat_field_raw = torch.tensor(self.flat_field_raw, device=self.device)

        flat_x_indices, flat_y_indices = torch.where(self.flat_field_raw == 0)
        self.flat_x_coords = flat_x_indices * self.horizontal_scale
        self.flat_y_coords = flat_y_indices * self.horizontal_scale

        if self.config.save_terrain:
            print("Saving this generated terrain")
            torch.save(
                {
                    "height_field_raw": self.height_field_raw,
                    "walkable_field_raw": self.walkable_field_raw,
                    "vertices": self.vertices,
                    "triangles": self.triangles,
                    "border_size": self.border_size,
                },
                self.config.terrain_path,
            )


    def generate_subterrains(self):
        self.flat_field_raw[:] = 0
        # Override to do nothing else
        pass

    