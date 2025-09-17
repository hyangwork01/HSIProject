import numpy as np
import math
import torch

from core.envs.hoi.components.terrains.utils.terrain_utils import (
    get_heights_jit, get_height_maps_jit
)
from core.envs.hoi.components.terrains.utils.terrain_config import TerrainConfig

import matplotlib.pyplot as plt
from core.envs.hoi.components.terrains.base import BaseTerrain
import math

class SceneTerrain(BaseTerrain):

    def __init__(self, config: TerrainConfig, num_envs: int, device) -> None:
        config.load_terrain = False
        config.save_terrain = False
        self.terrain_type = "Flatten" # 默认是平地，也可以修改为Trimesh

        self.config = config
        self.device = device
        self.num_scenes = num_envs

        self.spacing_between_scenes = config.spacing_between_scenes

        self.num_scenes_per_row = math.ceil(
            math.sqrt(self.num_scenes)
        ) # 尽可能的将场景排列成正方形，优先满足行即可能存在第一行有三个（num_scenes_per_row）而最后一行只有一个的情况

        self.env_length = config.map_length
        self.env_width = config.map_width
        self.border_size = config.border_size
        self.horizontal_scale = config.horizontal_scale
        self.vertical_scale = config.vertical_scale


        self.env_cols = self.num_scenes_per_row # 总共有多少列
        self.env_rows = math.ceil(self.num_scenes / self.num_scenes_per_row) # 总共有多少行
        self.num_maps = self.env_rows * self.env_cols

        self.check_info()

        self.single_env_cols = self.width_per_env_pixels + 2 * self.border
        self.single_env_rows = self.length_per_env_pixels + 2 * self.border
        # 声明所有需要的用于记录场景信息的数组（每个值代表对应的像素位置的值）
        # self.height_field_raw = np.zeros((self.single_env_rows, self.single_env_cols,self.num_scenes), dtype=np.int16)

        self.walkable_field_raw = np.zeros(
            (self.single_env_rows, self.single_env_cols,self.num_scenes), dtype=np.int16
        )

        self.scene_map = torch.zeros(
            (self.single_env_rows, self.single_env_cols,self.num_scenes), dtype=torch.bool, device=self.device
        )



        # 在高度图数组中存储的是像素化的高度值，需要将其转换为实际的高度值
        # self.height_samples = torch.tensor(self.height_field_raw, device=self.device, dtype=torch.float) * self.vertical_scale

        # 利用采样的宽度和采样点数，来先生成一个默认的采样点集以及这个点集中点的数量
        # self.num_height_points, self.height_points = self.init_height_points(num_envs)


        self._compute_walkable_coords()

        # if self.config.save_terrain:
        #     print("Saving this generated terrain")
        #     torch.save(
        #         {
        #             "height_field_raw": self.height_field_raw,
        #             "walkable_field_raw": self.walkable_field_raw,
        #             "vertices": self.vertices,
        #             "triangles": self.triangles,
        #             "border_size": self.border_size,
        #         },
        #         self.config.terrain_path,
        #     )

    def check_info(self):

        """
        判断长、宽、边界三个能否被像素化的scale整除
        """
        # 场景宽度长度以及间距对应的像素数
        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)
        self.border = int(self.border_size / self.horizontal_scale)

        assert (
            self.width_per_env_pixels != math.ceil(self.env_width / self.horizontal_scale)
        ), "Terrain的宽度不能被horizontal_scale整除"
        assert (
            self.length_per_env_pixels != math.ceil(self.env_length / self.horizontal_scale)
        ), "Terrain的长度不能被horizontal_scale整除"
        assert (
            self.border != math.ceil(self.border_size / self.horizontal_scale)
        ), "Terrain的边界不能被horizontal_scale整除"
        



    def get_env_center(self, env_ids):
        x_offsets, y_offsets = self.get_env_offsets(env_ids)
        return x_offsets + self.length_per_env_pixels / 2, y_offsets + self.width_per_env_pixels / 2


    def get_env_offsets(self, env_ids,border_offset = True):
        """
        计算给定 env_ids 的 (x, y) 偏移（不含 border 的有效区域左上角）。
        返回一维 torch.Tensor：x_offsets, y_offsets。

        Args:
            env_ids: list/ndarray/tensor，环境 ID 序列，范围 [0, self.num_scenes)
            in_world_units (bool): True 则返回米；False 返回像素

        Returns:
            x_offsets, y_offsets: 两个与 env_ids 等长的一维 torch.Tensor
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if torch.any(env_ids < 0) or torch.any(env_ids >= self.num_scenes):
            raise ValueError(f"env_ids 必须在 [0, {self.num_scenes}) 之间。")

        b = self.border
        w = self.width_per_env_pixels  + 2 * b  # 每块（含两侧 border）宽度-列方向
        h = self.length_per_env_pixels + 2 * b  # 每块（含两侧 border）高度-行方向

        row_idx = torch.div(env_ids, self.env_cols, rounding_mode='floor')
        col_idx = env_ids % self.env_cols

        # 先到块左上角（含 border），再加 b 移到有效区域左上角
        if border_offset:
            x_offsets = col_idx * w + b
            y_offsets = row_idx * h + b
        else:
            x_offsets = col_idx * w 
            y_offsets = row_idx * h
        x_offsets = x_offsets.to(torch.float32) * self.horizontal_scale
        y_offsets = y_offsets.to(torch.float32) * self.horizontal_scale


        return x_offsets, y_offsets


    def _compute_walkable_coords(self):
        """
        生成可行走掩码，并预先把每个 env 的可行走像素坐标收集好。
        约定：
        - walkable_field_raw == 0 表示可行走，== 1 表示不可行走（边界）
        - self.walkable_xy[env_id]: (K,2) 的像素坐标，原点是“有效区域左上角”（已去掉边界）
            顺序是 [x_rel_px, y_rel_px]
        """
        b = self.border

        # 1) 先清零，再把四周边界打成 1（不可行走）
        self.walkable_field_raw.fill(0)
        if b > 0:
            self.walkable_field_raw[:b, :, :]  = 1
            self.walkable_field_raw[-b:, :, :] = 1
            self.walkable_field_raw[:, :b, :]  = 1
            self.walkable_field_raw[:, -b:, :] = 1

        # 2) 转成 torch，找出所有可行走像素（值为 0）
        self.walkable_field = torch.as_tensor(
            self.walkable_field_raw, device=self.device, dtype=torch.uint8
        )
        y_idx, x_idx, env_idx = torch.where(self.walkable_field == 0)  # 注意顺序: 行(y), 列(x), env

        # 3) 把“含边界”的块坐标，变成相对“有效区域左上角(去边界)”的像素坐标
        x_rel = (x_idx - b).to(torch.int32)  # 范围: [0, width_per_env_pixels-1]
        y_rel = (y_idx - b).to(torch.int32)  # 范围: [0, length_per_env_pixels-1]

        # 4) 按 env 分桶，便于后续 O(1) 采样
        self.walkable_xy = [None] * self.num_scenes
        for env_id in range(self.num_scenes):
            m = (env_idx == env_id)
            self.walkable_xy[env_id] = torch.stack((x_rel[m], y_rel[m]), dim=1).contiguous()

    def sample_valid_locations(self, env_ids):
        """
        从给定 env_ids 中各随机采样 1 个可行走位置（世界坐标，单位：米）。

        Args:
            env_ids: 1D list/ndarray/tensor，取值范围 [0, self.num_scenes)

        Returns:
            Tensor，形状 [len(env_ids), 2]，每行为 [x_world, y_world] (float32, meters)
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).view(-1)

        if torch.any(env_ids < 0) or torch.any(env_ids >= self.num_scenes):
            raise ValueError(f"env_ids 必须在 [0, {self.num_scenes}) 之间。")

        # 有效区域左上角（世界坐标）
        # border_offset=True 表示直接给“有效区域(去边界)”左上角的世界坐标
        x0, y0 = self.get_env_offsets(env_ids, border_offset=True)  # [N], [N]

        out = []
        for i, env_id in enumerate(env_ids.tolist()):
            pool = self.walkable_xy[env_id]  # (K,2): [x_rel_px, y_rel_px]
            if pool.numel() == 0:
                raise RuntimeError(f"Env {env_id} 没有可行走点，请检查边界/尺寸设置。")

            sel = torch.randint(0, pool.shape[0], (1,), device=self.device)
            xy_rel_px = pool[sel].squeeze(0).to(torch.float32)  # (2,)

            # 像素 -> 米：世界坐标 = 有效区域左上角(米) + 相对像素 * horizontal_scale(米/像素)
            xy_world = torch.stack((
                x0[i] + xy_rel_px[0] * self.horizontal_scale,
                y0[i] + xy_rel_px[1] * self.horizontal_scale
            ), dim=0)  # (2,)

            out.append(xy_world)

        return torch.stack(out, dim=0)  # [N, 2]
    


    # def init_height_points(self, num_envs):
    #     """
    #     Pre-defines the grid for the height-map observation.
    #     """
    #     y = torch.tensor(
    #         np.linspace(
    #             -self.config.sample_width,
    #             self.config.sample_width,
    #             self.config.num_samples_per_axis,
    #         ),
    #         device=self.device,
    #         requires_grad=False,
    #     )
    #     x = torch.tensor(
    #         np.linspace(
    #             -self.config.sample_width,
    #             self.config.sample_width,
    #             self.config.num_samples_per_axis,
    #         ),
    #         device=self.device,
    #         requires_grad=False,
    #     )
    #     grid_x, grid_y = torch.meshgrid(x, y)

    #     num_height_points = grid_x.numel()
    #     points = torch.zeros(
    #         num_envs,
    #         num_height_points,
    #         3,
    #         device=self.device,
    #         requires_grad=False,
    #     )
    #     points[:, :, 0] = grid_x.flatten()
    #     points[:, :, 1] = grid_y.flatten()
    #     return num_height_points, points
    
    # def get_ground_heights(self, locations: torch.Tensor) -> torch.Tensor:
    #     """
    #     Get the height of the terrain at the specified locations.
    #     """
    #     return get_heights_jit(
    #         locations=locations,
    #         height_samples=self.height_samples,
    #         horizontal_scale=self.horizontal_scale,
    #     )

    # def get_height_maps(self, root_states, env_ids=None, return_all_dims=False):
    #     """
    #     Generates a 2D heightmap grid observation rotated w.r.t. the character's heading.
    #     Each sample is the billinear interpolation between adjacent points.
    #     """
    #     if env_ids is not None:
    #         height_points = self.height_points[env_ids].clone()
    #     else:
    #         height_points = self.height_points.clone()

    #     return get_height_maps_jit(
    #         base_rot=root_states.root_rot,
    #         base_pos=root_states.root_pos,
    #         height_points=height_points,
    #         height_samples=self.height_samples,
    #         num_height_points=self.num_height_points,
    #         terrain_horizontal_scale=self.horizontal_scale,
    #         w_last=True,
    #         return_all_dims=return_all_dims,
    #     )

