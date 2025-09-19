import logging
# ===== 放在文件靠前位置：轻量随机材质桶 =====
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import random
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    modify_collision_properties,
    modify_mass_properties,
)
import os
from typing import Dict, Tuple, Optional

from isaaclab.assets import RigidObjectCfg


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class _PhysMat:
    static_friction: float
    dynamic_friction: float
    restitution: float

@dataclass
class _ColliderPreset:
    rest_offset: float
    contact_offset: float

@dataclass
class _MassPreset:
    mass: Optional[float] = None
    density: Optional[float] = None

class RandomMaterialBucket:
    """按范围随机生成若干“物理材质/质量/碰撞偏移”预设，并能应用到给定 prim。"""
    def __init__(
        self,
        *,
        phys_friction_range: Tuple[float, float],
        restitution_range: Tuple[float, float],
        mass_range: Tuple[float, float],
        rest_offset_range: Tuple[float, float],
        contact_offset_margin: float,
        num_phys: int,
        num_mass: int,
        num_collider: int,
        enforce_static_gte_dynamic: bool = True,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
        self._phys: Dict[str, _PhysMat] = {}
        self._mass: Dict[str, _MassPreset] = {}
        self._coll: Dict[str, _ColliderPreset] = {}
        self._spawned_phys_paths: Dict[str, str] = {}

        for i in range(max(0, int(num_phys))):
            sf = random.uniform(*phys_friction_range)
            df = random.uniform(phys_friction_range[0], sf) if enforce_static_gte_dynamic else random.uniform(*phys_friction_range)
            rs = random.uniform(*restitution_range)
            self._phys[f"phys_rand_{i}"] = _PhysMat(sf, df, rs)

        for i in range(max(0, int(num_mass))):
            m = random.uniform(*mass_range)
            self._mass[f"mass_rand_{i}"] = _MassPreset(mass=m, density=None)

        for i in range(max(0, int(num_collider))):
            r = random.uniform(*rest_offset_range)
            c = r + random.uniform(0.0, max(0.0, contact_offset_margin))
            self._coll[f"collider_rand_{i}"] = _ColliderPreset(r, c)

    # 预先在 /World/Materials/Physics 生成刚体材质（便于后续绑定）
    def pre_spawn_all_phys_materials(self):
        for name in self._phys.keys():
            self._spawn_phys(name)

    def _spawn_phys(self, name: str) -> str:
        if name in self._spawned_phys_paths:
            return self._spawned_phys_paths[name]
        p = self._phys[name]
        prim_path = f"/World/Materials/Physics/{name}"
        mat_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=p.static_friction,
            dynamic_friction=p.dynamic_friction,
            restitution=p.restitution,
        )
        mat_cfg.func(prim_path, mat_cfg)  # 生成材质 Prim（Isaac Lab spawner API）
        self._spawned_phys_paths[name] = prim_path
        return prim_path

    def phys_keys(self) -> List[str]: return list(self._phys.keys())
    def mass_keys(self) -> List[str]: return list(self._mass.keys())
    def coll_keys(self) -> List[str]: return list(self._coll.keys())

    def sample_combo(self) -> Tuple[str, str, str]:
        import random
        return (random.choice(self.phys_keys()),
                random.choice(self.coll_keys()),
                random.choice(self.mass_keys()))

    def apply_to_prim(self, prim_path: str, *, physics_mat: Optional[str], collider: Optional[str], mass: Optional[str]):
        if physics_mat:
            path = self._spawn_phys(physics_mat)
            # 运行时绑定刚体材质（覆盖子树。Isaac Lab 提供 sim.utils 的绑定函数）
            sim_utils.bind_physics_material(prim_path, path, stronger_than_descendants=True)  # :contentReference[oaicite:3]{index=3}
        if collider:
            cp = self._coll[collider]
            modify_collision_properties(prim_path, CollisionPropertiesCfg(rest_offset=cp.rest_offset, contact_offset=cp.contact_offset))  # :contentReference[oaicite:4]{index=4}
        if mass:
            mp = self._mass[mass]
            modify_mass_properties(prim_path, MassPropertiesCfg(mass=mp.mass, density=mp.density))  # :contentReference[oaicite:5]{index=5}


class SceneLib:
    def __init__(self,cfg,num_envs: int,device: str = "cpu"):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        self.scene_id: Optional[str] = None
        self._rigid_objects_cfg_list: Dict[str, RigidObjectCfg] = {}
        self._rigid_objects_collection_cfg_list: Dict[str, List[RigidObjectCollectionCfg]] = {}
        self.num_objects_per_scene: Optional[int] = None


        # ===== 在 SceneLib 内声明“随机材质桶”的参数（可在外部修改这些成员来重建桶） =====
        self.num_rand_phys: int = 5
        self.num_rand_mass: int = 5
        self.num_rand_collider: int = 5
        self.phys_friction_range: Tuple[float, float] = (0.0, 1.2)
        self.restitution_range: Tuple[float, float] = (0.0, 0.5)
        self.mass_range: Tuple[float, float] = (0.1, 5.0)
        self.rest_offset_range: Tuple[float, float] = (0.0, 0.005)
        self.contact_offset_margin: float = 0.005
        self.enforce_static_gte_dynamic: bool = True
        self.rand_seed: Optional[int] = None

        # 构建场景配置（你已有的逻辑）
        self.create_scene_cfg()
        self.count_objects()

        # 基于上面成员“生成随机材质桶”
        self._build_random_material_bucket()

    # ---------------- 公共 API ----------------
    def configure_random_bucket(
        self,
        *,
        num_rand_phys: Optional[int] = None,
        num_rand_mass: Optional[int] = None,
        num_rand_collider: Optional[int] = None,
        phys_friction_range: Optional[Tuple[float, float]] = None,
        restitution_range: Optional[Tuple[float, float]] = None,
        mass_range: Optional[Tuple[float, float]] = None,
        rest_offset_range: Optional[Tuple[float, float]] = None,
        contact_offset_margin: Optional[float] = None,
        enforce_static_gte_dynamic: Optional[bool] = None,
        seed: Optional[int] = None,
        rebuild: bool = True,
    ):
        """外部可随时修改 SceneLib 的随机化参数；需要的话重建桶。"""
        if num_rand_phys is not None: self.num_rand_phys = num_rand_phys
        if num_rand_mass is not None: self.num_rand_mass = num_rand_mass
        if num_rand_collider is not None: self.num_rand_collider = num_rand_collider
        if phys_friction_range is not None: self.phys_friction_range = phys_friction_range
        if restitution_range is not None: self.restitution_range = restitution_range
        if mass_range is not None: self.mass_range = mass_range
        if rest_offset_range is not None: self.rest_offset_range = rest_offset_range
        if contact_offset_margin is not None: self.contact_offset_margin = contact_offset_margin
        if enforce_static_gte_dynamic is not None: self.enforce_static_gte_dynamic = enforce_static_gte_dynamic
        if seed is not None: self.rand_seed = seed
        if rebuild:
            self._build_random_material_bucket()

    def reset_randomize_all(self) -> Dict[str, Tuple[str, str, str]]:
        """
        在 reset 时对所有对象进行随机化，返回分配表：
        { obj_name: (physics_mat_name, collider_preset_name, mass_preset_name) }
        """
        assignments: Dict[str, Tuple[str, str, str]] = {}
        for obj_name in self._rigid_objects_cfg_list.keys():
            assignments[obj_name] = self.rand_bucket.sample_combo()
        self.apply_bucket_assignments(assignments)
        return assignments

    def apply_bucket_assignments(self, assignments: Dict[str, Tuple[str, str, str]]):
        """
        把给定的(物理材质, 碰撞器偏移, 质量)三元组分配应用到每个对象。
        注意：如果 prim_path 模板里有 env 通配符，本函数会对每个 env 展开。
        """
        for obj_name, (pm, cp, ms) in assignments.items():
            obj_cfg = self._rigid_objects_cfg_list.get(obj_name)
            if obj_cfg is None:
                continue
            # 展开 env 通配路径（/World/envs/env_.*/Object_X -> /World/envs/env_i/Object_X）
            prim_paths = self._expand_prim_paths(obj_cfg.prim_path)
            for prim_path in prim_paths:
                self.rand_bucket.apply_to_prim(prim_path, physics_mat=pm, collider=cp, mass=ms)

    # ---------------- 内部：构桶 & 工具 ----------------
    def _build_random_material_bucket(self):
        """根据 SceneLib 成员生成 RandomMaterialBucket。"""
        self.rand_bucket = RandomMaterialBucket(
            phys_friction_range=self.phys_friction_range,
            restitution_range=self.restitution_range,
            mass_range=self.mass_range,
            rest_offset_range=self.rest_offset_range,
            contact_offset_margin=self.contact_offset_margin,
            num_phys=self.num_rand_phys,
            num_mass=self.num_rand_mass,
            num_collider=self.num_rand_collider,
            enforce_static_gte_dynamic=self.enforce_static_gte_dynamic,
            seed=self.rand_seed,
        )

        # 可选：提前把所有物理材质 prim 生成出来，避免 reset 时首次绑定的延迟
        for name in self.rand_bucket.phys_keys():
            self.rand_bucket._spawn_phys_material(name)

    def _expand_prim_paths(self, prim_path_template: str) -> List[str]:
        """将形如 '/World/envs/env_.*/Object_1' 的模板展开为每个 env 的实际 prim 路径。"""
        # 简单替换：若模板包含 'env_.*/'，按 num_envs 展开
        marker = "env_.*/"
        if marker in prim_path_template:
            paths = []
            for i in range(self.num_envs):
                paths.append(prim_path_template.replace(marker, f"env_{i}/"))
            return paths
        else:
            return [prim_path_template]

    # ---------------- 你已有的场景构建逻辑（示例占位） ----------------
    def create_scene_cfg(self):
        scenes = self.cfg.get("scenes", []) or []
        scenes_abspath = self.cfg.get("scenes_abspath")
        scenes_name = self.cfg.get("scenes_name")

        for scene in scenes:
            if self.scene_id is None:
                self.scene_id = scene.get("scene_id")
            if scene.get("scene_id") != self.scene_id:
                continue

            scene_path = scene.get("scene_path")
            scene_abspath = os.path.join(scenes_abspath, scene_path)
            filter_list = set(scene.get("filter_list") or [])
            rigidbody_list = set(scene.get("rigidbody_list") or [])
            print(f"About to create the “{scenes_name}” scene at path: {scene_abspath}")

            for obj in scene.get("objects", []):
                obj_id = obj.get("obj_id")
                if obj_id in filter_list:
                    continue

                obj_abspath = os.path.join(scene_abspath, obj.get("obj_path"))
                pos = tuple(obj.get("obj_translate", [0.0, 0.0, 0.0]))
                q = obj.get("obj_rotateXYZW", [0.0, 0.0, 0.0, 1.0])
                rot_wxyz = (q[3], q[0], q[1], q[2])  # (w,x,y,z)
                scale = tuple(obj.get("obj_scale", [1.0, 1.0, 1.0]))

                if obj_id in rigidbody_list:
                    rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False)
                    collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0)
                    mass_props = sim_utils.MassPropertiesCfg(density=None)

                    spawn_cfg = sim_utils.UsdFileCfg(
                        usd_path=str(obj_abspath),
                        scale=scale,
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                        mass_props=mass_props,
                        activate_contact_sensors=True,
                    )
                    name = f"Object_{obj_id}"
                    self._rigid_objects_cfg_list[name] = RigidObjectCfg(
                        prim_path=f"/World/envs/env_.*/{name}",
                        spawn=spawn_cfg,
                        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot_wxyz),
                    )
            break  # 只处理匹配的第一个 scene_id

    def count_objects(self):
        self.num_objects_per_scene = len(self._rigid_objects_cfg_list)