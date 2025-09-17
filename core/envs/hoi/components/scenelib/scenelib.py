import logging
import random
import copy
from dataclasses import dataclass, field, MISSING
from typing import List, Optional, Tuple, Dict,Type
from isaac_utils import torch_utils
import torch
import trimesh
import os
from pathlib import Path

from core.envs.hoi.env_utils.object_utils import (
    as_mesh,
    compute_bounding_box,
)
# from core.scenelib.utils.usd_utils import (compute_usd_dims)

from core.envs.hoi.components.terrains.hoi_terrain import SceneTerrain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ObjectMotion:
    """
    Contains motion data for an object.
    Frames is a list of dictionaries. Each frame is expected to have keys:
      'translation': tuple of 3 floats,
      'rotation': tuple of 4 floats.
    fps: Frames per second for the motion (default is 30.0).
    """
    frames: List[dict] = field(default_factory=list)
    fps: float = None


@dataclass
class ObjectOptions:
    """
    Contains options for configuring object properties in the simulator.
    """
    # fix_base_link: bool = field(default=MISSING)
    # vhacd_enabled: bool = None
    # vhacd_params: Dict = field(default_factory=lambda: {
    #     "resolution": None,
    #     "max_convex_hulls": None,
    #     "max_num_vertices_per_ch": None,
    # })
    # density: float = None
    # angular_damping: float = None
    # linear_damping: float = None
    # max_angular_velocity: float = None
    # default_dof_drive_mode: str = None
    # override_com: bool = None
    # override_inertia: bool = None
    visual_material: bool = True # 若为False，则不加载visual

    def to_dict(self) -> Dict:
        """Convert options to a dictionary, excluding None values."""
        options_dict = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                options_dict[field_name] = field_value
        return options_dict

@dataclass
class RigidOptions(ObjectOptions):
    # mass_props
    density: float = None # 加载mass_props时使用

    # rigid_props
    rigid_body_enabled: bool = True 
    kinematic_enabled: bool = False # False则不固定物体，True则固定物体
    angular_damping: float = None
    linear_damping: float = None
    max_linear_velocity: float = None
    max_angular_velocity: float = None
    retain_accelerations: bool = None

    # collision_props
    collision_enabled: bool = True
    
    
    # activate_contact_sensors
    activate_contact_sensors: bool = False

    

@dataclass
class ArticulationOptions(ObjectOptions):
    # mass_props
    density: float = None

    # rigid_props
    rigid_body_enabled: bool = True 
    kinematic_enabled: bool = True 
    angular_damping: float = None
    linear_damping: float = None
    max_linear_velocity: float = None
    max_angular_velocity: float = None
    retain_accelerations: bool = None

    # articulation_props
    articulation_enabled: bool = True
    enabled_self_collisions: bool = False
    fix_root_link: bool = True

@dataclass
class SceneObject:
    """
    Represents an object inside a scene.
    Defaults to translation (0,0,0) and rotation (0,0,0,1) if not provided.
    """
    object_path: str = field(default=MISSING)
    translation: Tuple[float, float, float] = field(default=MISSING)
    rotation: Tuple[float, float, float, float] = field(default=MISSING)
    scale: Tuple[float, float, float] = field(default=MISSING)
    object_type: str = field(default=MISSING)
    id: int = field(default=MISSING)
    options: ObjectOptions = field(default_factory=ObjectOptions)
    motion: Optional[ObjectMotion] = None
    object_dims: Tuple[float, float, float, float, float, float] = None




@dataclass
class Scene:
    """
    Represents a scene consisting of one or more SceneObjects.
    An offset (x, y) indicates the scene's location.
    """
    # rigid_objects: List[SceneObject] = field(default_factory=list)
    # articulation_objects: List[SceneObject] = field(default_factory=list)
    objects: List[SceneObject] = field(default_factory=list)
    offset: Tuple[float, float] = (0.0, 0.0)

    def add_object(self, scene_object: SceneObject):
        """Add an object to the scene."""
        self.objects.append(scene_object)


@dataclass
class ObjectState:
    """
    Represents the state of an object, including translations and rotations as torch tensors.
    """
    translations: torch.Tensor
    rotations: torch.Tensor


# @dataclass
# class SpawnInfo:
#     """
#     Contains information about how to spawn an object in the scene.
#     """
#     id: int
#     object_path: str
#     object_options: ObjectOptions
#     object_dims: Tuple[float, float, float, float, float, float] = None
#     is_first_instance: bool = True
#     first_instance_id: int = None

class SceneLib:
    """
    A simplified scene library.

    - The user instantiates SceneLib with a config dictionary (which contains num_envs and scene offset parameters) and a device string.
    - Scenes are provided as a list of Scene dataclasses via create_scenes(). If fewer scenes than num_envs are provided,
      scenes are replicated (either sequentially or randomly).
    - Object motions (a list of frames per SceneObject) are combined into unified tensors stored within the SceneLib instance.
    - The get_object_pose method interpolates the pose of an object at a specified time using delta time (dt).

    Note: SceneObjects no longer have explicit IDs; their order defines their unique index.
    """
    def __init__(self,cfg, num_envs: int,device: str = "cpu"):
        """
        Args:
            config: Dictionary containing keys:
                - num_envs: int, number of environments.
            device: Device identifier for torch tensors.
        """
        self.device = device
        self.num_envs = num_envs
        self.scenes: List[Scene] = []
        self.num_objects_per_scene = None
        self._total_spawned_scenes = 0
        self._scene_offsets = []
        # self._object_spawn_list = []
        # self._object_path_to_id = {}

        # Placeholders for aggregated motion data
        self.object_translations = None
        self.object_rotations = None
        self.motion_lengths = None  # In seconds, per object
        self.motion_starts = None   # Starting index in the unified tensor for each object
        self.motion_dts = None      # Delta time for each object's motion
        self.motion_num_frames = None  # Number of frames in each object's motion
        
        self.cfg = cfg
            

    def create_single_scene(self,scene_id) -> Scene:
        flag = False
        scenes =self.cfg.get("scenes",[])
        scenes_abspath = self.cfg.get("scenes_abspath")
        scenes_name = self.cfg.get("scenes_name")
        for scene in scenes:
            if scene.get("scene_id") == scene_id:
                from core.scenelib.utils.usd_utils import compute_usd_dims

                flag = True
                scene_path = scene.get("scene_path")
                scene_abspath = os.path.join(scenes_abspath,scene_path)
                filter_list = scene.get("filter_list")
                rigidbody_list = scene.get("rigidbody_list")
                articulation_list = scene.get("articulation_list")
                print(f"About to create the “{scenes_name}” scene at path: {scene_abspath}")
                objects_list = []
                idx = 0
                for obj in scene.get("objects"):
                    obj_id = obj.get("obj_id")
                    if obj_id in filter_list:
                        continue
                    obj_abspath = os.path.join(scene_abspath,obj.get("obj_path"))
                    obj_translate = obj.get("obj_translate")
                    obj_rotateXYZW = obj.get("obj_rotateXYZW")
                    obj_scale = obj.get("obj_scale")
                    obj_name = obj.get("obj_name")
                    if obj_id in rigidbody_list:
                        object_options = RigidOptions(
                            kinematic_enabled=False,
                            density=1000,
                            )
                        min_pt, max_pt = compute_usd_dims(obj_abspath)
                        min_x, min_y, min_z = min_pt[0], min_pt[1], min_pt[2]
                        max_x, max_y, max_z = max_pt[0], max_pt[1], max_pt[2]    
                        object = SceneObject(
                            object_path=obj_abspath,
                            options=object_options,
                            translation=tuple(obj_translate),    
                            rotation=tuple(obj_rotateXYZW),
                            scale=tuple(obj_scale),
                            object_type= "rigid",
                            id=idx,
                            object_dims = (min_x, max_x, min_y, max_y, min_z, max_z), # 除过scale后的尺寸
                        )
                        objects_list.append(object)
                        print(f"Create rigidbody object,Idx: {idx}, obj_id: {obj_id}, obj_name: {obj_name}") 
                    elif obj_id in articulation_list:
                        min_pt, max_pt = compute_usd_dims(obj["Objpath"])
                        min_x, min_y, min_z = min_pt[0], min_pt[1], min_pt[2]
                        max_x, max_y, max_z = max_pt[0], max_pt[1], max_pt[2]

                        object_options = ArticulationOptions(
                            density=1000,
                            kinematic_enabled=True,
                            articulation_enabled=True,
                            enabled_self_collisions=False,
                            fix_root_link=True,
                            )
                        object = SceneObject(
                            object_path=obj_abspath,
                            options=object_options,
                            translation=tuple(obj_translate),    
                            rotation=tuple(obj_rotateXYZW),
                            scale=tuple(obj_scale),
                            object_type= "articulation",
                            id=idx,
                            object_dims = (min_x, max_x, min_y, max_y, min_z, max_z),
                        )
                        objects_list.append(object)
                        print(f"Create articulation object,Idx: {idx}, obj_id: {obj_id}, obj_name: {obj_name}")                        
                    else:
                        raise ValueError(f"Object {obj_id} is not in the rigidbody_list or articulation_list.")
                    idx += 1
                break
        if flag == False:
            raise ValueError(f"Scene {scene_id} does not exist in the config file.")
        return Scene(objects=objects_list)

    def create_scenes(self,terrain: SceneTerrain,scene_id: str = None) -> List[Scene]:
        if scene_id is None:
            scenes = self.cfg.get("scenes")
            scene_id = scenes[0].get("scene_id")
        scenes_list = []

        scene = self.create_single_scene(scene_id)
        scenes_list.append(scene)
        assigned_scenes = list(scenes_list)
        num_scenes = len(scenes_list)
        if num_scenes < self.num_envs:
            for i in range(self.num_envs - num_scenes):
                scene = copy.deepcopy(assigned_scenes[0])
                assigned_scenes.append(scene)

        rigid_objects_counts = []
        articulation_objects_counts = []



        for idx, scene in enumerate(assigned_scenes):
            # x_offset = ((idx % terrain.num_scenes_per_column + 1) * terrain.spacing_between_scenes + terrain.border * terrain.horizontal_scale)
            # y_offset = ((idx // terrain.num_scenes_per_column + 1) * terrain.spacing_between_scenes + terrain.scene_y_offset)
            cx, cy = terrain.get_env_center([idx])     
            x_offset = float(cx[0].item())
            y_offset = float(cy[0].item())    
            scene.offset = (x_offset, y_offset)
            
            # Compute integer grid location
            logger.info("Assigned scene id %s to offset (%.2f, %.2f)", idx, x_offset, y_offset)
            self._scene_offsets.append((x_offset, y_offset))
        
            # Check that all scenes have the same number of rigid/articulation objects
            rigid_count = 0
            articulation_count = 0
            for obj in scene.objects:
                if obj.object_type == "rigid":
                    rigid_count += 1
                elif obj.object_type == "articulation":
                    articulation_count += 1
                if obj.object_dims is None:
                    logger.error(f"Scene {idx} object {obj.id} does not have dimensions")
                    raise ValueError("Object does not have dimensions")
            rigid_objects_counts.append(rigid_count)
            articulation_objects_counts.append(articulation_count)

        if len(set(rigid_objects_counts)) != 1:
            logger.error("All scenes must have the same number of rigid_objects. Found counts: %s", rigid_objects_counts)
            raise ValueError("Scenes have inconsistent number of rigid_objects: " + str(rigid_objects_counts))
        if len(set(articulation_objects_counts)) != 1:
            logger.error("All scenes must have the same number of articulation_objects. Found counts: %s", articulation_objects_counts)
            raise ValueError("Scenes have inconsistent number of articulation_objects: " + str(articulation_objects_counts))

        self.num_objects_per_scene = rigid_objects_counts[0] + articulation_objects_counts[0]

        self._total_spawned_scenes = len(assigned_scenes)
        self.scenes = assigned_scenes
        # Automatically combine object motions so the user does not have to call it
        # TODO:后续需要去优化object的motion的读取等。self.combine_object_motions()
        return assigned_scenes
    




    @property
    def total_spawned_scenes(self) -> int:
        """Returns the total number of scenes that were spawned."""
        return self._total_spawned_scenes

    @property
    def scene_offsets(self) -> List[Tuple[float, float]]:
        """Returns the list of scene offsets."""
        return self._scene_offsets

    # @property
    # def object_spawn_list(self) -> List[SpawnInfo]:
    #     """Returns the list of canonical objects with their properties."""
    #     return self._object_spawn_list

    # @property
    # def object_path_to_id(self) -> Dict[str, int]:
    #     """Returns the mapping from object paths to their spawn list indices."""
    #     return self._object_path_to_id


# ----------------------------------------------------------------------------
# Example usage:
# if __name__ == "__main__":
#     import torch
    
#     # Define a dummy Terrain for example usage
#     class DummyTerrain:
#         def __init__(self):
#             self.num_scenes_per_column = 2
#             self.spacing_between_scenes = 5.0
#             self.border = 2.0
#             self.horizontal_scale = 1.0
#             self.scene_y_offset = 0.0
#             self.device = "cpu"

#         def is_valid_spawn_location(self, locations):
#             return torch.tensor(True)

#         def mark_scene_location(self, x, y):
#             pass

#     scene_lib = SceneLib(num_envs=4, device="cpu")

#     # Create SceneObjects with options
#     obj1 = SceneObject(
#         object_path="cup.urdf",
#         translation=(1.0, 0.0, 0.0),
#         rotation=(0.0, 0.0, 0.0, 1.0),
#         motion=ObjectMotion(
#             frames=[
#                 {"translation": (1.0, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0, 1.0)},
#                 {"translation": (1.5, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0, 1.0)}
#             ],
#             fps=30.0
#         ),
#         options=ObjectOptions(
#             vhacd_enabled=True,
#             vhacd_params={
#                 "resolution": 50000,
#                 "max_convex_hulls": 128,
#                 "max_num_vertices_per_ch": 64
#             },
#             fix_base_link=True
#         )
#     )

#     obj2 = SceneObject(
#         object_path="obstacle.urdf",
#         options=ObjectOptions(
#             vhacd_enabled=True,
#             vhacd_params={"resolution": 50000},
#             fix_base_link=True
#         )
#     )
#     scene1 = Scene(id=1, objects=[obj1, obj2])

#     obj3 = SceneObject(
#         object_path="chair.urdf",
#         translation=(2.0, 2.0, 0.0),
#         options=ObjectOptions(
#             vhacd_enabled=True,
#             vhacd_params={"resolution": 50000},
#             fix_base_link=True
#         )
#     )
#     obj4 = SceneObject(
#         object_path="table.urdf",
#         translation=(2.5, 2.0, 0.0),
#         options=ObjectOptions(
#             vhacd_enabled=True,
#             vhacd_params={"resolution": 50000},
#             fix_base_link=True
#         )
#     )
#     scene2 = Scene(id=2, objects=[obj3, obj4])

#     scenes = [scene1, scene2]

#     terrain = DummyTerrain()
#     assigned_scenes = scene_lib.create_scenes(scenes, terrain, replicate_method="random")
#     for idx, scene in enumerate(assigned_scenes):
#         logger.info("Environment %d assigned Scene with objects %s with offset %s", idx, scene.objects, scene.offset)

#     # get_object_pose now returns an ObjectState and combine_object_motions is automatically called in create_scenes()
#     time = 1. / 30 * 0.5
#     pose_obj0 = scene_lib.get_object_pose(object_indices=torch.tensor([0]), time=torch.tensor([time]))
#     logger.info("Pose for object at index 0 at time %s:\nTranslations: %s\nRotations: %s", time, pose_obj0.translations, pose_obj0.rotations)
