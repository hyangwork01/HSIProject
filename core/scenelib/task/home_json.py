from core.scenelib.scenelib import Scene,SceneObject,RigidOptions,SceneLib,ArticulationOptions
from typing import List
import copy
import logging
from core.envs.base_env.env_utils.terrains.terrain import Terrain
import torch
from core.scenelib.utils.usd_utils import compute_usd_dims
import os
from core.scenelib.task.basetask import BaseTaskScene
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 根据需求创建相应类的scene即可
class TaskScene(BaseTaskScene):
    """euler_to_quat
    Represents a scene consisting of one or more SceneObjects.
    An offset (x, y) indicates the scene's location.
    """


    # def get_usd_path(self,objects_path: str) -> list[str]:
    #     root = Path(objects_path)
    #     usd_paths: list[str] = []
    #     if not root.is_dir():
    #         raise ValueError(f"Objects path '{objects_path}' 不存在或不是目录。")
    #     for usd_file in root.rglob("instance.usd"):
    #         usd_paths.append(str(usd_file.resolve()))
    #     return usd_paths

    

    def __init__(self,terrain: Terrain,num_envs: int =1, objects_path: str = None, assign_method: str = "sequential", replicate_method: str = "random",create_type:str ="single"):
        if objects_path is None:
            print("No objects_path provided")
            raise ValueError("No objects_path provided")
        else:
            # objects_path 存放的是usd总的文件路径。
            self.objects_path = objects_path

        self.task_name: str = "Home"

        # self.home_usd_path = os.path.join(self.objects_path,"../Home/001/instance.usd")
        self.home_cfg_path = "/home/luohy/MyRepository/MyDataSets/Data/Home/config.json"
        self.home_type ="Livingroom"
        self.home_idx = 7
        # 取用USD时的方法，是否打乱顺序，还是按顺序取用
        self.assign_method = assign_method
        # numenv大于asssets时，复制方法，是否随机复制，还是按顺序复制，用于multi
        self.replicate_method = replicate_method
        # 创建场景的方法，在create_scenes时是只用第一个路径创建单个场景，还是用多个路径创建多个场景
        self.create_type = create_type


        self.terrain = terrain
        self.terrain.spacing_between_scenes = 20
        self.num_envs = num_envs
        self.idx = 0



    def create_scenes(self) -> List[Scene]:

        if self.create_type == "single":
            assigned_scenes = self.create_single_scenes()
        elif self.create_type == "multi":   
            assigned_scenes = self.create_multi_scenes()
        else:
            logger.error("Unknown assign method: %s", self.assign_method)
            raise ValueError("Assign method must be either 'sequential' or 'random'.")


        return assigned_scenes

    def create_single_scenes(self) -> List[Scene]:
        scenes_list = []
        scenes_list.append(self.set_scene())
        assigned_scenes = list(scenes_list)
        num_scenes = len(scenes_list)
        if num_scenes < self.num_envs:
            for i in range(self.num_envs - num_scenes):
                scene = copy.deepcopy(assigned_scenes[0])
                assigned_scenes.append(scene)

        
        return assigned_scenes

    #TODO：生成多个场景，如果不足num_envs并将其全部复制到num_envs个场景中。目前create_multi_scenes实现了，但是其配套相关的reset暂时未实现。
    def create_multi_scenes(self):

        pass

    # 用于搭建一整个场景。请传入用于搭建场景的相关物体的Path。 
    def set_scene(self) -> Scene:
        home = self.set_home(self.home_cfg_path,self.home_type,self.home_idx)        
        return Scene(objects= home)

        
    # 读取一整个完整的usd大小的场景，然后将其中的所有物体依次序加入到objects中
    def set_home(self,home_cfg_path: str,home_type: str = "Livingroom",home_idx: int = 1) -> List[SceneObject]:
        with open(home_cfg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        scene_path = None
        for scene in data.get("scenes", []):
            if scene.get("scene_id") == f"{home_idx:03d}" and scene.get("scene_type") == home_type:
                scene_path = os.path.join(os.path.dirname(home_cfg_path),scene.get("cfg_path"))
        if scene_path is None:
            raise ValueError(f"Home with id {home_idx} and type {home_type} not found in {home_cfg_path}")
        
        with open(scene_path, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
        
        filter_list = scene_data["filter_list"]
        obj_list = []
        for obj in scene_data["objects"]:
            if obj["Id"]  not in filter_list:
                # 需要将obj的相对路径改成绝对路径
                obj["Objpath"] = os.path.join(os.path.dirname(scene_path),obj["Objpath"])
                obj_list.append(obj)

        obj_list.sort(key=lambda x: x["Id"])
        scene_list = []



        # test_list =[163,] # 用于测试的list
        for idx,obj in enumerate(obj_list):
            # if obj["Id"] in test_list:
                
            #     object_options = RigidOptions(
            #         kinematic_enabled=True,
            #         density=1000,
            #         )
            #     min_pt, max_pt = compute_usd_dims(obj["Objpath"])
            #     min_x, min_y, min_z = min_pt[0], min_pt[1], min_pt[2]
            #     max_x, max_y, max_z = max_pt[0], max_pt[1], max_pt[2]

            #     obj_dir = os.path.join(os.path.dirname(scene_path),"cube")
            #     os.makedirs(obj_dir, exist_ok=True)
            #     obj_path = os.path.join(obj_dir,f"{obj['Id']}.usda")
            #     obj["Objpath"] = obj_path
            #     from pxr import Usd,UsdGeom,Gf,UsdPhysics

            #     stage = Usd.Stage.CreateNew(obj_path)
            #     root = stage.DefinePrim("/Root", "Xform")
            #     stage.SetDefaultPrim(root)
            #     stage.DefinePrim("/Root/Meshes", "Scope")
            #     cube = UsdGeom.Cube.Define(stage, "/Root/Meshes/BoxGeom")
            #     cube.GetSizeAttr().Set(1.0)
            #     length, width, height = max_x - min_x, max_y - min_y, max_z - min_z
            #     xformable = UsdGeom.Xformable(root)                  # 转为 Xformable 接口
            #     translateOp = xformable.AddTranslateOp()
            #     translateOp.Set(Gf.Vec3d(0.0, 0.0, 6.0))
            #     scaleOp = xformable.AddScaleOp()                                                   
            #     scaleOp.Set(Gf.Vec3d(length, width, height))        


            #     UsdPhysics.RigidBodyAPI.Apply(root)    # 标记该 prim 为刚体驱动对象  
            #     UsdPhysics.CollisionAPI.Apply(root)    # 为其几何形状添加碰撞定义  
            #     UsdPhysics.MassAPI.Apply(root)         # （可选）定义质量属性 
            #     stage.Save()
                

            #     # object = SceneObject(
            #     #     object_path=obj["Objpath"],
            #     #     options=object_options,
            #     #     translation=(obj["Translate"][0]-0.12,obj["Translate"][1]+1,obj["Translate"][2]+0.72),    
            #     #     # rotation=(0.0, 0.0, 0.0, 1.0),
            #     #     rotation=obj["RotateXYZW"],
            #     #     scale=obj["Scale"],
            #     #     object_type= "rigid",
            #     #     id=idx,
            #     #     object_dims = (min_x, max_x, min_y, max_y, min_z, max_z), # 除过scale后的尺寸
            #     # )

            #     object = SceneObject(
            #         object_path=obj["Objpath"],
            #         options=object_options,
            #         translation=obj["Translate"],    
            #         # rotation=(0.0, 0.0, 0.0, 1.0),
            #         rotation=obj["RotateXYZW"],
            #         scale=(length, width, height),
            #         object_type= "rigid",
            #         id=idx,
            #         object_dims = (min_x, max_x, min_y, max_y, min_z, max_z), # 除过scale后的尺寸
            #     )
            #     scene_list.append(object)
            #     continue


                
            if obj["Id"] in scene_data["rigidbody_list"]:
                # if obj["Id"] ==19:
                #     object_options = RigidOptions(
                #         kinematic_enabled=False,
                #         density=1000,
                #         )
                # else:
                #     object_options = RigidOptions(
                #         kinematic_enabled=True,
                #         density=1000,
                #         )
                if obj["Id"] !=19:
                    continue
                object_options = RigidOptions(
                    kinematic_enabled=False,
                    density=1000,
                    )
                min_pt, max_pt = compute_usd_dims(obj["Objpath"])
                min_x, min_y, min_z = min_pt[0], min_pt[1], min_pt[2]
                max_x, max_y, max_z = max_pt[0], max_pt[1], max_pt[2]

                object = SceneObject(
                    object_path=obj["Objpath"],
                    options=object_options,
                    # translation=obj["Translate"],    
                    translation=(obj["Translate"][0], obj["Translate"][1],obj["Translate"][2]+4),    

                    # rotation=(0.0, 0.0, 0.0, 1.0),
                    rotation=obj["RotateXYZW"],
                    scale=obj["Scale"],
                    object_type= "rigid",
                    id=idx,
                    object_dims = (min_x, max_x, min_y, max_y, min_z, max_z), # 除过scale后的尺寸
                )
                scene_list.append(object)

            elif obj["Id"] in scene_data["articulation_list"]:
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
                    object_path=obj["Objpath"],
                    options=object_options,
                    translation=obj["Translate"],    
                    rotation=obj["RotateXYZW"],
                    scale=obj["Scale"],
                    object_type= "articulation",
                    id=idx,
                    object_dims = (min_x, max_x, min_y, max_y, min_z, max_z),
                )
                scene_list.append(object)

            else:
                raise ValueError("Unknown object type")
            
                # if obj["Objname"] =="Multipersonsofa":
                #     print(object.id)

        return scene_list


