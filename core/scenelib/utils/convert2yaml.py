from isaaclab.app import AppLauncher

headless = True
app_launcher = AppLauncher({"headless": headless})
simulation_app = app_launcher.app

from omegaconf import OmegaConf
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
import os
import json
import argparse
from pxr import Usd, UsdGeom, Gf, UsdUtils,Sdf
from pxr import UsdPhysics,Vt
import numpy as np
from core.scenelib.utils.usd_utils import get_homeassets_list,euler_to_quat



def has_hinge(stage, prim):
    # 判断 Revolute Joint
    revolute = UsdPhysics.RevoluteJoint.Get(stage, prim.GetPath())
    if revolute:  # __bool__() 会判断 prim 是否有效且类型匹配
        return True

    prismatic = UsdPhysics.PrismaticJoint.Get(stage, prim.GetPath())
    if prismatic:
        return True
    return False



def flatten_file(usd_path: str, name: str) -> str:
    # 1. 打开 Stage 并确保加载所有 payload
    stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadAll)
    # 2. 准备输出路径
    root_layer = stage.GetRootLayer()
    orig_dir   = os.path.dirname(os.path.abspath(root_layer.realPath))
    out_path   = os.path.join(orig_dir, f"{name}.usda")
    print(f"{usd_path} -> {out_path}")
    # 3. 直接扁平化并导出。Export 内部使用了 Flatten，避免手动 CopySpec 导致的空弱引用崩溃
    stage.Export(out_path)  # 写出合成后的单层 Usd 文本文件
    return out_path

def del_fixed_joint(usd_path) -> bool:

    stage = Usd.Stage.Open(usd_path)
    flag = False
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsFixedJoint":
            stage.RemovePrim(prim.GetPath())
            stage.Save()
            flag = True
            print(f"  已删除 FixedJoint: {prim.GetPath()}")
    return flag


def cleanup_stage(src, dst = None,scale = None):
    stage = Usd.Stage.Open(src)

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            xformable = UsdGeom.Xformable(prim)

            # 1. 原始顶点
            pts = mesh.GetPointsAttr().Get()
            pts_np = np.array(pts)
            ones   = np.ones((pts_np.shape[0], 1))
            pts4   = np.hstack([pts_np, ones])

            # 2. 构建总变换矩阵
            mat = np.eye(4)
            for opName in xformable.GetXformOpOrderAttr().Get() or []:
                attr = prim.GetAttribute(opName)
                if not attr or not attr.IsValid():
                    continue
                val = attr.Get()

                if opName.endswith("translate"):
                    T = np.eye(4); T[:3, 3] = val; mat = mat @ T
                elif opName.endswith("rotateXYZ"):
                    rx, ry, rz = np.deg2rad(val)
                    Rx = np.array([[1,0,0,0],[0,np.cos(rx),-np.sin(rx),0],[0,np.sin(rx),np.cos(rx),0],[0,0,0,1]])
                    Ry = np.array([[np.cos(ry),0,np.sin(ry),0],[0,1,0,0],[-np.sin(ry),0,np.cos(ry),0],[0,0,0,1]])
                    Rz = np.array([[np.cos(rz),-np.sin(rz),0,0],[np.sin(rz),np.cos(rz),0,0],[0,0,1,0],[0,0,0,1]])
                    mat = mat @ (Rx @ Ry @ Rz)
                # elif opName.endswith("scale") and (scale is not None):
                #     S = np.diag([val[0]*scale[0], val[1]*scale[1], val[2]*scale[2], 1.0]); mat = mat @ S
                # elif (opName.endswith("scale")) and (scale is None):
                #     S = np.diag([val[0], val[1], val[2], 1.0]); mat = mat @ S
                # elif (not opName.endswith("scale")) and (scale is not None):
                #     S = np.diag([scale[0], scale[1], scale[2], 1.0]); mat = mat @ S
                elif opName.endswith("scale"):
                    S = np.diag([val[0], val[1], val[2], 1.0]); mat = mat @ S


            # 3. 应用并写回
            new_pts = (pts4 @ mat.T)[:, :3]
            # pts_list = []
            if scale is not None:
                for pt in new_pts:
                    pt[0] *= scale[0]
                    pt[1] *= scale[1]
                    pt[2] *= scale[2]
                # pts_list.append(pt)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new_pts), Usd.TimeCode.Default())

            # 4. 清空并重置 xformOp
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set((0,0,0))
            xformable.AddRotateXYZOp().Set((0,0,0))
            xformable.AddScaleOp().Set((1,1,1))

            if dst:
                stage.Export(dst)
                print(f"Export:{dst}")
            else:
                stage.Save()
                print(f"Export:{src}")


def to_flow(seq):
    cs = CommentedSeq(seq)
    cs.fa.set_flow_style()  # 打开 flow 模式，使列表输出为 [a, b, c] :contentReference[oaicite:0]{index=0}
    return cs

def preprocess_single_scene(single_scene_usd_path,scene_cfg):
    filter_name_list = ["Bathheater","Tuyere","Downlights","Tracklight","Chandelier","Clotheshanger",]
    # scene = {
    #     "filter_list": [],
    #     "rigidbody_list": [],
    #     "articulation_list": [],
    #     "objects": [],        
    # }
    objects = get_homeassets_list(single_scene_usd_path)
    for id,home_asset in enumerate(objects):
        if home_asset["assetPath"] is None:
            raise ValueError("assetPath is None")
        else:
            home_asset["assetPath"] = home_asset["assetPath"]
            # 创建一个新的usda导出flatten的文件
            name = home_asset["prim_category"]+"_flatten"
            old_usd_path = os.path.join(os.path.dirname(single_scene_usd_path),home_asset["assetPath"])
            obj_path = os.path.join(os.path.dirname(home_asset["assetPath"]),f"{name}.usda")
            new_usd_path = os.path.join(os.path.dirname(single_scene_usd_path),obj_path)
            flatten_file(old_usd_path, name)           

        if home_asset["translate"] is None:
            obj_translate = (0.0, 0.0, 0.0)
        else:
            obj_translate = (float(home_asset["translate"][0]), float(home_asset["translate"][1]), float(home_asset["translate"][2]))
        if home_asset["rotateXYZ"] is None:
            obj_rotateXYZW = (0.0, 0.0, 0.0, 1.0)
        else:
            obj_rotateXYZW = euler_to_quat(home_asset["rotateXYZ"])
        if home_asset["scale"] is None:
            obj_scale = (1.0, 1.0, 1.0)
            cleanup_stage(src=new_usd_path)
        else:
            s = home_asset["scale"]
            obj_scale = tuple(abs(float(v)) for v in s)
            sign_scale = tuple(np.sign(s).astype(float))
            cleanup_stage(src=new_usd_path,scale=sign_scale)

            # obj_scale = tuple(float(v) for v in s)
            # cleanup_stage(src=new_usd_path)


        obj = {
            "obj_id": id,
            "obj_path": obj_path,
            "obj_name": home_asset["prim_category"],
            "obj_translate": to_flow(obj_translate),
            "obj_rotateXYZW": to_flow(obj_rotateXYZW),
            "obj_scale": to_flow(obj_scale),
        }
        scene_cfg["objects"].append(obj)        



    for id, obj in enumerate(scene_cfg["objects"]):

        new_usd_path = os.path.join(os.path.dirname(single_scene_usd_path),obj["obj_path"])

        flag = False
        stage = Usd.Stage.Open(new_usd_path,load=Usd.Stage.LoadAll)
        for prim in stage.Traverse():
            if flag:
                break
            if has_hinge(stage, prim):
                scene_cfg["articulation_list"].append(obj["obj_id"])
                flag = True
        
        if not flag:
            for prim in stage.Traverse():
                if flag:
                    break
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    scene_cfg["rigidbody_list"].append(obj["obj_id"])
                    flag = True       
            if not flag:
                raise ValueError(f"ID:{obj['obj_id']} Objname:{obj['obj_name']} 该物体没有rigidbody和articulation")
            
        if obj["obj_name"] in filter_name_list:
            scene_cfg["filter_list"].append(obj["obj_id"])

    for id in scene_cfg["rigidbody_list"]:
        new_usd_path = os.path.join(os.path.dirname(single_scene_usd_path),obj["obj_path"])
        for prim in stage.Traverse():
            del_fixed_joint(new_usd_path)
    

    # json_path = os.path.join(os.path.dirname(single_scene_usd_path), "config.json")
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(scene, f, ensure_ascii=False, indent=4)
    # print(f"导出完成：{json_path}")
    





def export_scene_yaml(usd_path,output_dir):
    conf_dict = {}
    if not os.path.isdir(usd_path):
        raise ValueError("输入路径不是目录")
    if os.path.basename(usd_path) == "Home":
        for name in os.listdir(usd_path): # 遍历所有的Home下的目录（bedroom、livingroom）
            dir_path = os.path.join(usd_path, name)

            if os.path.isdir(dir_path):
                conf_dict = {}
                # conf_dict["defaults"] = [{"base": None}]
                conf_dict["defaults"] = ["base"]
                conf_dict["scenes_abspath"] = os.path.abspath(dir_path)
                conf_dict["scenes_name"] = name.capitalize()
                conf_dict["scenes"] = []

                for scene_name in os.listdir(dir_path): # 遍历所有的scene目录(001、002)
                    scene_path = os.path.join(dir_path, scene_name)
                    if os.path.isdir(scene_path):
                        for usd_file in os.listdir(scene_path): # 遍历每个scene目录下的所有文件
                            if os.path.basename(usd_file) == "Instance.usda":
                                scene_cfg = {}
                                scene_cfg["scene_id"] = scene_name
                                scene_cfg["scene_path"] = os.path.relpath(scene_path,dir_path)
                                scene_cfg["filter_list"] = []
                                scene_cfg["rigidbody_list"] = []
                                scene_cfg["articulation_list"] = []
                                scene_cfg["objects"] = []
                                
                                usd_file_path = os.path.join(scene_path, usd_file)

                                preprocess_single_scene(usd_file_path,scene_cfg)

                                scene_cfg["filter_list"] = to_flow(scene_cfg["filter_list"])
                                scene_cfg["rigidbody_list"] = to_flow(scene_cfg["rigidbody_list"])
                                scene_cfg["articulation_list"] = to_flow(scene_cfg["articulation_list"])
                                conf_dict["scenes"].append(scene_cfg)


                # 确保输出目录存在
                os.makedirs(output_dir, exist_ok=True)
                yaml_path = os.path.join(output_dir, f"{name.lower()}.yaml")
                yaml = YAML()
                yaml.indent(mapping=2, sequence=4, offset=2)

                # 3. 写文件——先写 Hydra 的 package 注释，然后 dump
                with open(yaml_path, "w", encoding="utf-8") as f:
                    f.write("# @package _global_\n\n")  # Hydra 全局包指令 :contentReference[oaicite:2]{index=2}
                    yaml.dump(conf_dict, f)
                print(f"已导出 {yaml_path}")



    elif os.path.basename(usd_path) == "Object":
        print("TODO:coming soon ...")
        raise ValueError("Object preprocess will coming soon ...")      
    else:
        raise ValueError("暂不支持输入非Home目录和Object目录")
                                         







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 USD 场景物体相对变换导出为指定格式 YAML")
    parser.add_argument("usd_dir",   help="输入一类的 USD 目录路径",nargs="?",default="/home/luohy/MyRepository/MyDataSets/Data/Home")

    args = parser.parse_args()

    export_scene_yaml(
        usd_path=args.usd_dir,
        output_dir="/home/luohy/vs-code-projects/ISAACLAB/HSIProject/core/config/scene"
    )
