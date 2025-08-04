from isaaclab.app import AppLauncher

headless = True
app_launcher = AppLauncher({"headless": headless})
simulation_app = app_launcher.app

import os
import json
import argparse
from pxr import Usd, UsdGeom, Gf, UsdUtils,Sdf
from pxr import UsdPhysics,Vt
import numpy as np
from core.scenelib.utils.usd_utils import get_homeassets_list,euler_to_quat

def check_rigidbody(stage:Usd.Stage):
    # 1. 先遍历一遍，记录三个 API 是否出现过
    flag_rigid     = False
    flag_collision = False
    flag_mass      = False

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RigidBodyAPI):
            flag_rigid = True
        if prim.IsA(UsdPhysics.CollisionAPI):
            flag_collision = True
        if prim.IsA(UsdPhysics.MassAPI):
            flag_mass = True
        # 如果三种都找到了，就没必要再继续遍历了
        if flag_rigid and flag_collision and flag_mass:
            break

    # 2. 根据结果分三种情况处理
    # 2.1 如果完全没找到任何 API，就把所有 API 都加到最外层 root 上
    if not (flag_rigid or flag_collision or flag_mass):
        root = stage.GetDefaultPrim()
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.CollisionAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root)

    # 2.2 如果只找到了部分 API，就把缺少的那些加到已有 API 的同一个 prim 上
    elif not (flag_rigid and flag_collision and flag_mass):
        for prim in stage.Traverse():
            # 只对已经应用过至少一个 Physics API 的 prim 做补充
            has_any = (
                prim.IsA(UsdPhysics.RigidBodyAPI) or
                prim.IsA(UsdPhysics.CollisionAPI) or
                prim.IsA(UsdPhysics.MassAPI)
            )
            if not has_any:
                continue

            # 缺哪个补哪个
            if not prim.IsA(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(prim)
            if not prim.IsA(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if not prim.IsA(UsdPhysics.MassAPI):
                UsdPhysics.MassAPI.Apply(prim)

    # 2.3 如果三种 API 都已经存在，则不做任何操作
    else:
        pass  # 全部就绪，跳过    

def preprocess_mesh(stage:Usd.Stage):
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
                elif opName.endswith("scale"):
                    S = np.diag([val[0], val[1], val[2], 1.0]); mat = mat @ S


            # 3. 应用并写回
            new_pts = (pts4 @ mat.T)[:, :3]
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new_pts), Usd.TimeCode.Default())

            # 4. 清空并重置 xformOp
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set((0,0,0))
            xformable.AddRotateXYZOp().Set((0,0,0))
            xformable.AddScaleOp().Set((1,1,1))

def preprocess_rigidbody(usd_path,output_path=None):
    """
    支持为rigidbody添加碰撞体、 添加刚体属性，以及实现将物体处理到相对于其自身坐标系下的中心点
    """
    if not (usd_path.endswith('.usda') or usd_path.endswith('.usd')):
        ValueError('usd_path must be a .usda or .usd file')



    stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadAll)
    #检查rigidbody的API是否完整并补全
    check_rigidbody(stage)

    # 处理usda自身坐标系的中心点问题
    preprocess_mesh(stage)

    if output_path is not None:
        if output_path.endswith('.usda') or output_path.endswith('.usd'):
            stage.Export(output_path)
            print('')
        elif os.path.isdir(output_path):
            stage.Export(os.path.join(output_path,os.path.basename(usd_path)))
        else:
            ValueError('output_path must be a .usda or .usd file or a directory')
    else:
        output_path = usd_path
        stage.Save()

    print(f"preprocess_rigidbody done:{usd_path} -> {output_path}")
    

if __name__ == '__main__':
    object_dir = "/home/luohy/MyRepository/MyDataSets/Data/Object"

    # 遍历所有子目录
    for dirpath, dirnames, filenames in os.walk(object_dir):
        # 计算当前目录相对于 object_dir 的相对路径
        rel = os.path.relpath(dirpath, object_dir)
        # 跳过根目录自己
        if rel == '.':
            continue

        # 按文件系统分隔符切分，相对路径只有两段时才是二级目录
        parts = rel.split(os.sep)
        if len(parts) != 2:
            continue

        # 到了二级目录，处理该目录下的 USD/usda 文件
        for filename in filenames:
            if (filename.endswith('.usda') or filename.endswith('.usd')) \
               and not any(filename.endswith(suf) for suf in (
                   '_payload.usda', '_geo.usd', '_look.usda', 'Instance.usda'
               )):
                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(dirpath, 'Instance.usda')
                # print(f"Processing {input_path} → {output_path}")
                preprocess_rigidbody(input_path, output_path)

    print("All done")