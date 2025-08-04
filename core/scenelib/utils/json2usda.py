from isaaclab.app import AppLauncher

headless = True
app_launcher = AppLauncher({"headless": headless})
simulation_app = app_launcher.app


from pxr import Usd, UsdGeom, Gf, UsdUtils,Sdf
from pxr import UsdPhysics
from core.scenelib.utils.usd_utils import get_homeassets_list,euler_to_quat
import json
import os

if __name__ == "__main__":
    scene_path = "/home/luohy/MyRepository/MyDataSets/Data/Home/Livingroom/002/config.json"
    dir_path = os.path.dirname(scene_path)
    dst = "/home/luohy/MyRepository/MyDataSets/Data/Home/Livingroom/002/test.usda"
    with open(scene_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)

    
    # create new stage
    stage_new = Usd.Stage.CreateNew(dst)
    UsdGeom.SetStageUpAxis(stage_new, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage_new, 1.0)
    root = stage_new.DefinePrim("/Root", "Xform")
    stage_new.SetDefaultPrim(root)
    stage_new.DefinePrim("/Root/Meshes", "Scope")

    # iterate through to create new actors
    for index, actor in enumerate(scene_data["objects"]):
        stage_new.DefinePrim(f"/Root/Meshes/obj", "Scope")

        new_actor_prim = f"/Root/Meshes/obj/obj_{index:04d}"
        prim = stage_new.DefinePrim(new_actor_prim, "Xform")
        xform = UsdGeom.Xform(prim)
        # set asset reference or load new asset?

        xform.GetPrim().GetReferences().AddReference(assetPath=actor["Objpath"])  # or use reference

        # 平移
        tx, ty, tz = actor["Translate"]
        xform.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))

        # 旋转（四元数格式 [x, y, z, w]）
        qx, qy, qz, qw = actor["RotateXYZW"]
        xform.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
        # 缩放
        sx, sy, sz = actor["Scale"]
        xform.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

    stage_new.GetRootLayer().Save()
