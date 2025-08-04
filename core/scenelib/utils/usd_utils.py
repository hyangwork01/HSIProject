import typing

# import carb
# import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade



def compute_usd_dims(object_path: str) -> typing.Tuple[Gf.Vec3d, Gf.Vec3d]:
    """
    计算给定Object的边界框，给个USD路径
    Compute Bounding Box using omni.usd.UsdContext.compute_path_world_bounding_box
    See https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd/omni.usd.UsdContext.html#omni.usd.UsdContext.compute_path_world_bounding_box

    Args:
        prim_path: A prim path to compute the bounding box.
    Returns: 
        A range (i.e. bounding box) as a minimum point and maximum point.
    """
    if not object_path.lower().endswith((".usd", ".usda", ".usdc")):
        raise ValueError(f"错误：文件路径 '{object_path}' 不是以 USD 格式结尾。")
    
    stage = Usd.Stage.Open(object_path)
    if not stage:
        raise RuntimeError(f"无法打开 USD 文件：{object_path}")
    for prim in stage.Traverse():
        print(prim.GetPath())
    prim = stage.GetDefaultPrim()
    # prim = stage.GetPrimAtPath()
    if not prim or not prim.IsValid():
        prim = stage.GetPseudoRoot()
    if not prim or not prim.IsValid():
        raise ValueError(f"找不到 Prim: {'<DefaultPrim/PseudoRoot>'}")
    min_pt, max_pt =  None, None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                            [UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    min_pt, max_pt = bbox.GetMin(), bbox.GetMax()

    # for child in prim.GetChildren():
    #     name = child.GetName().lower()
    #     if "instance" in name:
    #         xformable = UsdGeom.Xformable(child)
    #         cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    #         bbox = cache.ComputeWorldBound(child).ComputeAlignedBox()
    #         min_pt, max_pt = bbox.GetMin(), bbox.GetMax()


    if min_pt is None or max_pt is None:
        raise ValueError(f"不存在Instance这个prim/找不到边界框: {prim.GetPath()}")
        
    return (min_pt, max_pt)

# TODO:后续可以添加一个参数用于屏蔽不想使用的assets
def get_homeassets_list(src):
    stage = Usd.Stage.Open(src)

    actor_to_add = []

    # traverse all scene to get all assets
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xform):
            # find category by path
            prim_category = prim.GetPath().GetParentPath().name

            # find reference asset path
            assetPath = None
            stack = prim.GetPrimStack()


            for prim_spec in stack:

                # for reference in prim_spec.referenceList.GetAppliedItems():
                #     assetPath = reference.assetPath
                for ref_item in prim_spec.referenceList.GetAddedOrExplicitItems():
                    assetPath = ref_item.assetPath
            if prim_category == "/":
                continue

            if assetPath is None:
                # print("No assetPath found for prim: ", prim.GetPath())
                continue
            # find existing actor transforms
            translate = prim.GetAttribute('xformOp:translate').Get()
            rotateXYZ = prim.GetAttribute('xformOp:rotateXYZ').Get()
            scale = prim.GetAttribute('xformOp:scale').Get()

            actor_to_add.append({
                "prim_category": prim_category,
                "assetPath": assetPath,
                "translate": translate,
                "rotateXYZ": rotateXYZ,
                "scale": scale,
            })

    return actor_to_add


def euler_to_quat(euler: Gf.Vec3f, order: str = "XYZ") -> typing.Tuple[float, float, float, float]:
    # 1. 将 Gf.Vec3f 组件拆解为 Python float
    ex, ey, ez = float(euler[0]), float(euler[1]), float(euler[2])
    # 2. 构造 Vec3d 轴向量 + 角度（度数）
    rot_x = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), ex)
    rot_y = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), ey)
    rot_z = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), ez)
    # 3. 按顺序串联（R = Rz * Ry * Rx）
    r = rot_z * rot_y * rot_x
    # 4. 返回双精度四元数
    quat = r.GetQuat()
    imag = quat.GetImaginary()   # Gf.Vec3d
    x, y, z = float(imag[0]), float(imag[1]), float(imag[2])
    w = float(quat.GetReal())
    return (x,y,z,w)


# def 

if __name__ == "__main__":
    usd_path = "/home/luohy/Downloads/empty_room/1.usd"

    try:
        min_pt, max_pt = compute_usd_dims(usd_path)
        print(f"Bounding box for '{usd_path}':")
        print(f"  Min: ({min_pt[0]:.6f}, {min_pt[1]:.6f}, {min_pt[2]:.6f})")
        print(f"  Max: ({max_pt[0]:.6f}, {max_pt[1]:.6f}, {max_pt[2]:.6f})")
        print(f"  Size: ({max_pt[0] - min_pt[0]:.6f}, {max_pt[1] - min_pt[1]:.6f}, {max_pt[2] - min_pt[2]:.6f})")
    
    except Exception as e:
        print(f"Error computing bounding box: {e}")    
