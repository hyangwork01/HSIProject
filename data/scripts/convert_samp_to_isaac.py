# This code is adapted from https://github.com/zhengyiluo/phc/ and generalized to work with any humanoid.
# https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/convert_amass_isaac.py

import os
import uuid
from pathlib import Path
from typing import Optional

import ipdb
import yaml
import numpy as np
import torch
import typer
from scipy.spatial.transform import Rotation as sRot
import pickle
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from tqdm import tqdm

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
import time
from datetime import timedelta

TMP_SMPL_DIR = "/tmp/smpl"




def main(
    root_dir: Path,
    robot_type: str = None,
    humanoid_type: str = "smpl", # "smpl", "smplx", "smplh",使用的数据类型
    force_remake: bool = False, # 是否强制重新生成
    # force_neutral_body: bool = True,    # 是否强制生成中性身体
    generate_flipped: bool = False, #是否生成镜像数据
    not_upright_start: bool = False,  # By default, let's start upright (for consistency across all models).
    humanoid_mjcf_path: Optional[str] = None,
    force_retarget: bool = False,
    output_dir: Path = None,
):
    if output_dir is None:
        output_dir = root_dir.parent / f"{root_dir.name}_isaac"
    output_dir.mkdir(exist_ok=True, parents=True)

    if robot_type is None:
        robot_type = humanoid_type
    elif robot_type in ["h1","g1"]:
        assert (
            force_retarget
        ), f"Data is either SMPL or SMPL-X. The {robot_type} robot must use the retargeting pipeline."
    
    assert humanoid_type in [
        "smpl",
        "smplx",
        "smplh",
    ], "Humanoid type must be one of smpl, smplx, smplh"

    
    upright_start = not not_upright_start

    if humanoid_mjcf_path is not None:
        skeleton_tree = SkeletonTree.from_mjcf(humanoid_mjcf_path)
    else:
        skeleton_tree = None
    

    append_name = robot_type
    if force_retarget:
        append_name += "_retargeted"
    if humanoid_type == "smpl":
        mujoco_joint_names = SMPL_MUJOCO_NAMES
        joint_names = SMPL_BONE_ORDER_NAMES
    elif humanoid_type == "smplx" or humanoid_type == "smplh":
        mujoco_joint_names = SMPLH_MUJOCO_NAMES
        joint_names = SMPLH_BONE_ORDER_NAMES
    else:
        raise NotImplementedError
    
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": upright_start,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": humanoid_type,
        "sim": "isaacgym",
    }

    smpl_local_robot = SMPL_Robot(
        robot_cfg,
        data_dir="data/smpl",
    )

    if generate_flipped:
        left_to_right_index = []
        for idx, entry in enumerate(mujoco_joint_names):
            # swap text "R_" and "L_"
            if entry.startswith("R_"):
                left_to_right_index.append(mujoco_joint_names.index("L_" + entry[2:]))
            elif entry.startswith("L_"):
                left_to_right_index.append(mujoco_joint_names.index("R_" + entry[2:]))
            else:
                left_to_right_index.append(idx)

    all_files_in_folder = [
        f
        for f in Path(root_dir).glob("**/*.[np][pk][lz]")
        if (f.name != "shape.npz" and "stagei.npz" not in f.name)
    ]
    save_dir = output_dir / f"{root_dir.name}_{append_name}"

    if not force_remake:
        # Only count files that don't already have outputs
        files_to_process = [
            f
            for f in all_files_in_folder
            if not (
                save_dir
                / f.relative_to(root_dir).parent
                / f.name.replace(".npz", ".npy")
                .replace(".pkl", ".npy")
                .replace("-", "_")
                .replace(" ", "_")
                .replace("(", "_")
                .replace(")", "_")
            ).exists()
        ]
    else:
        files_to_process = all_files_in_folder
    print(
        f"Processing {len(files_to_process)}/{len(all_files_in_folder)} files in {root_dir}"
    )        

    for file_path in tqdm(files_to_process):
        print(f"Processing file {file_path}")

        # if not force_remake and file_path.exists():
        #     continue
        relative_path_dir = file_path.relative_to(root_dir).parent
        outpath = (
            save_dir
            / relative_path_dir
            / file_path.name.replace(".npz", ".npy")
            .replace(".pkl", ".npy")
            .replace("-", "_")
            .replace(" ", "_")
            .replace("(", "_")
            .replace(")", "_")
        )
        os.makedirs(outpath.parent, exist_ok=True)

        if file_path.suffix == ".pkl":
            try: 
                with open(file_path, 'rb') as f:
                    motion_data = pickle.load(f, encoding="latin1")
                # betas = motion_data['shape_est_betas'][:16]
                samp_pose = motion_data["pose_est_fullposes"]
                samp_trans = motion_data["pose_est_trans"]
                mocap_fr = motion_data["mocap_framerate"]


            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                raise e

        else:
            print(f"Skipping {file_path} as it is not a valid file")
            continue
        
        pose_aa = torch.tensor(samp_pose)
        samp_trans = torch.tensor(samp_trans)

        # 默认使用 neutral_body中性的smpl body
        betas = torch.zeros((1, 16)).float()
        gender = "neutral"

        motion_data = {
            "pose_aa": pose_aa.numpy(),
            "trans": samp_trans.numpy(),
            "beta": betas.numpy(),
            "gender": gender,
        }

        smpl_2_mujoco = [
            joint_names.index(q) for q in mujoco_joint_names if q in joint_names
        ]
        batch_size = motion_data["pose_aa"].shape[0]


        if humanoid_type == "smpl":
            pose_aa = np.concatenate(
                [motion_data["pose_aa"][:, :66], np.zeros((batch_size, 6))],
                axis=1,
            )  # TODO: need to extract correct handle rotations instead of zero
            pose_aa_mj = pose_aa.reshape(batch_size, 24, 3)[:, smpl_2_mujoco]
            pose_quat = (
                sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
                .as_quat()
                .reshape(batch_size, 24, 4)
            )
        else:
            pose_aa = np.concatenate(
                [
                    motion_data["pose_aa"][:, :66],
                    motion_data["pose_aa"][:, 75:],
                ],
                axis=-1,
            )
            pose_aa_mj = pose_aa.reshape(batch_size, 52, 3)[:, smpl_2_mujoco]
            pose_quat = (
                sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
                .as_quat()
                .reshape(batch_size, 52, 4)
            )

        gender_number = [0]

        if skeleton_tree is None:
            uuid_str = uuid.uuid4()
            smpl_local_robot.load_from_skeleton(
                betas=betas, gender=gender_number, objs_info=None
            )
            smpl_local_robot.write_xml(
                f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
            )
            skeleton_tree = SkeletonTree.from_mjcf(
                f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
            )

        root_trans_offset = (
            torch.from_numpy(motion_data["trans"])
            + skeleton_tree.local_translation[0]
        )

        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True,
        )

        if generate_flipped:
            formats = ["regular", "flipped"]
        else:
            formats = ["regular"]

        for format in formats:
            if robot_cfg["upright_start"]:
                B = pose_aa.shape[0]
                pose_quat_global = (
                    (
                        sRot.from_quat(
                            sk_state.global_rotation.reshape(-1, 4).numpy()
                        )
                        * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                    )
                    .as_quat()
                    .reshape(B, -1, 4)
                )
            else:
                pose_quat_global = sk_state.global_rotation.numpy()

            trans = root_trans_offset.clone()
            if format == "flipped":
                pose_quat_global = pose_quat_global[:, left_to_right_index]
                pose_quat_global[..., 0] *= -1
                pose_quat_global[..., 2] *= -1
                trans[..., 1] *= -1

            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,
                torch.from_numpy(pose_quat_global),
                trans,
                is_local=False,
            )

            new_sk_motion = SkeletonMotion.from_skeleton_state(
                new_sk_state, fps=mocap_fr
            )

            if force_retarget:
                from data.scripts.retargeting.mink_retarget import (
                    retarget_motion,
                )

                print("Force retargeting motion using mink retargeter...")
                # Convert to 30 fps to speedup Mink retargeting
                skip = int(mocap_fr // 30)
                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,
                    torch.from_numpy(pose_quat_global[::skip]),
                    trans[::skip],
                    is_local=False,
                )
                new_sk_motion = SkeletonMotion.from_skeleton_state(
                    new_sk_state, fps=30
                )

                if robot_type in ["smpl", "smplx", "smplh"]:
                    robot_type = f"{robot_type}_humanoid"
                new_sk_motion = retarget_motion(
                    motion=new_sk_motion, robot_type=robot_type, render=False
                )

            if format == "flipped":
                outpath = outpath.with_name(
                    outpath.stem + "_flipped" + outpath.suffix
                )
            print(f"Saving to {outpath}")
            if robot_type in ["h1", "g1"]:
                torch.save(new_sk_motion, str(outpath))
            else:
                new_sk_motion.to_file(str(outpath))





if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)
