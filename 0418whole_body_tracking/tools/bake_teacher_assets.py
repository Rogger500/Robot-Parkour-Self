import argparse
import json
import pathlib
from typing import Tuple

import numpy as np


def quat_wxyz_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    yaw_rad = np.deg2rad(float(yaw_deg))
    half = 0.5 * yaw_rad
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def load_manifest_entry(manifest_path: pathlib.Path, motion_file: pathlib.Path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    trajectories = payload.get("trajectories", [])

    motion_name = motion_file.name
    motion_path_str = str(motion_file)

    for item in trajectories:
        traj_path = item.get("trajectory_path", "")
        traj_name = item.get("trajectory_name", "")

        if pathlib.Path(traj_path).name == motion_name:
            return item
        if traj_name and motion_name.startswith(traj_name):
            return item
        if traj_path and motion_path_str.endswith(traj_path):
            return item

    raise RuntimeError(f"Could not find matching trajectory entry in manifest for {motion_name}")


def compute_replay_root_offset(manifest_entry, body_pos_w_np: np.ndarray, root_body_idx: int) -> np.ndarray:
    if "skill_anchor" not in manifest_entry or "segments" not in manifest_entry:
        raise RuntimeError("Manifest entry missing 'skill_anchor' or 'segments'.")

    target_root = np.asarray(manifest_entry["skill_anchor"]["root_translation"], dtype=np.float64)

    skill_output_start = None
    for seg in manifest_entry["segments"]:
        if seg.get("mode") == "skill_execution":
            skill_output_start = int(seg["output_start_frame"])
            break

    if skill_output_start is None:
        raise RuntimeError("No 'skill_execution' segment found in manifest.")

    if skill_output_start >= len(body_pos_w_np):
        raise RuntimeError("skill_execution.output_start_frame exceeds trajectory length.")

    observed_root = np.asarray(body_pos_w_np[skill_output_start, root_body_idx], dtype=np.float64)
    offset = target_root - observed_root
    return offset


def extract_terrain_pose(manifest_entry) -> Tuple[np.ndarray, float, np.ndarray]:
    terrain_pose = manifest_entry.get("terrain_world_pose", None)
    if terrain_pose is None:
        raise RuntimeError("Manifest entry missing 'terrain_world_pose'.")

    translation = np.asarray(terrain_pose["translation"], dtype=np.float64)
    yaw_deg = float(terrain_pose["yaw_deg"])
    quat_wxyz = quat_wxyz_from_yaw_deg(yaw_deg)
    return translation, yaw_deg, quat_wxyz


def bake_motion_npz(
    motion_path: pathlib.Path,
    manifest_entry,
    output_motion_path: pathlib.Path,
    root_body_idx: int,
    z_offset: float,
):
    data = np.load(motion_path, allow_pickle=True)

    out_dict = {}
    for k in data.files:
        out_dict[k] = data[k]

    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64).copy()

    replay_root_offset = compute_replay_root_offset(manifest_entry, body_pos_w, root_body_idx)
    total_offset = replay_root_offset.copy()
    total_offset[2] += float(z_offset)

    # 整条轨迹所有 body 的世界坐标统一平移
    body_pos_w += total_offset.reshape(1, 1, 3)
    out_dict["body_pos_w"] = body_pos_w.astype(np.float32)

    output_motion_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_motion_path, **out_dict)

    print("=" * 80)
    print("[MOTION BAKE DONE]")
    print(f"input motion      : {motion_path}")
    print(f"output motion     : {output_motion_path}")
    print(f"root_body_idx     : {root_body_idx}")
    print(f"replay_root_offset: {replay_root_offset.tolist()}")
    print(f"z_offset          : {z_offset}")
    print(f"total_offset      : {total_offset.tolist()}")
    print("=" * 80)

    return replay_root_offset, total_offset


def save_terrain_pose_json(manifest_entry, output_json_path: pathlib.Path):
    translation, yaw_deg, quat_wxyz = extract_terrain_pose(manifest_entry)

    payload = {
        "translation_xyz": translation.tolist(),
        "yaw_deg": yaw_deg,
        "quat_wxyz": quat_wxyz.tolist(),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("[TERRAIN POSE EXTRACTED]")
    print(f"output json       : {output_json_path}")
    print(f"translation_xyz   : {translation.tolist()}")
    print(f"yaw_deg           : {yaw_deg}")
    print(f"quat_wxyz         : {quat_wxyz.tolist()}")
    print("=" * 80)

    return translation, yaw_deg, quat_wxyz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--manifest_file", type=str, required=True)
    parser.add_argument("--output_motion_file", type=str, required=True)
    parser.add_argument("--output_terrain_pose_json", type=str, required=True)
    parser.add_argument("--root_body_idx", type=int, default=0)
    parser.add_argument("--z_offset", type=float, default=0.188)
    args = parser.parse_args()

    motion_path = pathlib.Path(args.motion_file).resolve()
    manifest_path = pathlib.Path(args.manifest_file).resolve()
    output_motion_path = pathlib.Path(args.output_motion_file).resolve()
    output_terrain_pose_json = pathlib.Path(args.output_terrain_pose_json).resolve()

    if not motion_path.exists():
        raise FileNotFoundError(motion_path)

    manifest_entry = load_manifest_entry(manifest_path, motion_path)

    bake_motion_npz(
        motion_path=motion_path,
        manifest_entry=manifest_entry,
        output_motion_path=output_motion_path,
        root_body_idx=args.root_body_idx,
        z_offset=args.z_offset,
    )

    save_terrain_pose_json(
        manifest_entry=manifest_entry,
        output_json_path=output_terrain_pose_json,
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()