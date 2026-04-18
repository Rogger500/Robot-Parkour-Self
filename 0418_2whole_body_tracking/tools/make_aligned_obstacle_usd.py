import argparse
import json
import pathlib

from pxr import Usd, UsdGeom, Gf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_obstacle_usd", type=str, required=True)
    parser.add_argument("--terrain_pose_json", type=str, required=True)
    parser.add_argument("--output_obstacle_usd", type=str, required=True)
    args = parser.parse_args()

    source_usd = pathlib.Path(args.source_obstacle_usd).resolve()
    pose_json = pathlib.Path(args.terrain_pose_json).resolve()
    output_usd = pathlib.Path(args.output_obstacle_usd).resolve()

    if not source_usd.exists():
        raise FileNotFoundError(source_usd)
    if not pose_json.exists():
        raise FileNotFoundError(pose_json)

    pose = json.loads(pose_json.read_text(encoding="utf-8"))
    translation = pose["translation_xyz"]
    quat_wxyz = pose["quat_wxyz"]

    output_usd.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_usd))

    # 外层 wrapper：只负责位姿
    wrapper = UsdGeom.Xform.Define(stage, "/ObstacleWrapper")
    wrapper_prim = wrapper.GetPrim()
    wrapper_xf = UsdGeom.Xformable(wrapper_prim)

    wrapper_xf.AddTranslateOp().Set(
        Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))
    )
    wrapper_xf.AddOrientOp().Set(
        Gf.Quatf(
            float(quat_wxyz[0]),
            Gf.Vec3f(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
        )
    )

    # 子 prim：只负责引用原始 obstacle USD
    asset = UsdGeom.Xform.Define(stage, "/ObstacleWrapper/ObstacleAsset")
    asset_prim = asset.GetPrim()
    asset_prim.GetReferences().AddReference(str(source_usd))

    stage.SetDefaultPrim(wrapper_prim)
    stage.Save()

    print("=" * 80)
    print("Aligned obstacle USD generated")
    print(f"source : {source_usd}")
    print(f"pose   : {pose_json}")
    print(f"output : {output_usd}")
    print("=" * 80)


if __name__ == "__main__":
    main()