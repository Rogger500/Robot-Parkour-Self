import argparse
import json
import os


def _quat_wxyz_from_json(payload):
    quat = payload.get("quat_wxyz", None)
    if quat is not None and len(quat) == 4:
        return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
    yaw_deg = float(payload.get("yaw_deg", 0.0))
    import math
    half = math.radians(yaw_deg) * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def main():
    parser = argparse.ArgumentParser(description="Build a single terrain USD containing ground + transformed obstacle reference.")
    parser.add_argument("--source_obstacle_usd", required=True, help="Original obstacle USD that already renders correctly when loaded directly")
    parser.add_argument("--terrain_pose_json", required=True, help="JSON with translation_xyz / quat_wxyz / yaw_deg")
    parser.add_argument("--output_usd", required=True, help="Output combined terrain USD")
    parser.add_argument("--ground_size", type=float, default=20.0)
    parser.add_argument("--ground_z", type=float, default=0.0)
    parser.add_argument("--root_prim", default="/TerrainCombined")
    args = parser.parse_args()

    from pxr import Usd, UsdGeom, Gf, UsdPhysics

    with open(args.terrain_pose_json, "r", encoding="utf-8") as f:
        pose = json.load(f)

    translation = pose.get("translation_xyz", pose.get("translation", [0.0, 0.0, 0.0]))
    translation = [float(translation[0]), float(translation[1]), float(translation[2])]
    quat = _quat_wxyz_from_json(pose)

    out_abs = os.path.abspath(args.output_usd)
    src_abs = os.path.abspath(args.source_obstacle_usd)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)

    stage = Usd.Stage.CreateNew(out_abs)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, args.root_prim)
    stage.SetDefaultPrim(root.GetPrim())

    half = 0.5 * float(args.ground_size)
    z = float(args.ground_z)
    ground = UsdGeom.Mesh.Define(stage, f"{args.root_prim}/Ground")
    ground.CreatePointsAttr([
        Gf.Vec3f(-half, -half, z),
        Gf.Vec3f(+half, -half, z),
        Gf.Vec3f(+half, +half, z),
        Gf.Vec3f(-half, +half, z),
    ])
    ground.CreateFaceVertexCountsAttr([3, 3])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    ground.CreateSubdivisionSchemeAttr("none")
    ground.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    wrapper = UsdGeom.Xform.Define(stage, f"{args.root_prim}/ObstacleWrapper")
    wxf = UsdGeom.Xformable(wrapper.GetPrim())
    wxf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(translation[0], translation[1], translation[2])
    )
    wxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(quat[0], Gf.Vec3d(quat[1], quat[2], quat[3]))
    )

    asset = UsdGeom.Xform.Define(stage, f"{args.root_prim}/ObstacleWrapper/ObstacleAsset")
    asset.GetPrim().GetReferences().AddReference(src_abs)

    stage.GetRootLayer().Save()

    print("=" * 80)
    print("Combined terrain USD generated")
    print(f"source obstacle : {src_abs}")
    print(f"terrain pose    : {os.path.abspath(args.terrain_pose_json)}")
    print(f"output          : {out_abs}")
    print(f"translation_xyz : {translation}")
    print(f"quat_wxyz       : {quat}")
    print(f"default prim    : {args.root_prim}")
    print("=" * 80)


if __name__ == "__main__":
    main()
