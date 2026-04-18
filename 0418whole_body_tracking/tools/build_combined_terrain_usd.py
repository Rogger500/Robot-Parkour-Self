import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obstacle_usd", required=True, help="Aligned obstacle USD path")
    parser.add_argument("--output_usd", required=True, help="Output combined terrain USD path")
    parser.add_argument("--ground_size", type=float, default=20.0, help="Ground plane size in meters")
    parser.add_argument("--ground_z", type=float, default=0.0, help="Ground plane z height")
    parser.add_argument("--ground_name", default="Ground")
    parser.add_argument("--obstacle_name", default="Obstacle")
    args = parser.parse_args()

    try:
        from pxr import Usd, UsdGeom, Gf, UsdPhysics
        try:
            from pxr import PhysxSchema  # optional in some Isaac / USD builds
        except Exception:
            PhysxSchema = None
    except Exception as exc:
        raise RuntimeError(
            "pxr is not available. Please run this script inside the Isaac Sim / Isaac Lab Python environment."
        ) from exc

    os.makedirs(os.path.dirname(os.path.abspath(args.output_usd)), exist_ok=True)

    stage = Usd.Stage.CreateNew(args.output_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/TerrainCombined")
    stage.SetDefaultPrim(root.GetPrim())

    # ------------------------------------------------------------------
    # Ground mesh: one large quad (2 triangles), static collision enabled
    # ------------------------------------------------------------------
    half = args.ground_size * 0.5
    z = args.ground_z
    ground_path = f"/TerrainCombined/{args.ground_name}"
    ground = UsdGeom.Mesh.Define(stage, ground_path)
    ground.CreatePointsAttr([
        Gf.Vec3f(-half, -half, z),
        Gf.Vec3f( half, -half, z),
        Gf.Vec3f( half,  half, z),
        Gf.Vec3f(-half,  half, z),
    ])
    ground.CreateFaceVertexCountsAttr([3, 3])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    ground.CreateSubdivisionSchemeAttr("none")
    ground.CreateDoubleSidedAttr(True)

    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    if PhysxSchema is not None:
        try:
            PhysxSchema.PhysxCollisionAPI.Apply(ground.GetPrim())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Obstacle reference under same root
    # ------------------------------------------------------------------
    obstacle_path = f"/TerrainCombined/{args.obstacle_name}"
    obstacle_xf = UsdGeom.Xform.Define(stage, obstacle_path)
    obstacle_xf.GetPrim().GetReferences().AddReference(os.path.abspath(args.obstacle_usd))

    stage.GetRootLayer().Save()

    print("=" * 80)
    print("Combined terrain USD generated")
    print(f"obstacle : {os.path.abspath(args.obstacle_usd)}")
    print(f"output   : {os.path.abspath(args.output_usd)}")
    print(f"ground   : size={args.ground_size} m, z={args.ground_z}")
    print("default  : /TerrainCombined")
    print("=" * 80)


if __name__ == "__main__":
    main()
