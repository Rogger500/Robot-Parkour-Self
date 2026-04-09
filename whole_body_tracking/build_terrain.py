from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf, UsdPhysics

print("正在通过‘几何缝合’技术熔铸单体跑酷大陆...")

stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

root_prim = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(root_prim)

# 🌟 唯一的一个 Mesh，这样 RayCaster 才不会报错
mesh = UsdGeom.Mesh.Define(stage, "/World/terrain")

# 坐标参数：BX = -1.32 (动作起跳点)
BX, BW, BD, BH = -1.4, 1.0, 3.0, 0.1

# 1. 顶点汇总 (0-3 是大地, 4-11 是箱子)
vertices = [
    [-50.0, -50.0, 0.0], [50.0, -50.0, 0.0], [50.0, 50.0, 0.0], [-50.0, 50.0, 0.0], # 大地
    [BX-BW/2, -BD/2, 0.0], [BX+BW/2, -BD/2, 0.0], [BX+BW/2, BD/2, 0.0], [BX-BW/2, BD/2, 0.0], # 箱底
    [BX-BW/2, -BD/2, BH], [BX+BW/2, -BD/2, BH], [BX+BW/2, BD/2, BH], [BX-BW/2, BD/2, BH]  # 箱顶
]
mesh.CreatePointsAttr(vertices)

# 2. 面数汇总 (1个大地面 + 6个箱子面 = 7个四边形)
mesh.CreateFaceVertexCountsAttr([4] * 7)

# 3. 拓扑顺序 (缝合)
mesh.CreateFaceVertexIndicesAttr([
    0, 1, 2, 3,       # 面 0: 大地 (法线向上)
    4, 7, 6, 5,       # 面 1: 箱底 (法线向下)
    8, 9, 10, 11,     # 面 2: 箱顶 (法线向上)
    4, 5, 9, 8,       # 面 3: 箱前
    5, 6, 10, 9,      # 面 4: 箱右
    6, 7, 11, 10,     # 面 5: 箱后
    7, 4, 8, 11       # 面 6: 箱左
])

# 4. 颜色区分：大地灰色，箱子红色
face_colors = [(0.4, 0.4, 0.4)] + [(1.0, 0.0, 0.0)] * 6
color_attr = mesh.CreateDisplayColorAttr(face_colors)
color_attr.SetMetadata("interpolation", "uniform")

# 5. 物理注入
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
mesh_col = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
mesh_col.CreateApproximationAttr().Set("none") # 必须为 none 才能识别出箱子的高度

save_path = "/home/ai/whole_body_tracking/artifacts/custom_jump.usd"
stage.Export(save_path)
print(f"✅ 地形缝合完毕！路径: {save_path}")
simulation_app.close()