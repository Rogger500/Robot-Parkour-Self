import math
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics

print("正在生成完美对齐的跳远地垫...")

stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root_prim = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(root_prim)
mesh = UsdGeom.Mesh.Define(stage, "/World/terrain")

NUM_ENVS = 4096
ENV_SPACING = 2.5

# ==========================================================
# 🌟 填入你刚刚获取的绝对坐标！
# ==========================================================
BX = -1.318
BY = -2.346

# 🌟 高度设为 2厘米 (0.02m)，仅作雷达触发用，防止绊倒
BH = 0.02    
BW = 0.65     # 垫子宽度 (X轴方向的厚度)
BD = 3.0     # 垫子长度 (Y轴方向的长度，设宽一点防止跑歪)

num_cols = math.ceil(math.sqrt(NUM_ENVS))

# 铺设足以覆盖绝对坐标的大地面 (注意扩大范围)
vertices = [[-50.0, -50.0, 0.0], [150.0, -50.0, 0.0], [150.0, 150.0, 0.0], [-50.0, 150.0, 0.0]]
face_vertex_counts = [4]
face_vertex_indices = [0, 1, 2, 3]
face_colors = [(0.4, 0.4, 0.4)]

v_idx = 4
for i in range(NUM_ENVS):
    env_x = (i // num_cols) * ENV_SPACING
    env_y = (i % num_cols) * ENV_SPACING
    
    # 🌟 精确的绝对坐标定位
    cx = env_x + BX
    cy = env_y + BY
    
    vertices.extend([
        [cx-BW/2, cy-BD/2, 0.0], [cx+BW/2, cy-BD/2, 0.0], [cx+BW/2, cy+BD/2, 0.0], [cx-BW/2, cy+BD/2, 0.0],
        [cx-BW/2, cy-BD/2, BH],  [cx+BW/2, cy-BD/2, BH],  [cx+BW/2, cy+BD/2, BH],  [cx-BW/2, cy+BD/2, BH]  
    ])
    face_vertex_counts.extend([4] * 6)
    face_vertex_indices.extend([
        v_idx, v_idx+3, v_idx+2, v_idx+1,
        v_idx+4, v_idx+5, v_idx+6, v_idx+7,
        v_idx, v_idx+1, v_idx+5, v_idx+4,
        v_idx+1, v_idx+2, v_idx+6, v_idx+5,
        v_idx+2, v_idx+3, v_idx+7, v_idx+6,
        v_idx+3, v_idx, v_idx+4, v_idx+7
    ])
    face_colors.extend([(1.0, 0.0, 0.0)] * 6)
    v_idx += 8

mesh.CreatePointsAttr(vertices)
mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
color_attr = mesh.CreateDisplayColorAttr(face_colors)
color_attr.SetMetadata("interpolation", "uniform")

UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
mesh_col = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
mesh_col.CreateApproximationAttr().Set("none") 

save_path = "/home/ai/whole_body_tracking/artifacts/custom_jump.usd"
stage.Export(save_path)
print(f"✅ 精准跳远地垫生成完毕！路径: {save_path}")
simulation_app.close()