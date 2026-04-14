import numpy as np

# 加载动作文件
path = "artifacts/jumps1_subject1:v0/motion.npz"
data = np.load(path)

# body_pos_w 的维度通常是 [总帧数, 身体部位数, 3]
# 索引 0 通常是根节点（盆骨）
body_pos_w = data['body_pos_w']
root_pos = body_pos_w[:, 0, :] # 取出每一帧的根节点 XYZ

# 1. 找到 Z 轴（高度）最大的那一帧
jump_apex_idx = np.argmax(root_pos[:, 2])
target_bx = root_pos[jump_apex_idx, 0]
max_height = root_pos[jump_apex_idx, 2]

# 2. 顺便看一下起跳前的初始 X 坐标，确认它是不是从 0 开始的
start_x = root_pos[0, 0]

print(f"--- 动作分析结果 ---")
print(f"起始 X 坐标: {start_x:.4f}")
print(f"检测到机器人在 X = {target_bx:.4f} 处达到跳跃最高点 (高度: {max_height:.4f}m)")
print(f"建议将 build_terrain.py 里的 BX 设为: {target_bx:.2f}")