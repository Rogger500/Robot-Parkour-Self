import numpy as np

# 1. 加载你的动作数据文件
# file_path = "/home/ai/whole_body_tracking/artifacts/jumps1_subject1:v0/motion.npz"
file_path = "/home/ai/whole_body_tracking/artifacts/aligned_teacher/climb_15_aligned_0000.npz"
data = np.load(file_path)

# body_pos_w 的形状大概是 (帧数, 刚体数, 3)
# 索引 0 通常是 Pelvis (骨盆/质心)
pelvis_pos = data["body_pos_w"][:, 0, :] 

# 提取 Z 轴 (高度) 和 X 轴 (前进方向)
z_heights = pelvis_pos[:, 2]
x_positions = pelvis_pos[:, 0]

# 找到高度最高的那一帧 (跳跃的最高点)
peak_idx = np.argmax(z_heights)

# 简单推算：起跳点大概是最高点往前推一点，落地点是往后推一点
# 我们直接打印最高点的 X 坐标和 Z 坐标
print("="*50)
print(f"🎯 动作分析完成！")
print(f"跳跃最高点发生在第 {peak_idx} 帧")
print(f"此时人的 X 轴坐标 (前方位置): {x_positions[peak_idx]:.2f} 米")
print(f"此时人的 Z 轴坐标 (绝对高度): {z_heights[peak_idx]:.2f} 米")
print("="*50)