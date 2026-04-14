import numpy as np

input_file = "/home/ai/whole_body_tracking/artifacts/jumps1_subject1:v0/motion.npz"
output_file = "/home/ai/whole_body_tracking/artifacts/short_jump.npz"

print(f"正在读取原始长动作: {input_file}...")
data = np.load(input_file)

# 打印出文件里到底有哪些数据
print("📦 文件中包含的变量 (Keys):", data.files)

# 1. 智能寻找包含根节点 3D 坐标的 Key
possible_pos_keys = ['root_pos', 'root_translation', 'root_trans', 'body_pos_w', 'position']
pos_key = None

for key in possible_pos_keys:
    if key in data.files:
        pos_key = key
        break

# 如果靠名字没找到，退而求其次靠形状找！(寻找类似 (N, 3) 或 (N, bodies, 3) 的大数组)
if pos_key is None:
    for key in data.files:
        shape = data[key].shape
        if len(shape) >= 2 and shape[-1] == 3:
            pos_key = key
            break

if pos_key is None:
    raise ValueError(f"🚨 致命错误：找不到表示坐标的 3D 数据！文件里只有: {data.files}")

print(f"✅ 自动锁定位置数据键: '{pos_key}' (数据形状: {data[pos_key].shape})")

positions = data[pos_key]

# 2. 提取盆骨 (Root) 高度轨迹
if len(positions.shape) == 3:
    # 形状若是 (num_frames, num_bodies, 3) -> 取第 0 个刚体作为 root
    pelvis_pos = positions[:, 0, :]  
elif len(positions.shape) == 2 and positions.shape[1] == 3:
    # 形状若是 (num_frames, 3) -> 这本来就是 root 的位置
    pelvis_pos = positions
else:
    raise ValueError(f"🚨 解析 '{pos_key}' 的形状 {positions.shape} 失败！")

z_coords = pelvis_pos[:, 2]
x_coords = pelvis_pos[:, 0]

# 3. 找到最高点 (Apex)
apex_frame = np.argmax(z_coords)
total_frames = len(z_coords)

# ==============================================================
# 🌟 核心设置：截取窗口大小 (假设控制频率为 50 FPS)
# ==============================================================
FPS = 50 
# 我们截取起跳前 1.5 秒，和落地后 1.0 秒，刚好是一个完整的短技能
frames_before = int(1.5 * FPS)
frames_after = int(1.0 * FPS)

start_frame = max(0, apex_frame - frames_before)
end_frame = min(total_frames, apex_frame + frames_after)

print(f"📊 原始总帧数: {total_frames}")
print(f"🚀 最高点位于第 {apex_frame} 帧 (当前高度: {z_coords[apex_frame]:.3f} m)")
print(f"🔪 截取区间: [{start_frame} : {end_frame}] (共 {end_frame - start_frame} 帧)")

# 4. 开始执行切割
new_data = {}
for key in data.files:
    array = data[key]
    # 动捕数据通常第一维度是时间 (Time/Frames)
    if isinstance(array, np.ndarray) and array.shape and array.shape[0] == total_frames:
        new_data[key] = array[start_frame:end_frame]
    else:
        # 像 fps、性别 等全局静态参数，直接保留
        new_data[key] = array

# 5. 保存为新的短技能文件
np.savez(output_file, **new_data)
print(f"✅ 短技能动作切割完毕，已保存至: {output_file}")

# 6. 计算绝对对齐参数
start_x = x_coords[start_frame]
apex_x = x_coords[apex_frame]
suggested_BX = apex_x - start_x
estimated_BH = z_coords[apex_frame] - 0.75 # 0.75 为机器人腿长估算值

print("\n" + "="*40)
print("🎯 地形对齐参数建议 (请把以下数值填入 build_terrain.py):")
print(f"👉 箱子相对距离 (BX) : {suggested_BX:.3f} 米")
print(f"👉 箱子最大高度 (BH) : 不超过 {estimated_BH:.3f} 米")
print("="*40)