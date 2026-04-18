import numpy as np

# ==============================================================
# 1. 输入 / 输出文件
# ==============================================================
input_file = "/home/ai/whole_body_tracking/artifacts/aligned_teacher/climb_15_aligned_0000.npz"
output_file = "/home/ai/whole_body_tracking/artifacts/aligned_teacher/climb_15_aligned_0000_box_95_190.npz"

print(f"正在读取原始动作: {input_file}...")
data = np.load(input_file)

print("📦 文件中包含的变量 (Keys):", data.files)

# ==============================================================
# 2. 自动寻找位置数据 Key
# ==============================================================
possible_pos_keys = ["body_pos_w", "root_pos", "root_translation", "root_trans", "position"]
pos_key = None

for key in possible_pos_keys:
    if key in data.files:
        pos_key = key
        break

if pos_key is None:
    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray) and len(arr.shape) >= 2 and arr.shape[-1] == 3:
            pos_key = key
            break

if pos_key is None:
    raise ValueError(f"🚨 找不到表示坐标的 3D 数据！文件里只有: {data.files}")

print(f"✅ 自动锁定位置数据键: '{pos_key}' (数据形状: {data[pos_key].shape})")

positions = data[pos_key]

# ==============================================================
# 3. 提取 pelvis / root 轨迹
# ==============================================================
if len(positions.shape) == 3:
    # (num_frames, num_bodies, 3)
    pelvis_pos = positions[:, 0, :]
elif len(positions.shape) == 2 and positions.shape[1] == 3:
    # (num_frames, 3)
    pelvis_pos = positions
else:
    raise ValueError(f"🚨 解析 '{pos_key}' 的形状 {positions.shape} 失败！")

x_coords = pelvis_pos[:, 0]
y_coords = pelvis_pos[:, 1]
z_coords = pelvis_pos[:, 2]

total_frames = len(z_coords)
apex_frame = int(np.argmax(z_coords))

print(f"📊 原始总帧数: {total_frames}")
print(f"🚀 高度最高点位于第 {apex_frame} 帧 (高度: {z_coords[apex_frame]:.3f} m)")

# ==============================================================
# 4. 手动指定我们要的“接近箱子关键段”
#    依据前面的分析，先用 [110, 190)
# ==============================================================
manual_start = 95
manual_end = 190

# 安全裁剪，避免越界
start_frame = max(0, manual_start)
end_frame = min(total_frames, manual_end)

if start_frame >= end_frame:
    raise ValueError(
        f"🚨 截取区间非法: start_frame={start_frame}, end_frame={end_frame}, total_frames={total_frames}"
    )

print(f"🔪 最终截取区间: [{start_frame} : {end_frame}] (共 {end_frame - start_frame} 帧)")

# ==============================================================
# 5. 执行切割
#    只切第一维等于 total_frames 的数组；其余全局参数原样保留
# ==============================================================
new_data = {}
for key in data.files:
    array = data[key]
    if isinstance(array, np.ndarray) and array.shape and array.shape[0] == total_frames:
        new_data[key] = array[start_frame:end_frame]
    else:
        new_data[key] = array

# ==============================================================
# 6. 保存
# ==============================================================
np.savez(output_file, **new_data)
print(f"✅ 关键段动作切割完毕，已保存至: {output_file}")

# ==============================================================
# 7. 额外打印一下裁剪段的起止信息，便于核对
# ==============================================================
crop_start_x = x_coords[start_frame]
crop_start_y = y_coords[start_frame]
crop_start_z = z_coords[start_frame]

crop_end_x = x_coords[end_frame - 1]
crop_end_y = y_coords[end_frame - 1]
crop_end_z = z_coords[end_frame - 1]

print("\n" + "=" * 60)
print("🎯 裁剪段信息")
print(f"起始帧: {start_frame}")
print(f"结束帧: {end_frame - 1}")
print(f"起始 pelvis 位置: x={crop_start_x:.3f}, y={crop_start_y:.3f}, z={crop_start_z:.3f}")
print(f"结束 pelvis 位置: x={crop_end_x:.3f}, y={crop_end_y:.3f}, z={crop_end_z:.3f}")
print(f"最高点帧(原序列): {apex_frame}, 高度={z_coords[apex_frame]:.3f} m")
print("=" * 60)