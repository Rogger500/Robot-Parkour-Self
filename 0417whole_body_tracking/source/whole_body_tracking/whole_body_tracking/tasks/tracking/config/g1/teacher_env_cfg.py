import torch
from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils

# 导入 mdp 库，使用原代码库中真实的函数
import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


def custom_get_motion_phase(env):
    """通过环境当前的步数计算动作进度 (0.0 到 1.0)，带终极安全锁"""
    try:
        max_steps = float(env.max_episode_length)
        if max_steps <= 0.0:
            max_steps = 500.0
    except Exception:
        max_steps = 500.0

    phase = env.episode_length_buf / max_steps
    phase = torch.nan_to_num(phase, nan=0.0, posinf=1.0, neginf=0.0)
    phase = torch.clamp(phase, min=0.0, max=1.0)
    return phase.unsqueeze(1)


@configclass
class G1TeacherEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        # 🌟 这一步极其重要：它会把 TrackingEnvCfg 里所有的设定加载进来
        # 包含：自适应采样、全局观测特征、基础奖励函数、基础领域随机化等
        super().__post_init__()

        # ==============================================================
        # 1. 机器人与基础配置
        # ==============================================================
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 覆盖基类的接触传感器配置 (修改了 history_length 和 track_air_time)
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False, force_threshold=10.0
        )
        
        # 🌟 专家动作输出不进行缩放 (严格对齐 PHP 论文)
        self.actions.joint_pos.scale = 1.0  
        
        # ==============================================================
        # 🌟 动作文件切换为 short_jump.npz
        # ==============================================================
        # self.commands.motion.motion_file = "/home/ai/whole_body_tracking/artifacts/jumps1_subject1:v0/motion.npz"
        self.commands.motion.motion_file = "/home/ai/whole_body_tracking/artifacts/short_jump.npz"
        self.commands.motion.debug_vis = False
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
            "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
            "torso_link", "left_shoulder_roll_link", "left_elbow_link",
            "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        # ==============================================================
        # 2. 地形与天眼配置
        # ==============================================================
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", 
            # terrain_type="usd",
            terrain_type="plane", # 🌟 保持原生平地
            # usd_path="/home/ai/whole_body_tracking/artifacts/custom_jump.usd", 
            collision_group=-1,
        )

        # 🌟 添加高度扫描 (天眼)
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            # 🌟 修复 Isaac Lab 警告：使用 ray_alignment 替代 attach_yaw_only
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.7, 0.7)), 
            debug_vis=True, 
            mesh_prim_paths=["/World/ground"], 
        )

        # ==============================================================
        # 3. 观测值 (Observations)：在基类基础上【追加】新特征
        # ==============================================================
        # 注意：基类已经塞入了 body_pos, base_lin_vel, joint_pos 等！我们只追加自定义的。
        
        # --- Policy (Actor) ---
        self.observations.policy.height_scan = ObservationTermCfg(
            func=mdp.height_scan, # 请确保你的 mdp 中实现了此函数
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0), 
        )
        self.observations.policy.motion_phase = ObservationTermCfg(
            func=custom_get_motion_phase, 
            clip=(0.0, 1.0),
        )

        # --- Critic (上帝视角) ---
        self.observations.critic.height_scan = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        self.observations.critic.motion_phase = ObservationTermCfg(
            func=custom_get_motion_phase,
            clip=(0.0, 1.0),
        )

        # ==============================================================
        # 4. 奖励配置 (Rewards)：直接修改基类的权重以对齐 PHP 论文
        # ==============================================================
        # 基类的 body_pos 等权重本身就是 1.0，不需要改。我们只改不同的：
        self.rewards.motion_global_anchor_pos.weight = 10.0
        self.rewards.motion_global_anchor_ori.weight = 1.0
        self.rewards.undesired_contacts.weight = -0.5

        # 🌟 修复报错：删除 joint_accel，加大已有的 action_rate_l2 惩罚
        # 基类中 action_rate_l2 默认是 -0.1，这里将其放大到 -0.5 甚至 -1.0，来强力抑制高频打摆子
        self.rewards.action_rate_l2.weight = -0.5

        # ==============================================================
        # 5. 终止条件配置 (Terminations)：核心修复区
        # ==============================================================
        # 基类默认用的是Z轴误差 (bad_anchor_pos_z_only)。
        # PHP 论文中，专家用的是 3D追踪误差 0.5m，所以我们覆盖为库里真实的 bad_anchor_pos 函数：
        self.terminations.anchor_pos = TerminationTermCfg(
            func=mdp.bad_anchor_pos, 
            params={"command_name": "motion", "threshold": 0.5},
            # 🌟 极其关键：将容忍度放大到 2.5 米！让它能在平地上坚持把跳跃动作做完。
            # params={"command_name": "motion", "threshold": 2.5},
        )
        
        # 姿态倾斜也需要收紧，防止它学出奇怪的蹲姿
        self.terminations.anchor_ori.params["threshold"] = 0.5
        
        # 按照你的要求关闭末端误差终止
        self.terminations.ee_body_pos = None

        # ==============================================================
        # 6. 领域随机化配置 (Events)：直接修改基类设定
        # ==============================================================
        # 摩擦力范围修改 (基类名称为 physics_material)
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.3)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.3)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)

        # 机器人受力扰动范围修改 (基类名称为 push_robot)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)
        }
        
        # 质心偏移修改 (基类名称为 base_com，将y范围缩小至对齐PHP论文)
        self.events.base_com.params["com_range"] = {
            "x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.05, 0.05) 
        }
        
        # 关节初始位置偏差 (基类名称为 add_joint_default_pos，已自动包含)
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (-0.01, 0.01)