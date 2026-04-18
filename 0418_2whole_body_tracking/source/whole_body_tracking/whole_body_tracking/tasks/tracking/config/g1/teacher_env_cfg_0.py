import torch
from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils

# 导入 mdp 库
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
        super().__post_init__()

        # ==============================================================
        # 1. 机器人与基础配置
        # ==============================================================
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 覆盖基类的接触传感器配置
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False, force_threshold=10.0
        )
        
        # 🌟 专家动作输出不进行缩放 (严格对齐 PHP 论文)
        self.actions.joint_pos.scale = 1.0  
        
        # 动作文件与骨骼配置
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
            terrain_type="plane", # 🌟 改回原生平地
            # usd_path="/home/ai/whole_body_tracking/artifacts/custom_jump.usd", 
            collision_group=-1,
        )

        # 添加高度扫描
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.7, 0.7)), 
            debug_vis=True, 
            mesh_prim_paths=["/World/ground/terrain"], 
        )

        # 🌟 关闭高度扫描，平地不需要天眼，节省大量算力
        # self.scene.height_scanner = None

        # ==============================================================
        # 3. 观测值 (Observations)：在基类基础上【追加】新特征
        # ==============================================================
        # 注意：基类已经塞入了 body_pos, base_lin_vel, joint_pos 等！我们只追加自定义的。
        # 🌟 必须将 height_scan 设为 None，否则会因为找不到传感器而崩溃！
        # self.observations.policy.height_scan = None
        # self.observations.critic.height_scan = None
        
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
        # 4. 奖励配置 (Rewards) - 严格对齐 Table IV
        # ==============================================================
        # 追踪项权重 (基类通常默认为 1.0，但核心 Anchor 为 10)
        self.rewards.motion_global_anchor_pos.weight = 10.0
        self.rewards.motion_global_anchor_ori.weight = 1.0
        
        # 惩罚项权重
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.joint_limit.weight = -10.0     # 对齐 Table IV

        # ==============================================================
        # 5. 终止条件配置 (Terminations) - 严格对齐 Expert 设定
        # ==============================================================
        self.terminations.anchor_pos = TerminationTermCfg(
            func=mdp.bad_anchor_pos, 
            # params={"command_name": "motion", "threshold": 0.5}, # 对齐 Expert 0.5m 死亡线
            params={"command_name": "motion", "threshold": 2.5},
        )
        
        self.terminations.anchor_ori.params["threshold"] = 1.5 
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
        
        # 关节初始偏差
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (-0.01, 0.01)
        
        # (可选) 如果你基类里支持分离踝关节随机化，可开启以下配置对齐 Table V
        # self.events.add_ankle_default_pos.params["pos_distribution_params"] = (-0.03, 0.03)

        # ==============================================================
        # ⚠️ 终极补丁：治疗“高频鬼畜”作弊现象
        # ==============================================================
        self.observations.policy.enable_corruption = False  # 保持关闭
        
        # 1. 治本：加大平滑惩罚！
        # 论文里的 -0.1 压不住你的物理参数，给它上强度，直接加到 -1.0 甚至 -5.0
        # 这会逼迫它：只要敢高频抽搐，就扣光它的分，逼它用平滑连贯的动作去跳。
        self.rewards.action_rate_l2.weight = -1.0  
        
        # 2. 治标：提升关节位置的模仿权重
        # 告诉它：“光躯干到位不行，你的腿也必须给我弯成参考动作那样！”
        self.rewards.motion_body_pos.weight = 5.0  # 原来是 1.0，现在加大 5 倍
        