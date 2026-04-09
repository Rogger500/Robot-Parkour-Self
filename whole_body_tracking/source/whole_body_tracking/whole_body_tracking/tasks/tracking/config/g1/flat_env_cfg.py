from isaaclab.utils import configclass # 导入配置类装饰器

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG # 导入机器人配置
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE # 导入低频控制相关的参数调整比例
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg # 导入基础环境配置类


@configclass
class G1FlatEnvCfg(TrackingEnvCfg): # 这是我们新建的环境配置类，继承自 TrackingEnvCfg
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # 使用 G1 圆柱体机器人配置
        self.actions.joint_pos.scale = G1_ACTION_SCALE # 设置动作缩放比例
        self.commands.motion.motion_file = "/home/ai/whole_body_tracking/artifacts/jumps1_subject1:v0/motion.npz" # 设置要跟踪的动作文件路径
        self.commands.motion.anchor_body_name = "torso_link" # 设置动作锚点为躯干链接
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg): # 不提供状态估计输入的环境配置
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg): # 低频版本的环境配置
    def __post_init__(self): # 通过调整环境的决策频率和奖励权重来适配低频控制
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE) # 调整环境的决策频率
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE # 调整动作奖励权重，保持奖励尺度不变
