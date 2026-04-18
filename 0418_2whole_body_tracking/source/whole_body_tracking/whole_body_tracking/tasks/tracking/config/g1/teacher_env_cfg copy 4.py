import torch

from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.terrains import TerrainImporterCfg
import os
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg  # 🌟 改用原生的 USD 加载器

# 使用项目里的真实 mdp 函数
import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG, G1_ACTION_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


# ==========================================================
# Helper functions
# ==========================================================


def _safe_set_weight(cfg_obj, name, weight):
    term = getattr(cfg_obj, name, None)
    if term is not None and hasattr(term, "weight"):
        term.weight = weight


def _safe_set_param(cfg_obj, name, key, value):
    term = getattr(cfg_obj, name, None)
    if term is not None and hasattr(term, "params") and isinstance(term.params, dict):
        term.params[key] = value


def _safe_disable(cfg_obj, name):
    if hasattr(cfg_obj, name):
        setattr(cfg_obj, name, None)


@configclass
class G1TeacherEnvCfg(TrackingEnvCfg):
    """
    short_jump.npz 专用 teacher 配置。

    目标：
    - 仍然保持 teacher 的 tracking 设定
    - 但把过于激进的惩罚和随机化先降下来
    - 让 short_jump 这条 reference 至少能先稳定跟踪，再逐步加难度
    """

    def __post_init__(self):
        super().__post_init__()

        # ==============================================================
        # 1. Robot / action
        # ==============================================================
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=False,
            force_threshold=10.0,
        )

        # expert 动作不缩放
        if hasattr(self, "actions") and hasattr(self.actions, "joint_pos"):
            self.actions.joint_pos.scale = G1_ACTION_SCALE

        # ==============================================================
        # 2. Motion file: 固定用 short_jump.npz
        # ==============================================================
        # self.commands.motion.motion_file = "/home/ai/whole_body_tracking/artifacts/short_jump.npz"
        self.commands.motion.motion_file = "/home/ai/whole_body_tracking/artifacts/aligned_teacher/climb_15_aligned_0000.npz"
        self.commands.motion.debug_vis = False

        # 对 jump reference，先沿用你原来的 torso_link 作为 anchor
        self.commands.motion.anchor_body_name = "torso_link"
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
        
        # 🌟 终极方案：直接加载原生 USD 箱子
        self.scene.obstacles = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/obstacles",
            spawn=UsdFileCfg(
                usd_path="/home/ai/whole_body_tracking/artifacts/aligned_teacher/multi_boxes_aligned_0000.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
            ),
        )

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.7, 0.7)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # ==============================================================
        # 4. Observations
        # ==============================================================
        # 暂时关闭 height_scan，先确保 replay 能正常跑通
        if hasattr(self.observations, "policy"):
            pass

        if hasattr(self.observations, "critic"):
            pass
        # ==============================================================
        # 5. Rewards
        # ==============================================================
        # 你的原版这里有点过猛：action_rate_l2=-0.5 容易把起跳能力也压掉。
        # 先让 short_jump 做出来，再考虑收紧。
        _safe_set_weight(self.rewards, "motion_global_anchor_pos", 8.0)
        _safe_set_weight(self.rewards, "motion_global_anchor_ori", 1.0)
        _safe_set_weight(self.rewards, "undesired_contacts", -0.1)
        _safe_set_weight(self.rewards, "action_rate_l2", -0.05)

        # ==============================================================
        # 6. Terminations
        # ==============================================================
        # short_jump 的 root/anchor 偏移会比普通 walk 大，0.5 往往太苛刻。
        # 先放宽到 1.5，看它能不能完整跳完。
        if hasattr(mdp, "bad_anchor_pos"):
            self.terminations.anchor_pos = TerminationTermCfg(
                func=mdp.bad_anchor_pos,
                params={"command_name": "motion", "threshold": 1.5},
            )
        else:
            _safe_set_param(self.terminations, "anchor_pos", "threshold", 1.5)

        _safe_set_param(self.terminations, "anchor_ori", "threshold", 0.8)

        # 先关掉末端误差终止，避免 short_jump 还没学会就被手脚误差判死刑
        _safe_disable(self.terminations, "ee_body_pos")

        # ==============================================================
        # 7. Domain randomization / events
        # ==============================================================
        # 先把随机化调轻，不然 short_jump 本来就难，还要边跳边挨揍。
        _safe_set_param(self.events, "physics_material", "static_friction_range", (0.8, 1.2))
        _safe_set_param(self.events, "physics_material", "dynamic_friction_range", (0.8, 1.2))
        _safe_set_param(self.events, "physics_material", "restitution_range", (0.0, 0.0))

        _safe_disable(self.events, "push_robot")

        _safe_set_param(
            self.events,
            "base_com",
            "com_range",
            {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.02, 0.02)},
        )
        _safe_set_param(self.events, "add_joint_default_pos", "pos_distribution_params", (-0.005, 0.005))

        # ==============================================================
        # 8. Episode length
        # ==============================================================
        if hasattr(self, "episode_length_s"):
            self.episode_length_s = 6.0
