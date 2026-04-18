import torch

from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG, G1_ACTION_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


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

        if hasattr(self, "actions") and hasattr(self.actions, "joint_pos"):
            self.actions.joint_pos.scale = G1_ACTION_SCALE

        # ==============================================================
        # 2. Motion file: 用裁剪后的关键段
        # ==============================================================
        # self.commands.motion.motion_file = "/home/ai/whole_body_tracking/artifacts/aligned_teacher/climb_15_aligned_0000_box_95_190.npz"
        self.commands.motion.motion_file = "/home/ai/whole_body_tracking/artifacts/aligned_teacher/climb_15_aligned_0000.npz"
        self.commands.motion.debug_vis = False

        # 关键：和 bake 时的 root_body_idx=0 对齐，统一用 pelvis
        self.commands.motion.anchor_body_name = "pelvis"

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
        # 3. Terrain: 直接用 combined terrain
        # ==============================================================
        """ self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path="/home/ai/whole_body_tracking/artifacts/aligned_teacher/terrain_combined_0000.usd",
            collision_group=-1,
        ) """
                # ==============================================================
        # 3. Terrain + obstacles
        # 先回到最稳的方案：平地 + 单独障碍物 USD
        # ==============================================================
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
        )

        self.scene.obstacles = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/obstacles",
            spawn=UsdFileCfg(
                usd_path="/home/ai/whole_body_tracking/artifacts/aligned_teacher/multi_boxes_aligned_0000.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
            ),
        )

        """ # 不再单独挂 obstacles
        if hasattr(self.scene, "obstacles"):
            self.scene.obstacles = None """

        # ==============================================================
        # 4. Observations
        # teacher 阶段先不要 height_scan，先把 imitation 训通
        # ==============================================================
        if hasattr(self.observations, "policy") and hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None

        if hasattr(self.observations, "critic") and hasattr(self.observations.critic, "height_scan"):
            self.observations.critic.height_scan = None

        # ==============================================================
        # 5. Rewards
        # ==============================================================
        _safe_set_weight(self.rewards, "motion_global_anchor_pos", 10.0)
        _safe_set_weight(self.rewards, "motion_global_anchor_ori", 2.0)
        _safe_set_weight(self.rewards, "motion_body_pos", 2.0)
        _safe_set_weight(self.rewards, "motion_body_ori", 0.5)
        _safe_set_weight(self.rewards, "motion_body_lin_vel", 1.0)
        _safe_set_weight(self.rewards, "motion_body_ang_vel", 0.5)

        _safe_set_weight(self.rewards, "undesired_contacts", -0.05)
        _safe_set_weight(self.rewards, "action_rate_l2", -0.02)
        _safe_set_weight(self.rewards, "joint_limit", -0.05)

        # ==============================================================
        # 6. Terminations
        # 先放宽，避免刚学起跳就判死
        # ==============================================================
        if hasattr(mdp, "bad_anchor_pos"):
            self.terminations.anchor_pos = TerminationTermCfg(
                func=mdp.bad_anchor_pos,
                params={"command_name": "motion", "threshold": 2.0},
            )
        else:
            _safe_set_param(self.terminations, "anchor_pos", "threshold", 2.0)

        _safe_set_param(self.terminations, "anchor_ori", "threshold", 1.2)

        _safe_disable(self.terminations, "ee_body_pos")

        # ==============================================================
        # 7. Domain randomization / events
        # ==============================================================
        _safe_set_param(self.events, "physics_material", "static_friction_range", (0.9, 1.1))
        _safe_set_param(self.events, "physics_material", "dynamic_friction_range", (0.9, 1.1))
        _safe_set_param(self.events, "physics_material", "restitution_range", (0.0, 0.0))

        _safe_disable(self.events, "push_robot")

        _safe_set_param(
            self.events,
            "base_com",
            "com_range",
            {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.01, 0.01)},
        )
        _safe_set_param(self.events, "add_joint_default_pos", "pos_distribution_params", (-0.002, 0.002))

        # ==============================================================
        # 8. Episode length
        # 100~200 帧如果是 50fps，大约 2 秒
        # ==============================================================
        if hasattr(self, "episode_length_s"):
            self.episode_length_s = 2.2