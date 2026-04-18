import torch

from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


def custom_get_motion_phase(env):
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
        # 1. Robot
        # ==============================================================
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=False,
            force_threshold=10.0,
        )

        if hasattr(self, "actions") and hasattr(self.actions, "joint_pos"):
            self.actions.joint_pos.scale = 1.0

        # ==============================================================
        # 2. Motion
        # ==============================================================
        self.commands.motion.motion_file = (
            "/home/ai/whole_body_tracking/artifacts/aligned_teacher/"
            "climb_15_aligned_0000_box_95_190.npz"
        )
        self.commands.motion.debug_vis = False
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
        # 3. Terrain
        # 地面 + 障碍物分开加载
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

        # 高度扫描同时包含地面和障碍物
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.1,
                size=(0.9, 0.9),
            ),
            debug_vis=False,
            mesh_prim_paths=[
                "/World/ground",
                "/World/envs/env_.*/obstacles",
            ],
        )

        # ==============================================================
        # 4. Observations
        # ==============================================================
        if hasattr(self.observations, "policy"):
            self.observations.policy.motion_phase = ObservationTermCfg(
                func=custom_get_motion_phase,
                clip=(0.0, 1.0),
            )

        if hasattr(self.observations, "critic"):
            self.observations.critic.motion_phase = ObservationTermCfg(
                func=custom_get_motion_phase,
                clip=(0.0, 1.0),
            )

        # ==============================================================
        # 5. Rewards
        # ==============================================================
        _safe_set_weight(self.rewards, "motion_global_anchor_pos", 10.0)
        _safe_set_weight(self.rewards, "motion_global_anchor_ori", 2.0)
        _safe_set_weight(self.rewards, "motion_body_pos", 2.0)
        _safe_set_weight(self.rewards, "motion_body_ori", 0.5)
        _safe_set_weight(self.rewards, "motion_body_lin_vel", 1.0)
        _safe_set_weight(self.rewards, "motion_body_ang_vel", 0.5)
        _safe_set_weight(self.rewards, "action_rate_l2", -0.02)
        _safe_set_weight(self.rewards, "undesired_contacts", -0.05)

        # joint_limit 保留默认，但如果你想放松一点，也可以取消注释下面这行
        # _safe_set_weight(self.rewards, "joint_limit", -0.03)

        # ==============================================================
        # 6. Terminations
        # ==============================================================
        if hasattr(mdp, "bad_anchor_pos"):
            self.terminations.anchor_pos = TerminationTermCfg(
                func=mdp.bad_anchor_pos,
                params={"command_name": "motion", "threshold": 1.8},
            )
        else:
            _safe_set_param(self.terminations, "anchor_pos", "threshold", 1.8)

        _safe_set_param(self.terminations, "anchor_ori", "threshold", 1.2)
        _safe_disable(self.terminations, "ee_body_pos")

        # ==============================================================
        # 7. Events
        # ==============================================================
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
        _safe_set_param(
            self.events,
            "add_joint_default_pos",
            "pos_distribution_params",
            (-0.005, 0.005),
        )

        # ==============================================================
        # 8. Episode length
        # ==============================================================
        if hasattr(self, "episode_length_s"):
            self.episode_length_s = 3.0