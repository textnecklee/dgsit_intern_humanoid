# end_cfg_pos_custom_leggedgym_like.py
# Isaac Lab EnvCfg for quadruped (legged_gym-style rewards)

from ast import Tuple
import math
from re import T
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaaclab_mdp  # undesired_contacts 함수 사용을 위해
from . import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg


# --------------------------------------------------------------------------- #
#  Custom Articulation with Joint Limits Fix
# --------------------------------------------------------------------------- #
class QuadArticulation(Articulation):
    """Custom Articulation class that sets joint limits if they are [0.0, 0.0] in USD file."""
    
    def _validate_cfg(self):
        """Validate configuration and fix joint limits if needed."""
        # Get joint limits from simulation (after PhysX view is initialized)
        joint_pos_limits = self.root_physx_view.get_dof_limits()[0].to(self.device)
        
        # Default joint limits from URDF (in radians)
        # URDF values: HAA: ±30°, HIP: -45°~+135°, KNEE: 45°~135°
        default_limits = {
            "FLHAA": (-0.523, 0.523),   # Hip abduction/adduction: ±30 degrees
            "FLHIP": (-0.785, 2.355),   # Hip flexion/extension: -45° to +135°
            "FLKNEE": (0.785, 2.355),  # Knee: 45° to 135° (cannot fully extend)
            "FRHAA": (-0.523, 0.523),
            "FRHIP": (-0.785, 2.355),
            "FRKNEE": (0.785, 2.355),
            "RLHAA": (-0.523, 0.523),
            "RLHIP": (-0.785, 2.355),
            "RLKNEE": (0.785, 2.355),
            "RRHAA": (-0.523, 0.523),
            "RRHIP": (-0.785, 2.355),
            "RRKNEE": (0.785, 2.355),
        }
        
        # Check which joints need limits set
        needs_fix = []
        for idx, joint_name in enumerate(self.joint_names):
            limits = joint_pos_limits[idx]
            lower, upper = limits[0].item(), limits[1].item()
            
            # If limits are [0.0, 0.0], set default values
            if abs(lower) < 1e-6 and abs(upper) < 1e-6:
                if joint_name in default_limits:
                    needs_fix.append((idx, joint_name, default_limits[joint_name]))
        
        # Set joint limits if needed
        if needs_fix:
            # Create limits tensor: (num_instances, num_joints, 2)
            new_limits = self._data.joint_pos_limits.clone()
            for idx, joint_name, (lower, upper) in needs_fix:
                new_limits[:, idx, 0] = lower
                new_limits[:, idx, 1] = upper
            
            # Write to simulation
            self.write_joint_position_limit_to_sim(new_limits, warn_limit_violation=False)
            # Update joint_pos_limits after writing
            joint_pos_limits = self.root_physx_view.get_dof_limits()[0].to(self.device)
        
        # Call parent validation (this checks if default joint positions are within limits)
        super()._validate_cfg()


# --------------------------------------------------------------------------- #
#  Robot Articulation
# --------------------------------------------------------------------------- #
QUAD_CONFIG = ArticulationCfg(
    class_type=QuadArticulation,  # Use custom articulation class
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/home/teamquad/Desktop/Intern/IsaacLab/E2E_locomotion_v2/assets/MCL_Quad_serial_usd/MCL_Quad_serial.usd",
        usd_path="/home/teamquad/Desktop/Intern/urdf/Quad_v2_serial_v4/Quad_v2_serial _v4.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # activate_contact_sensors=False,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.36),
        rot=(0.0, 0.0, 0.0, 1.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos={
            # URDF 제한값 기준 초기 위치 설정
            # HAA: [-0.523, 0.523] rad (±30°)
            # HIP: [-0.785, 2.355] rad (-45° ~ +135°)
            # KNEE: [0.785, 2.355] rad (45° ~ 135°) - 최소 45도 이상 필수!
            "FLHAA": 0.0,      # 0° (중립)
            "FLHIP": 0.785,    # 45° (약간 구부림)
            "FLKNEE": 1.571,   # 90° (직각) - URDF 최소값 0.785 이상
            "FRHAA": 0.0,
            "FRHIP": 0.785,
            "FRKNEE": 1.571,
            "RLHAA": 0.0,
            "RLHIP": 0.785,
            "RLKNEE": 1.571,
            "RRHAA": 0.0,
            "RRHIP": 0.785,
            "RRKNEE": 1.571,
        },
    ),
    actuators={
        name: ImplicitActuatorCfg(
            joint_names_expr=[name],
            stiffness=40,
            damping=1,
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
        )
        for name in [
            "FLHAA", "FLHIP", "FLKNEE",
            "FRHAA", "FRHIP", "FRKNEE",
            "RLHAA", "RLHIP", "RLKNEE",
            "RRHAA", "RRHIP", "RRKNEE",
        ]
    },
)


# --------------------------------------------------------------------------- #
#  Scene + contact sensors
# --------------------------------------------------------------------------- #
@configclass
class LegSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1000.0, 1000.0)),
    )

    robot: ArticulationCfg = QUAD_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    feet_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*foot", ### 여기 foot이랑 fixed joint 로 해결가능
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=["/World/ground"],
        debug_vis=True,
        force_threshold=10.0,  # 시각화를 위한 힘 임계값 (기본값 1.0은 너무 높음) 
    )


    link_collision_sensor = ContactSensorCfg(
        # prim_path="{ENV_REGEX_NS}/Robot/.*link.*",
        prim_path="{ENV_REGEX_NS}/Robot/.*(torso|thigh|shank).*", #
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        debug_vis=True,
    )

    def __post_init__(self):
        print("[LegSceneCfg] __post_init__ start")
        print(f"[LegSceneCfg] num_envs={self.num_envs}, env_spacing={self.env_spacing}")
        print(f"[LegSceneCfg] robot prim_path={self.robot.prim_path}")
        print("[LegSceneCfg] __post_init__ end")


# --------------------------------------------------------------------------- #
#  Actions
# --------------------------------------------------------------------------- #
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FLHAA", "FLHIP", "FLKNEE",
            "FRHAA", "FRHIP", "FRKNEE",
            "RLHAA", "RLHIP", "RLKNEE",
            "RRHAA", "RRHIP", "RRKNEE",
        ],
        scale={
            ".*HAA": 0.125,  # HipX (HAA) 관절: 0.125
            ".*HIP": 0.25,   # HipY (HIP) 관절: 0.25
            ".*KNEE": 0.25,  # Knee 관절: 0.25
        },
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )


# --------------------------------------------------------------------------- #
#  Observations
# --------------------------------------------------------------------------- #
@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FLHAA", "FLHIP", "FLKNEE",
                    "FRHAA", "FRHIP", "FRKNEE",
                    "RLHAA", "RLHIP", "RLKNEE",
                    "RRHAA", "RRHIP", "RRKNEE",
                ],
            )},
            scale=1.0,
        )

        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FLHAA", "FLHIP", "FLKNEE",
                    "FRHAA", "FRHIP", "FRKNEE",
                    "RLHAA", "RLHIP", "RLKNEE",
                    "RRHAA", "RRHIP", "RRKNEE",
                ],
            )},
            scale=0.05,
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            scale=0.25,
        )

        base_z_pos = ObsTerm(
            func=mdp.base_pos_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        actions = ObsTerm(
            func=mdp.last_action,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# --------------------------------------------------------------------------- #
#  Commands
# --------------------------------------------------------------------------- #
@configclass
class CommandsCfg:
    # RL 설정과 동일하게: 소수 환경은 정지 명령, 전 환경에서 heading 명령 활성화
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10, 10),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

# --------------------------------------------------------------------------- #
#  Events (resets)
# --------------------------------------------------------------------------- #
@configclass
class EventCfg:
    reset_body = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FLHAA", "FLHIP", "FLKNEE",
                    "FRHAA", "FRHIP", "FRKNEE",
                    "RLHAA", "RLHIP", "RLKNEE",
                    "RRHAA", "RRHIP", "RRKNEE",
                ],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


# --------------------------------------------------------------------------- #
#  RewardsCfg : legged_gym scales 매핑 (dt 스케일 전)
# --------------------------------------------------------------------------- #
@configclass
class RewardsCfg:
    """legged_gym LeggedRobotCfg.rewards.scales 를 Isaac Lab 스타일로 매핑"""

    termination = RewTerm(
        func=mdp.rew_termination,
        weight=0.0,
    )

    # alive = RewTerm(
    #     func=mdp.rew_alive,
    #     weight=0.01,
    # )

    tracking_lin_vel = RewTerm(
        func=mdp.rew_tracking_lin_vel,
        weight=3.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.5),  
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    tracking_ang_vel = RewTerm(
        func=mdp.rew_tracking_ang_vel,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.5),  
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    lin_vel_z = RewTerm(
        func=mdp.rew_lin_vel_z,
        weight=-4.0,  ##next height reward 값 조절
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    ang_vel_xy = RewTerm(
        func=mdp.rew_ang_vel_xy,
        weight=-0.05, #-0.2
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    orientation = RewTerm(
        func=mdp.rew_orientation,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    base_height = RewTerm(
        func=mdp.rew_base_height,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.3836, #3536   qqqq
        },
    )

    torques = RewTerm(
        func=mdp.rew_torques,
        weight=-2.5e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FLHAA", "FLHIP", "FLKNEE",
                    "FRHAA", "FRHIP", "FRKNEE",
                    "RLHAA", "RLHIP", "RLKNEE",
                    "RRHAA", "RRHIP", "RRKNEE",
                ],
            ),
        },
    )

    dof_vel = RewTerm(
        func=mdp.rew_dof_vel,
        weight= 0.00,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FLHAA", "FLHIP", "FLKNEE",
                    "FRHAA", "FRHIP", "FRKNEE",
                    "RLHAA", "RLHIP", "RLKNEE",
                    "RRHAA", "RRHIP", "RRKNEE",
                ],
            ),
        },
    )

    dof_acc = RewTerm(
        func=mdp.rew_dof_acc,
        weight=-1e-8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FLHAA", "FLHIP", "FLKNEE",
                    "FRHAA", "FRHIP", "FRKNEE",
                    "RLHAA", "RLHIP", "RLKNEE",
                    "RRHAA", "RRHIP", "RRKNEE",
                ],
            ),
        },
    )

    joint_power = RewTerm(
        func=mdp.rew_joint_power,
        weight=-2e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FLHAA", "FLHIP", "FLKNEE",
                    "FRHAA", "FRHIP", "FRKNEE",
                    "RLHAA", "RLHIP", "RLKNEE",
                    "RRHAA", "RRHIP", "RRKNEE",
                ],
            ),
        },
    )

    action_rate = RewTerm(
        func=mdp.rew_action_rate,
        weight=-0.02,
    )

    dof_pos_limits = RewTerm(
        func=mdp.rew_dof_pos_limits,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FLHAA", "FLHIP", "FLKNEE",
                    "FRHAA", "FRHIP", "FRKNEE",
                    "RLHAA", "RLHIP", "RLKNEE",
                    "RRHAA", "RRHIP", "RRKNEE",
                ],
            ),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.rew_feet_air_time,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contact_sensor", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    feet_air_time_variance = RewTerm(
        func=mdp.rew_feet_air_time_variance,
        weight=-8.0,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contact_sensor", body_names=[".*foot.*"]),
        },
    )

    feet_gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.5, 
        params={
            "std": math.sqrt(0.5),  # exponential kernel의 표준편차
            "command_name": "base_velocity",
            "max_err": 0.2,  # 최대 오차 클리핑
            "velocity_threshold": 0.5,  # body velocity 
            "command_threshold": 0.1,  # command velocity 
            # synced_feet_pair_names: 동기화할 발 쌍 (trot 보행: 대각선 발 쌍)
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")), 
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("feet_contact_sensor"),
        },
    )

    foot_contact_forces = RewTerm(
        func=mdp.rew_foot_contact_forces,
        weight=-0.1,
        params={
            "threshold": 200.0,
            "sensor_cfg": SceneEntityCfg("feet_contact_sensor", body_names=[".*foot.*"]),  # rl_training과 동일: 발 접촉 힘만 측정
        },
    )

    undesired_contacts = RewTerm(
        func=isaaclab_mdp.undesired_contacts,
        weight=-0.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("link_collision_sensor", body_names=["^(?!.*foot).*"]),  # rl_training과 동일: foot이 아닌 모든 링크
        },
    )

    feet_stumble = RewTerm(
        func=mdp.rew_feet_stumble,
        weight=-0.0,
    )

    stand_still = RewTerm(
        func=mdp.rew_stand_still,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FLHAA", "FLHIP", "FLKNEE",
                    "FRHAA", "FRHIP", "FRKNEE",
                    "RLHAA", "RLHIP", "RLKNEE",
                    "RRHAA", "RRHIP", "RRKNEE",
                ],
            ),
        },
    )

    feet_slide = RewTerm(
        func=mdp.rew_feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contact_sensor"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )

    feet_height = RewTerm(
        func=mdp.rew_feet_height,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
            "target_height": 0.05, 
            "tanh_mult": 2.0,
            "command_name": "base_velocity",
        },
    )

    feet_height_body = RewTerm(
        func=mdp.rew_feet_height_body,
        weight=-2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
            "target_height": -0.3536,
            "tanh_mult": 2.0,
            "command_name": "base_velocity",
        },
    )

    joint_mirror = RewTerm(
        func=mdp.rew_joint_mirror,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FL.*", "RR.*"],
                ["FR.*", "RL.*"],
            ],
        },
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.rew_feet_contact_without_cmd,
        weight=0.1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("feet_contact_sensor"),
        },
    )

    joint_deviation_l1 = RewTerm(
        func=mdp.rew_joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FLHAA", "FRHAA", "RLHAA", "RRHAA",  # HipX만 (rough task와 동일)
                ],
            ),
        },
    )


# --------------------------------------------------------------------------- #
#  Terminations
# --------------------------------------------------------------------------- #
@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    body_height_oob = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": 0.15,
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": math.pi / 6, ##6에서 바꿈
        },
    )


# --------------------------------------------------------------------------- #
#  Final EnvCfg
# --------------------------------------------------------------------------- #
@configclass
class Quad_EnvCfg(ManagerBasedRLEnvCfg):
    scene: LegSceneCfg = LegSceneCfg(num_envs=1, env_spacing=2.0, clone_in_fabric=False)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and hasattr(reward_attr, 'weight') and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)

    def __post_init__(self) -> None:
        print("[Quad_EnvCfg] __post_init__ start")

        # simulation / control 설정
        self.decimation = 4
        self.episode_length_s = 20

        self.viewer.eye = (8.0, 0.0, 5.0)

        self.sim.dt = 0.005  # rl_training과 동일하게 맞춤 (200 Hz)
        self.sim.render_interval = self.decimation

        # ------------------------------
        # legged_gym 스타일: weight * dt
        # ------------------------------
        # ctrl_dt = self.decimation * self.sim.dt

        # for name, term in vars(self.rewards).items():
        #     if isinstance(term, RewTerm):
        #         term.weight *= ctrl_dt

        # If the weight of rewards is 0, set rewards to None
        self.disable_zero_weight_rewards()

        print(f"[Quad_EnvCfg] sim.dt={self.sim.dt}, render_interval={self.sim.render_interval}")
        # print(f"[Quad_EnvCfg] ctrl_dt={ctrl_dt}")
        print(f"[Quad_EnvCfg] scene.num_envs={self.scene.num_envs}, device={self.sim.device}")
        print("[Quad_EnvCfg] __post_init__ end")