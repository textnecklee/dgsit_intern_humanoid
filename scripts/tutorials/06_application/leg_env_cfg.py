# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

# ───────────────────────────────────────────────────────────────────────────────
# 1) 레그 로봇 Asset 구성 (USD 경로와 관절명은 실제 모델에 맞춰 둡니다)
#    - 슬라이더(base_slider_z)는 액션 제외. 필요 시 강한 드라이브로 0 근처 유지.
# ───────────────────────────────────────────────────────────────────────────────

#! sim_utils로 중력이나, colisions, 
LEGBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/teamquad/Desktop/JSHong/IsaacLab/assets/one_leg/one_leg.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,  # TGS 경고 회피용
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "base_slider_z": 0.0,
            "q1": 0,
            "q2": 0,
        },
        pos=(0, 0, 0),
    ),
    actuators={
        # 슬라이더는 액션에서 제외. 드라이브로 0 근처에 묶음(사실상 고정 효과).
        "slider_lock": ImplicitActuatorCfg(
            joint_names_expr=["base_slider_z"],
            stiffness=0, damping=100,
            effort_limit_sim=100, velocity_limit_sim=100
        ),
        # q1, q2만 제어 대상으로 둠
        "q1_act": ImplicitActuatorCfg(
            joint_names_expr=["q1"],
            stiffness=0, damping=1,
            effort_limit_sim=100.0, velocity_limit_sim=100.0
        ),
        "q2_act": ImplicitActuatorCfg(
            joint_names_expr=["q2"],
            stiffness=0, damping=1,
            effort_limit_sim=100.0, velocity_limit_sim=100.0
        ),
    },
)


# ───────────────────────────────────────────────────────────────────────────────
# 2) 씬 구성
# ───────────────────────────────────────────────────────────────────────────────

@configclass
class LegSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    robot: ArticulationCfg = LEGBOT_CFG
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# ───────────────────────────────────────────────────────────────────────────────
# 3) MDP: Actions / Observations / Events / Rewards / Terminations
# ───────────────────────────────────────────────────────────────────────────────

@configclass
class ActionsCfg:
    """Action specs: q1, q2 토크 제어. 슬라이더는 액션에서 제외."""
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["q1", "q2"],
        scale=1.0,
    )

@configclass
class ObservationsCfg:
    """Observation specs: slider, q1, q2의 상대 위치/속도."""

    @configclass
    class PolicyCfg(ObsGroup):
        # 관절 순서 고정: [base_slider_z, q1, q2]

        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["base_slider_z", "q1", "q2"])},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["base_slider_z", "q1", "q2"])},
        )
        actions = ObsTerm(
            func=mdp.last_action,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Reset 초기화: 슬라이더 0 근처, q1/q2 소각도 범위."""
    reset_slider = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["base_slider_z"]),
            "position_range": (0.3, 0.3),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["q1", "q2"]),
            "position_range": (math.pi/6, 2*math.pi/3),
            "velocity_range": (0.0, 0.0),
        },
    )
    
@configclass
class RewardsCfg:
    """Reward shaping for standing/bending leg."""
    # 생존 보상
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # 종료 패널티
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # 서있기 유지
    slider_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-50.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["base_slider_z"]), "target": 0.25},
    )

    q1_joint_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-150.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["q1"]), "target": math.pi/3},
    )   
    q2_joint_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-150.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["q2"]), "target": 2*math.pi/3},
    )

    # 속도 감쇠
    joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["base_slider_z", "q1", "q2"])},
    )


@configclass
class TerminationsCfg:
    """Termination conditions."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 슬라이더 z 경계
    slider_oob = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["base_slider_z"]), "bounds": (0.2, 3.0)},
    )
    # 관절 각 경계
    joints_oob = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["q1", "q2"]), "bounds": (0, math.pi)},
    )


# ───────────────────────────────────────────────────────────────────────────────
# 4) 환경 구성
# ───────────────────────────────────────────────────────────────────────────────

@configclass
class LegEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: LegSceneCfg = LegSceneCfg(num_envs=32, env_spacing=1.0, clone_in_fabric=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation