# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

LEGBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/teamquad/Desktop/JSHong/IsaacLab/assets/one_leg/one_leg.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "base_slider_z": 0.0,
            "q1": 0.0,
            "q2": 0.0,
        },
        pos=(0.0, 0.0, 0.0),  # Dofbot과 충돌 피하도록 위치 변경
    ),
    actuators={
        # 실제 조인트명으로 매핑. 3자유도 모두 구동 예시
        "slider_act": ImplicitActuatorCfg(
            joint_names_expr=["base_slider_z"], effort_limit_sim=200.0, velocity_limit_sim=2.0,
            stiffness=5000.0, damping=200.0
        ),
        "q1_act": ImplicitActuatorCfg(
            joint_names_expr=["q1"], effort_limit_sim=100.0, velocity_limit_sim=100.0,
            stiffness=10000.0, damping=100.0
        ),
        "q2_act": ImplicitActuatorCfg(
            joint_names_expr=["q2"], effort_limit_sim=100.0, velocity_limit_sim=100.0,
            stiffness=10000.0, damping=100.0
        ),
    },
)




class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Legbot = LEGBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Leg")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_legbot_state = scene["Legbot"].data.default_root_state.clone()
            root_legbot_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["Legbot"].write_root_pose_to_sim(root_legbot_state[:, :7])
            scene["Legbot"].write_root_velocity_to_sim(root_legbot_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Legbot"].data.default_joint_pos.clone(),
                scene["Legbot"].data.default_joint_vel.clone(),
            )
            scene["Legbot"].write_joint_state_to_sim(joint_pos, joint_vel)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting state...")

        # wave
        leg_action = scene["Legbot"].data.default_joint_pos
        # leg_action[:, 0] = 0.0   # base_slider_z
        # leg_action[:, 1] = 0.5   # q1
        # leg_action[:, 2] = -0.5  # q2
        leg_action[:, 0:3] = 0.1 * np.sin(2 * np.pi * 0.5 * sim_time)
        scene["Legbot"].set_joint_position_target(leg_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
