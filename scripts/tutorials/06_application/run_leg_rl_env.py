# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run RL env for 3-DoF leg robot.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from leg_env_cfg import LegEnvCfg  # <- 당신이 만든 cfg 모듈



def main():
    """Main function."""
    # create environment configuration
    env_cfg = LegEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            steps_per_ep = int(round(env.cfg.episode_length_s / (env.cfg.sim.dt * env.cfg.decimation)))
            if count % steps_per_ep == 0:
            # if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # q1, q2 토크 액션 두 개. shape은 env.action_manager.action와 동일.
            joint_efforts = torch.randn_like(env.action_manager.action)

            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)

            # 관측 순서: [slider, q1, q2, slider_vel, q1_vel, q2_vel]
            # q1 = obs["policy"][0][1].item()
            # q2 = obs["policy"][0][2].item()
            # slider_z = obs["policy"][0][0].item()
            # print(f"[Env 0] slider_z={slider_z:.3f}, q1={q1:.3f}, q2={q2:.3f}")

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()