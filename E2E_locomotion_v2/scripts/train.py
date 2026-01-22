import argparse
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
sys.path.append(str(Path(__file__).resolve().parents[2]))

from isaaclab.app import AppLauncher

# ----------------- 1) CLI 설정 -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--total_steps", type=int, default=100_000_000_0)
parser.add_argument("--rollout_len", type=int, default=24)  # num_steps_per_env
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional path to an existing model_*.pt checkpoint to continue training from.",
)
AppLauncher.add_app_launcher_args(parser)  # --headless, --device 등 추가
args = parser.parse_args()

simulation_app = AppLauncher(args).app

from isaaclab.envs import ManagerBasedRLEnv
from E2E_locomotion_v2.env_cfg import Quad_EnvCfg

# RSL-RL VecEnv 래퍼
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    print("RslRlVecEnvWrapper: isaaclab_rl.rsl_rl ✓")
except Exception:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    print("RslRlVecEnvWrapper: omni.isaac.lab_tasks.utils.wrappers.rsl_rl ✓")

# rsl-rl-lib 3.1.x API (config dict 기반)
try:
    from rsl_rl.runners import OnPolicyRunner
except Exception:
    from rsl_rl_lib.runners import OnPolicyRunner


def make_env():
    print("[make_env] creating Quad_EnvCfg")
    cfg = Quad_EnvCfg()
    print("[make_env] Quad_EnvCfg created")

    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    print(f"[make_env] cfg updated: num_envs={cfg.scene.num_envs}, device={cfg.sim.device}")

    from isaaclab.envs import ManagerBasedRLEnv  # 혹시 모를 import 순서 문제 방지용

    print("[make_env] before ManagerBasedRLEnv")
    base_env = ManagerBasedRLEnv(cfg=cfg)

    #####! Debugging
    sensor = base_env.scene["feet_contact_sensor"]
    contact_time = sensor.data.current_contact_time  # [num_env, num_feet]

    base_env.prev_is_contact = torch.zeros_like(
        contact_time,
        dtype=torch.bool,
        device=contact_time.device,
    )
    print("[make_env] initialized prev_is_contact with shape", contact_time.shape)

    #####! Debugging

    print("[make_env] after ManagerBasedRLEnv")
    wrapped_env = RslRlVecEnvWrapper(base_env)
    print("[make_env] after wrapper")
    return wrapped_env


def build_train_cfg(max_iters: int) -> dict:
    """RSL-RL OnPolicyRunner가 기대하는 형식의 설정 dict 생성."""
    return {
        "num_steps_per_env": args.rollout_len,
        "save_interval": max(1, args.save_interval),
        "max_iterations": max_iters,
        "logger": "tensorboard",
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,  #6
            "value_loss_coef": 2.0,
            "entropy_coef": 0, #0.01
            "learning_rate": 3e-4,
            # "schedule": "adaptive",
            # "desired_kl": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
            "actor_obs_normalization": True,
            "critic_obs_normalization": True,
            "init_noise_std": 1.0,
        },
    }

def main():
    """Main training loop."""
    env = None
    try:
        env = make_env()
        obse_sample = env.get_observations()
        obs_desc = {k: v.shape for k, v in obse_sample.items()}
        print(f"[env] observation groups: {obs_desc}")
        print(f"[env] action_dim = {env.num_actions}")

        max_iters = max(1, args.total_steps // (args.num_envs * args.rollout_len))
        train_cfg = build_train_cfg(max_iters  )

        run_root = Path("runs/legbot_rslrl")
        timestamp_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = run_root / timestamp_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"[train] Logging to {log_dir.resolve()}")

        runner = OnPolicyRunner(
            env=env,
            train_cfg=deepcopy(train_cfg),
            log_dir=str(log_dir),
            device=args.device,
        )
        print(f"[train] OnPolicyRunner created successfully")

        # 기존 체크포인트에서 이어서 학습 (옵션)
        if args.checkpoint is not None:
            print(f"[train] Loading checkpoint from {args.checkpoint}")
            runner.load(args.checkpoint, load_optimizer=True, map_location=args.device)

        # ----------------- 6) 학습 -----------------
        print(f"[train] Starting PPO for {max_iters} iterations...")
        runner.learn(num_learning_iterations=max_iters)

    except Exception as e:
        print(f"[train] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

# tensorboard --logdir runs/legbot_rslrl --port 6006

#! 이어서 학습
# python scripts/train.py \
#   --num_envs 4096 \
#   --total_steps 3000000000 \
#   --rollout_len 24 \
#   --save_interval 500 \
#   --checkpoint runs/legbot_rslrl/2026-01-04_20-43-13/model_56000.pt \
#   --device cuda \
#   --headless
