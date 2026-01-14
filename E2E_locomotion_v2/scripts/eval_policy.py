import argparse
import sys
import traceback
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from isaaclab.app import AppLauncher

# ----------------- 1) CLI 설정 -----------------
parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy on the legbot env.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved model_*.pt file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs to run during evaluation.")
parser.add_argument(
    "--rollout_steps",
    type=int,
    default=0,
    help="Number of control steps to run. Set <= 0 to run until the window is closed.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="(eval 전용) episode length in seconds. If not set, uses env default.",
)
AppLauncher.add_app_launcher_args(parser)  # --headless, --device 등 추가
args = parser.parse_args()

# ----------------- 2) Isaac Sim 부팅 -----------------
simulation_app = AppLauncher(args).app

# ----------------- 3) Isaac Lab / RSL-RL 임포트 (앱 생성 이후) -----------------
import torch
from isaaclab.envs import ManagerBasedRLEnv

# NOTE:
# Training was done with E2E_locomotion_v2.env_cfg.Quad_EnvCfg
# (obs dim = 46, including velocity_commands).
# To load that checkpoint correctly, eval must use the same env cfg.
from E2E_locomotion_v2.env_cfg import Quad_EnvCfg

# RSL-RL VecEnv 래퍼 (train_KJH.py와 동일한 fallback 구조)
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    print("[eval] RslRlVecEnvWrapper: isaaclab_rl.rsl_rl ✓")
except Exception:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    print("[eval] RslRlVecEnvWrapper: omni.isaac.lab_tasks.utils.wrappers.rsl_rl ✓")

# rsl-rl-lib 3.1.x API (config dict 기반)
try:
    from rsl_rl.runners import OnPolicyRunner
except Exception:
    from rsl_rl_lib.runners import OnPolicyRunner


def log(msg: str) -> None:
    """간단한 로거."""
    print(f"[eval] {msg}", flush=True)


def _to_device(obs, device):
    """관측치를 안전하게 device로 이동 (Tensor / dict 모두 지원)."""
    import torch

    if isinstance(obs, torch.Tensor):
        return obs.to(device)
    if isinstance(obs, dict):
        return {k: _to_device(v, device) for k, v in obs.items()}
    if isinstance(obs, (list, tuple)):
        return type(obs)(_to_device(v, device) for v in obs)
    return obs


# ----------------- 4) 환경 생성 함수 (train과 동일 구조) -----------------
def make_env():
    cfg = Quad_EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device  # AppLauncher에서 넣어준 device (cuda/cpu)
    # eval 전용으로 episode length를 재설정할 수 있게 허용
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    base_env = ManagerBasedRLEnv(cfg=cfg)
    return RslRlVecEnvWrapper(base_env)  # 반드시 마지막 래퍼


# ----------------- 5) 학습 설정 dict (train과 동일 구조) -----------------
def build_train_cfg() -> dict:
    """OnPolicyRunner가 policy를 구성할 때 쓸 config.

    - train_KJH.py와 동일한 구조/하이퍼파라미터로 맞춰야
      checkpoint를 정확히 로드할 수 있음.
    - max_iterations / save_interval / num_steps_per_env 값은
      여기서는 실제로 사용되진 않지만, Runner 생성에 필요해서 채워준다.
    """
    return {
        "num_steps_per_env": 1,   # eval에서는 사용 안 되지만 필수 필드
        "save_interval": 1,
        "max_iterations": 1,
        "logger": "tensorboard",
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "value_loss_coef": 2.0,
            "entropy_coef": 0.0,
            "learning_rate": 3e-4,
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


env = None
try:
    # ----------------- 6) 환경 생성 -----------------
    log("Constructing evaluation environments")
    env = make_env()
    obs_sample = env.get_observations()
    obs_desc = {k: v.shape for k, v in obs_sample.items()}
    log(f"Observation groups: {obs_desc}")
    log(f"Action dim: {env.num_actions}")

    # ----------------- 7) 러너/정책 초기화 -----------------
    train_cfg = build_train_cfg()
    log("Initializing OnPolicyRunner for inference")
    runner = OnPolicyRunner(
        env=env,
        train_cfg=deepcopy(train_cfg),
        log_dir=None,           # eval에서는 로그 디렉토리 불필요
        device=args.device,
    )

    log(f"Loading checkpoint: {args.checkpoint}")
    runner.load(args.checkpoint, load_optimizer=False, map_location=args.device)
    log("Building inference policy")
    policy = runner.get_inference_policy(device=args.device)

    # ----------------- 8) 화살표 시각화 활성화 -----------------
    # UniformVelocityCommand에 내장된 화살표 시각화 기능 활성화
    # 초록색 화살표 = Command velocity, 파란색 화살표 = Actual velocity
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    if hasattr(base_env, 'command_manager'):
        base_env.command_manager.set_debug_vis(True)
        log("Velocity arrows enabled: Green = Command, Blue = Actual")
    else:
        log("Warning: command_manager not found, velocity arrows disabled")

    # ----------------- 9) 환경 리셋 -----------------
    log("Resetting environments")
    reset_out = env.reset()
    # reset()이 obs만 리턴인지, (obs, info) 튜플인지 버전에 따라 다를 수 있으니 안전하게 처리
    if isinstance(reset_out, tuple):
        obs, _ = reset_out
    else:
        obs = reset_out
    obs = _to_device(obs, args.device)
    log("Environment reset complete")

    # ----------------- 9) 롤아웃 루프 -----------------
    total_steps = 0
    target_steps = "infinite" if args.rollout_steps <= 0 else str(args.rollout_steps)
    msg = f"Running policy from {args.checkpoint} for {target_steps} steps."
    if args.rollout_steps <= 0:
        msg += " Close the viewer window to stop."
    log(msg)

    while simulation_app.is_running() and (
        args.rollout_steps <= 0 or total_steps < args.rollout_steps
    ):
        # Isaac Sim 업데이트
        simulation_app.update()

        with torch.inference_mode():
            # 정책 호출
            actions = policy(obs)
            # 환경 스텝
            obs, rew, dones, info = env.step(actions.to(env.device))
            obs = _to_device(obs, args.device)

        if total_steps % 100 == 0:
            mean_rew = rew.mean().item()
            term_fraction = dones.float().mean().item()
            
            log(
                f"step={total_steps:06d}  "
                f"mean_rew={mean_rew:.3f}  "
                f"terminated_frac={term_fraction:.3f}"
            )
            
        total_steps += 1

except Exception:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()


# 사용 예시:
# python scripts/eval_policy.py \
#   --checkpoint runs/legbot_rslrl/2026-01-04_17-55-58/model_25500.pt \
#   --num_envs 200 \
#   --rollout_steps 0 \
#   --device cuda
