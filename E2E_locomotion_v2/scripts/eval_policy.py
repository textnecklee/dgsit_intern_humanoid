import argparse
import sys
import traceback
from copy import deepcopy
from pathlib import Path

# [추가됨] 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from isaaclab.app import AppLauncher

# ----------------- 1) CLI 설정 -----------------
parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy on the legbot env.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved model_*.pt file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs to run during evaluation.")
parser.add_argument("--rollout_steps", type=int, default=0, help="Number of control steps to run.")
parser.add_argument("--episode_length_s", type=float, default=None, help="(eval 전용) episode length in seconds.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ----------------- 2) Isaac Sim 부팅 -----------------
simulation_app = AppLauncher(args).app

# ----------------- 3) Isaac Lab / RSL-RL 임포트 -----------------
import torch
from isaaclab.envs import ManagerBasedRLEnv
from E2E_locomotion_v2.env_cfg import Quad_EnvCfg

# RSL-RL VecEnv 래퍼
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    print("[eval] RslRlVecEnvWrapper: isaaclab_rl.rsl_rl ✓")
except Exception:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    print("[eval] RslRlVecEnvWrapper: omni.isaac.lab_tasks.utils.wrappers.rsl_rl ✓")

# rsl-rl-lib runner
try:
    from rsl_rl.runners import OnPolicyRunner
except Exception:
    from rsl_rl_lib.runners import OnPolicyRunner

def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)

def _to_device(obs, device):
    import torch
    if isinstance(obs, torch.Tensor): return obs.to(device)
    if isinstance(obs, dict): return {k: _to_device(v, device) for k, v in obs.items()}
    if isinstance(obs, (list, tuple)): return type(obs)(_to_device(v, device) for v in obs)
    return obs

# ----------------- 4) 환경 생성 함수 -----------------
def make_env():
    cfg = Quad_EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    base_env = ManagerBasedRLEnv(cfg=cfg)
    return RslRlVecEnvWrapper(base_env)

# ----------------- 5) 학습 설정 dict -----------------
def build_train_cfg() -> dict:
    return {
        "num_steps_per_env": 1, "save_interval": 1, "max_iterations": 1, "logger": "tensorboard",
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "algorithm": {"class_name": "PPO", "clip_param": 0.2, "num_learning_epochs": 5, "num_mini_batches": 4, "value_loss_coef": 2.0, "entropy_coef": 0.0, "learning_rate": 3e-4, "gamma": 0.99, "lam": 0.95, "max_grad_norm": 1.0},
        "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [256, 256], "critic_hidden_dims": [256, 256], "activation": "elu", "actor_obs_normalization": True, "critic_obs_normalization": True, "init_noise_std": 1.0},
    }

env = None
try:
    # ----------------- 6) 환경 및 정책 초기화 -----------------
    log("Constructing evaluation environments")
    env = make_env()
    
    # [중요] 센서 및 로봇 데이터 접근을 위해 base_env 확보
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env

    # [평가 전용] 전진 속도 1.0 고정 (y, yaw=0)
    if hasattr(base_env, "command_manager"):
        try:
            cmd_term = base_env.command_manager.get_term("base_velocity")
            cmd_term.cfg.ranges.lin_vel_x = (0.5, 0.5)
            cmd_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
            cmd_term.cfg.ranges.ang_vel_z = (0.0, 0.0)
            log("Command fixed: lin_vel_x=1.0, lin_vel_y=0.0, ang_vel_z=0.0")
        except Exception as e:
            log(f"Warning: failed to fix command ranges: {e}")

    # [추가] 센서 확인 로직
    # 보통 config에서 'contact_forces'라는 이름으로 정의됩니다.
    SENSOR_NAME = "feet_contact_sensor"
    has_contact_sensor = False
    
    if hasattr(base_env.scene, "sensors") and SENSOR_NAME in base_env.scene.sensors:
        has_contact_sensor = True
        log(f"Sensor found: '{SENSOR_NAME}' - Contact visualization enabled.")
    else:
        available_sensors = list(base_env.scene.sensors.keys()) if hasattr(base_env.scene, "sensors") else []
        log(f"Warning: Sensor '{SENSOR_NAME}' not found. Available: {available_sensors}")
        log("Only Base Height will be visualized.")

    train_cfg = build_train_cfg()
    log("Initializing OnPolicyRunner")
    runner = OnPolicyRunner(env=env, train_cfg=deepcopy(train_cfg), log_dir=None, device=args.device)

    log(f"Loading checkpoint: {args.checkpoint}")
    runner.load(args.checkpoint, load_optimizer=False, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    # ----------------- 8) 화살표 시각화 -----------------
    if hasattr(base_env, 'command_manager'):
        base_env.command_manager.set_debug_vis(True)
        log("Velocity arrows enabled")

    # ----------------- 9) 리셋 -----------------
    log("Resetting environments")
    reset_out = env.reset()
    if isinstance(reset_out, tuple): obs, _ = reset_out
    else: obs = reset_out
    obs = _to_device(obs, args.device)

    # ================= [시각화 데이터 초기화] =================
# ================= [시각화 데이터 초기화] =================
    height_history = []
    contact_history = [[], [], [], []]
    step_history = []
    
    PLOT_INTERVAL = 2
    CONTACT_THRESHOLD = 10.0
    MAX_HISTORY = 100

    # [수정] 로봇 이름 자동 탐색 변수
    target_robot_name = "robot" 
    robot_name_found = False
    contact_body_indices = None

    total_steps = 0
    target_steps = "infinite" if args.rollout_steps <= 0 else str(args.rollout_steps)
    log(f"Running policy for {target_steps} steps.")

    while simulation_app.is_running() and (args.rollout_steps <= 0 or total_steps < args.rollout_steps):
        simulation_app.update()

        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, info = env.step(actions.to(env.device))
            obs = _to_device(obs, args.device)

            # ----------------- [데이터 수집 (수정됨)] -----------------
            # 1. 로봇 이름 확인 및 Height 수집
            current_height = 0.0 # 기본값 (로봇 못 찾을 경우 0.0)

            # (1) 로봇 이름이 확인되지 않았다면 탐색
            if not robot_name_found:
                # scene에 있는 모든 rigid object 키를 가져옴
                available_keys = list(base_env.scene.articulations.keys())
                
                if target_robot_name in available_keys:
                    robot_name_found = True
                    log(f"DEBUG: Found robot named '{target_robot_name}'")
                elif len(available_keys) > 0:
                    # 'robot'이 없으면 첫 번째 객체를 로봇으로 가정하고 사용
                    target_robot_name = available_keys[0]
                    robot_name_found = True
                    log(f"DEBUG: 'robot' key not found. Using '{target_robot_name}' instead. (Available: {available_keys})")
                else:
                    if total_steps == 0: log(f"DEBUG: No rigid objects found in scene!")

            # (2) 데이터 수집
            if robot_name_found and target_robot_name in base_env.scene.articulations:
                robot = base_env.scene.articulations[target_robot_name]
                current_height = robot.data.root_pos_w[0, 2].item()
            
            # [중요] 로봇을 찾든 못 찾든 리스트에 추가해야 개수가 맞음 (Crash 방지)
            height_history.append(current_height)
            
            # 2. Contact Sensor
            if has_contact_sensor:
                sensor = base_env.scene.sensors[SENSOR_NAME]
                forces = sensor.data.net_forces_w[0]
                force_norms = torch.norm(forces, dim=-1)
                # Resolve body indices once using explicit foot names
                if contact_body_indices is None:
                    log(f"Contact sensor body_names: {sensor.body_names}")
                    name_to_idx = {name: i for i, name in enumerate(sensor.body_names)}
                    desired_names = ["FL_foot", "RR_foot", "FR_foot", "RL_foot"]
                    contact_body_indices = [name_to_idx.get(n, None) for n in desired_names]
                    log(f"Contact body indices (FL, RR, FR, RL): {contact_body_indices}")

                for i, body_idx in enumerate(contact_body_indices):
                    val = 0
                    if body_idx is not None and body_idx < len(force_norms):
                        val = 1 if (force_norms[body_idx].item() > CONTACT_THRESHOLD) else 0
                    contact_history[i].append(val)
            else:
                # 센서 없어도 리스트는 채워야 함 (Crash 방지)
                for i in range(4): contact_history[i].append(0)
            
            step_history.append(total_steps)

            # 데이터 길이 제한
            if len(step_history) > MAX_HISTORY:
                step_history.pop(0)
                height_history.pop(0)
                for i in range(4): contact_history[i].pop(0)
            # ----------------------------------------------------------

        # ----------------- [그래프 그리기] -----------------
        # 데이터 개수가 2개 이상일 때만 그리기
        if total_steps % PLOT_INTERVAL == 0 and len(step_history) > 1:
            try:
                plt.figure(1, figsize=(5, 6))
                plt.clf()

                # [상단] Base Height
                ax1 = plt.subplot(2, 1, 1)
                ax1.plot(step_history, height_history, 'b-', linewidth=1.5, label='Height')
                ax1.set_title(f"Base Height (Robot: {target_robot_name})")
                ax1.grid(True, alpha=0.3)

                # [하단] Foot Contact
                ax2 = plt.subplot(2, 1, 2, sharex=ax1)
                colors = ['r', 'g', 'b', 'orange']
                labels = ['FL', 'RR', 'FR', 'RL']
                
                for i in range(4):
                    if len(contact_history[i]) == len(step_history): # 개수 안전장치
                        offset_data = [val * 0.8 + (i * 1.5) for val in contact_history[i]]
                        ax2.fill_between(step_history, [i * 1.5]*len(step_history), offset_data, 
                                         color=colors[i], alpha=0.6, label=labels[i])
                
                ax2.set_yticks([0.4, 1.9, 3.4, 4.9])
                ax2.set_yticklabels(labels)
                ax2.set_title("Foot Contact")
                ax2.set_xlabel("Steps")
                ax2.grid(True, axis='x', alpha=0.3)

                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.001)
            except Exception as e:
                # 그래프 그리다 에러나도 시뮬레이션은 안 멈추게 처리
                print(f"Plotting Error: {e}")

        if total_steps % 100 == 0:
            mean_rew = rew.mean().item()
            term_fraction = dones.float().mean().item()
            log(f"step={total_steps:06d} | mean_rew={mean_rew:.3f} | term={term_fraction:.3f}")
            
        total_steps += 1

except Exception:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()
