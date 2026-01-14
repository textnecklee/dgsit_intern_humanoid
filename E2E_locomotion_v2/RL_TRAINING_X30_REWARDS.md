# rl_training X30 (Deeprobotics Lite3) Rewards 정리

## 📊 활성화된 Rewards (Weight ≠ 0)

`rough_env_cfg.py`에서 실제로 사용되는 rewards만 정리했습니다.

---

## 1. 🎯 Task-Specific Rewards (속도 추적)

### 1.1 `track_lin_vel_xy_exp`
- **Weight**: `3.5`
- **함수**: `mdp.track_lin_vel_xy_exp`
- **파라미터**:
  - `command_name`: `"base_velocity"`
  - `std`: `math.sqrt(0.5)` ≈ `0.707`
- **의미**: XY 평면 선형 속도 추적 (Exponential kernel)
- **계산**: `exp(-||cmd_xy - vel_xy||² / std²)`

### 1.2 `track_ang_vel_z_exp`
- **Weight**: `1.5`
- **함수**: `mdp.track_ang_vel_z_exp`
- **파라미터**:
  - `command_name`: `"base_velocity"`
  - `std`: `math.sqrt(0.5)` ≈ `0.707`
- **의미**: Z축 각속도(요) 추적 (Exponential kernel)
- **계산**: `exp(-(cmd_yaw - vel_yaw)² / std²)`

---

## 2. 🦶 Feet-Related Rewards

### 2.1 `feet_air_time`
- **Weight**: `5.0`
- **함수**: `mdp.feet_air_time_including_ang_z`
- **파라미터**:
  - `command_name`: `"base_velocity"`
  - `threshold`: `0.5` (초)
  - `sensor_cfg.body_names`: `[".*_FOOT"]`
- **의미**: 발이 공중에 떠있는 시간 보상 (보행 시)
- **계산**: 발이 `threshold` 이상 공중에 있으면 보상

### 2.2 `feet_air_time_variance`
- **Weight**: `-8.0` ⚠️ (Penalty)
- **함수**: `mdp.feet_air_time_variance_penalty`
- **파라미터**:
  - `sensor_cfg.body_names`: `[".*_FOOT"]`
- **의미**: 발 공중 시간의 분산 패널티 (보행 패턴 일관성)

### 2.3 `feet_slide`
- **Weight**: `-0.05` ⚠️ (Penalty)
- **함수**: `mdp.feet_slide`
- **파라미터**:
  - `sensor_cfg.body_names`: `[".*_FOOT"]`
  - `asset_cfg.body_names`: `[".*_FOOT"]`
- **의미**: 발 미끄러짐 패널티

### 2.4 `feet_height`
- **Weight**: `-0.2` ⚠️ (Penalty)
- **함수**: `mdp.feet_height`
- **파라미터**:
  - `asset_cfg.body_names`: `[".*_FOOT"]`
  - `target_height`: `0.05` (m)
  - `command_name`: `"base_velocity"`
  - `tanh_mult`: `2.0`
- **의미**: 발 높이 패널티 (지면에서 너무 높으면 penalty)

### 2.5 `feet_height_body`
- **Weight**: `-2.5` ⚠️ (Penalty)
- **함수**: `mdp.feet_height_body`
- **파라미터**:
  - `asset_cfg.body_names`: `[".*_FOOT"]`
  - `target_height`: `-0.35` (m) (body frame 기준)
  - `command_name`: `"base_velocity"`
  - `tanh_mult`: `2.0`
- **의미**: 발 높이 패널티 (body frame 기준)

### 2.6 `feet_gait`
- **Weight**: `0.5`
- **함수**: `mdp.GaitReward`
- **파라미터**:
  - `std`: `math.sqrt(0.5)` ≈ `0.707`
  - `command_name`: `"base_velocity"`
  - `max_err`: `0.2`
  - `velocity_threshold`: `0.5`
  - `command_threshold`: `0.1`
  - `synced_feet_pair_names`: 
    ```python
    [
        ["FL_FOOT", "HR_FOOT"],  # 대각선 발 쌍 1
        ["FR_FOOT", "HL_FOOT"]   # 대각선 발 쌍 2
    ]
    ```
- **의미**: 보행 패턴 보상 (대각선 발 동기화)

### 2.7 `feet_contact_without_cmd`
- **Weight**: `0.1`
- **함수**: `mdp.feet_contact_without_cmd`
- **파라미터**:
  - `sensor_cfg.body_names`: `[".*_FOOT"]`
  - `command_name`: `"base_velocity"`
- **의미**: 명령 없을 때 발 접촉 보상 (정지 상태 유지)

---

## 3. 📐 Body/Root State Penalties

### 3.1 `base_height_l2`
- **Weight**: `-10.0` ⚠️ (큰 Penalty)
- **함수**: `mdp.base_height_l2`
- **파라미터**:
  - `asset_cfg.body_names`: `["TORSO"]`
  - `sensor_cfg`: `SceneEntityCfg("height_scanner_base")`
  - `target_height`: `0.35` (m)
- **의미**: Base 높이 패널티 (지형에서의 높이)
- **계산**: `(height - target_height)²`

### 3.2 `lin_vel_z_l2`
- **Weight**: `-2.0` ⚠️ (Penalty)
- **함수**: `mdp.lin_vel_z_l2`
- **의미**: Z축 선형 속도 패널티 (상하 움직임 제한)
- **계산**: `(vel_z)²`

### 3.3 `ang_vel_xy_l2`
- **Weight**: `-0.05` ⚠️ (Penalty)
- **함수**: `mdp.ang_vel_xy_l2`
- **의미**: XY축 각속도 패널티 (롤/피치 회전 제한)
- **계산**: `(ang_vel_x)² + (ang_vel_y)²`

### 3.4 `flat_orientation_l2`
- **Weight**: `-5.0` ⚠️ (Penalty)
- **함수**: `mdp.flat_orientation_l2`
- **의미**: 기울어진 자세 패널티 (수평 유지)
- **계산**: `projected_gravity_b[:2]²`

---

## 4. 🔧 Joint State Penalties

### 4.1 `joint_torques_l2`
- **Weight**: `-2.5e-5` ⚠️ (작은 Penalty)
- **함수**: `mdp.joint_torques_l2`
- **파라미터**:
  - `asset_cfg.joint_names`: `[".*"]` (모든 관절)
- **의미**: 관절 토크 패널티
- **계산**: `Σ(τ²)`

### 4.2 `joint_acc_l2`
- **Weight**: `-1e-8` ⚠️ (매우 작은 Penalty)
- **함수**: `mdp.joint_acc_l2`
- **파라미터**:
  - `asset_cfg.joint_names`: `[".*"]` (모든 관절)
- **의미**: 관절 가속도 패널티 (부드러운 움직임)
- **계산**: `Σ(acc²)`

### 4.3 `joint_deviation_l1`
- **Weight**: `-0.5` ⚠️ (Penalty)
- **함수**: `mdp.joint_deviation_l1`
- **파라미터**:
  - `asset_cfg.joint_names`: `[".*HipX.*"]` (HipX 관절만)
- **의미**: 기본 관절 위치에서 벗어남 패널티 (HipX만)
- **계산**: `Σ|pos - default_pos|`

### 4.4 `joint_power`
- **Weight**: `-2e-5` ⚠️ (작은 Penalty)
- **함수**: `mdp.joint_power`
- **파라미터**:
  - `asset_cfg.joint_names`: `[".*"]` (모든 관절)
- **의미**: 관절 전력 소비 패널티
- **계산**: `Σ|τ × ω|`

### 4.5 `joint_pos_limits`
- **Weight**: `-5.0` ⚠️ (큰 Penalty)
- **함수**: `mdp.joint_pos_limits`
- **파라미터**:
  - `asset_cfg.joint_names`: `[".*"]` (모든 관절)
- **의미**: 관절 위치 한계 패널티 (한계 초과 시)

### 4.6 `joint_mirror`
- **Weight**: `-0.05` ⚠️ (Penalty)
- **함수**: `mdp.joint_mirror`
- **파라미터**:
  - `mirror_joints`: 
    ```python
    [
        ["FL_(HipX|HipY|Knee).*", "HR_(HipX|HipY|Knee).*"],  # 대각선 대칭
        ["FR_(HipX|HipY|Knee).*", "HL_(HipX|HipY|Knee).*"]   # 대각선 대칭
    ]
    ```
- **의미**: 관절 대칭성 패널티 (대각선 발 대칭 유지)

---

## 5. 🎮 Action Penalties

### 5.1 `action_rate_l2`
- **Weight**: `-0.02` ⚠️ (Penalty)
- **함수**: `mdp.action_rate_l2`
- **의미**: 액션 변화율 패널티 (부드러운 제어)
- **계산**: `Σ(action - prev_action)²`

---

## 6. 👣 Contact-Related Rewards

### 6.1 `contact_forces`
- **Weight**: `-1e-1` = `-0.1` ⚠️ (Penalty)
- **함수**: `mdp.contact_forces`
- **파라미터**:
  - `sensor_cfg.body_names`: `[".*_FOOT"]`
  - `threshold`: `100.0` (N)
- **의미**: 접촉 힘 패널티 (과도한 힘 방지)

### 6.2 `undesired_contacts`
- **Weight**: `-0.5` ⚠️ (Penalty)
- **함수**: `mdp.undesired_contacts`
- **파라미터**:
  - `sensor_cfg.body_names`: `[f"^(?!.*{self.foot_link_name}).*"]` (발 제외한 모든 부위)
  - `threshold`: `1.0` (N)
- **의미**: 원하지 않는 접촉 패널티 (발 외 부위 접촉)

---

## 7. 🛑 Stand Still Reward

### 7.1 `stand_still`
- **Weight**: `-0.5` ⚠️ (Penalty)
- **함수**: `mdp.stand_still_joint_deviation_l1`
- **파라미터**:
  - `asset_cfg.joint_names`: `self.joint_names` (모든 관절)
  - `command_name`: `"base_velocity"`
  - `command_threshold`: `0.1`
- **의미**: 명령 없을 때 정지 상태 유지 패널티

---

## 📊 Rewards 요약표

| 카테고리 | Reward 이름 | Weight | 타입 | 중요도 |
|---------|------------|--------|------|--------|
| **속도 추적** | `track_lin_vel_xy_exp` | `3.5` | ✅ Reward | ⭐⭐⭐⭐⭐ |
| | `track_ang_vel_z_exp` | `1.5` | ✅ Reward | ⭐⭐⭐⭐ |
| **발 관련** | `feet_air_time` | `5.0` | ✅ Reward | ⭐⭐⭐⭐⭐ |
| | `feet_gait` | `0.5` | ✅ Reward | ⭐⭐⭐ |
| | `feet_contact_without_cmd` | `0.1` | ✅ Reward | ⭐⭐ |
| | `feet_air_time_variance` | `-8.0` | ⚠️ Penalty | ⭐⭐⭐⭐⭐ |
| | `feet_height_body` | `-2.5` | ⚠️ Penalty | ⭐⭐⭐⭐ |
| | `feet_slide` | `-0.05` | ⚠️ Penalty | ⭐⭐ |
| | `feet_height` | `-0.2` | ⚠️ Penalty | ⭐⭐ |
| **Body/Root** | `base_height_l2` | `-10.0` | ⚠️ Penalty | ⭐⭐⭐⭐⭐ |
| | `flat_orientation_l2` | `-5.0` | ⚠️ Penalty | ⭐⭐⭐⭐ |
| | `lin_vel_z_l2` | `-2.0` | ⚠️ Penalty | ⭐⭐⭐ |
| | `ang_vel_xy_l2` | `-0.05` | ⚠️ Penalty | ⭐⭐ |
| **관절** | `joint_pos_limits` | `-5.0` | ⚠️ Penalty | ⭐⭐⭐⭐ |
| | `joint_deviation_l1` | `-0.5` | ⚠️ Penalty | ⭐⭐⭐ |
| | `joint_mirror` | `-0.05` | ⚠️ Penalty | ⭐⭐ |
| | `joint_torques_l2` | `-2.5e-5` | ⚠️ Penalty | ⭐ |
| | `joint_power` | `-2e-5` | ⚠️ Penalty | ⭐ |
| | `joint_acc_l2` | `-1e-8` | ⚠️ Penalty | ⭐ |
| **액션** | `action_rate_l2` | `-0.02` | ⚠️ Penalty | ⭐⭐ |
| **접촉** | `undesired_contacts` | `-0.5` | ⚠️ Penalty | ⭐⭐⭐ |
| | `contact_forces` | `-0.1` | ⚠️ Penalty | ⭐⭐ |
| **정지** | `stand_still` | `-0.5` | ⚠️ Penalty | ⭐⭐⭐ |

---

## 🎯 주요 특징

### 1. 속도 추적이 핵심
- `track_lin_vel_xy_exp` (3.5): 가장 큰 보상
- `track_ang_vel_z_exp` (1.5): 각속도 추적

### 2. 발 보행 패턴 중요
- `feet_air_time` (5.0): 큰 보상
- `feet_air_time_variance` (-8.0): 가장 큰 패널티 (일관성)
- `feet_gait` (0.5): 대각선 발 동기화

### 3. 안정성 유지
- `base_height_l2` (-10.0): 매우 큰 패널티 (높이 유지)
- `flat_orientation_l2` (-5.0): 자세 유지
- `joint_pos_limits` (-5.0): 관절 한계 보호

### 4. 부드러운 움직임
- `action_rate_l2` (-0.02): 액션 변화율
- `joint_acc_l2` (-1e-8): 관절 가속도

---

## 📝 참고

- **설정 파일**: `rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py`
- **기본 RewardsCfg**: `rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- **Reward 함수**: `rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
