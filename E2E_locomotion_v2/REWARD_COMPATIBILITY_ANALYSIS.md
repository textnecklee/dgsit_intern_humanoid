# E2E_locomotion_v2 vs rl_training Reward 호환성 분석

## 📋 개요

이 문서는 E2E_locomotion_v2의 reward 함수들이 rl_training의 형식으로 적용 가능한지, 또는 그 반대로 적용 가능한지를 검토합니다.

---

## 🔍 주요 Reward 함수 비교

### 1. Velocity Tracking Rewards

#### E2E_locomotion_v2
```python
def rew_tracking_lin_vel(env, command_name: str, tracking_sigma: float, asset_cfg: SceneEntityCfg):
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = asset.data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(cmd - vel), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)  # exp(-error / sigma)
```

#### rl_training
```python
def track_lin_vel_xy_exp(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg):
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)  # exp(-error / std²)
```

**✅ 호환성**: **부분 호환**
- **차이점**: 
  - E2E: `exp(-error / sigma)` (sigma 직접 사용)
  - rl_training: `exp(-error / std²)` (std 제곱 사용)
- **변환 방법**: 
  - E2E의 `tracking_sigma=0.25` → rl_training의 `std=sqrt(0.25)=0.5`
  - 또는 rl_training 함수를 E2E 형식으로 수정 필요

---

### 2. Joint Torques

#### E2E_locomotion_v2
```python
def rew_torques(env, asset_cfg: SceneEntityCfg):
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
```

#### rl_training (Isaac Lab 기본)
```python
def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
```

**✅ 호환성**: **완전 호환**
- **동일한 계산 방식**: L2 squared penalty
- **적용 방법**: E2E의 `rew_torques`를 `mdp.joint_torques_l2`로 교체 가능

---

### 3. Joint Velocity

#### E2E_locomotion_v2
```python
def rew_dof_vel(env, asset_cfg: SceneEntityCfg):
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
```

#### rl_training (Isaac Lab 기본)
```python
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
```

**✅ 호환성**: **완전 호환**
- **동일한 계산 방식**: L2 squared penalty
- **적용 방법**: E2E의 `rew_dof_vel`를 `mdp.joint_vel_l2`로 교체 가능

---

### 4. Joint Acceleration

#### E2E_locomotion_v2
```python
def rew_dof_acc(env, asset_cfg: SceneEntityCfg):
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
```

#### rl_training (Isaac Lab 기본)
```python
def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
```

**✅ 호환성**: **완전 호환**
- **동일한 계산 방식**: L2 squared penalty
- **적용 방법**: E2E의 `rew_dof_acc`를 `mdp.joint_acc_l2`로 교체 가능

---

### 5. Joint Power

#### E2E_locomotion_v2
```python
def rew_joint_power(env, asset_cfg: SceneEntityCfg):
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(power), dim=1)  # |torque * velocity|
```

#### rl_training
```python
def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
```

**✅ 호환성**: **완전 호환**
- **동일한 계산 방식**: `|torque × velocity|`의 합
- **적용 방법**: E2E의 `rew_joint_power`를 `mdp.joint_power`로 교체 가능

---

### 6. Action Rate

#### E2E_locomotion_v2
```python
def rew_action_rate(env):
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )
```

#### rl_training (Isaac Lab 기본)
```python
def action_rate_l2(env: ManagerBasedRLEnv):
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
```

**✅ 호환성**: **완전 호환**
- **동일한 계산 방식**: L2 squared penalty
- **적용 방법**: E2E의 `rew_action_rate`를 `mdp.action_rate_l2`로 교체 가능

---

### 7. Base Height

#### E2E_locomotion_v2
```python
def rew_base_height(env, target_height: float, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)
```

#### rl_training
```python
def base_height_l2(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg | None = None):
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        # Rough terrain support: adjust target height using sensor
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
```

**✅ 호환성**: **완전 호환** (기본 사용 시)
- **기본 계산 방식**: 동일 (L2 squared penalty)
- **추가 기능**: rl_training은 rough terrain 지원 (sensor_cfg 사용)
- **적용 방법**: 
  - Flat terrain: E2E의 `rew_base_height`를 `mdp.base_height_l2`로 교체 가능
  - Rough terrain: `sensor_cfg=None`으로 설정하면 동일하게 동작

---

### 8. Linear Velocity Z

#### E2E_locomotion_v2
```python
def rew_lin_vel_z(env, asset_cfg: SceneEntityCfg):
    return torch.square(asset.data.root_lin_vel_b[:, 2])
```

#### rl_training (Isaac Lab 기본)
```python
def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
```

**✅ 호환성**: **완전 호환**
- **동일한 계산 방식**: L2 squared penalty
- **적용 방법**: E2E의 `rew_lin_vel_z`를 `mdp.lin_vel_z_l2`로 교체 가능

---

### 9. Angular Velocity XY

#### E2E_locomotion_v2
```python
def rew_ang_vel_xy(env, asset_cfg: SceneEntityCfg):
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
```

#### rl_training (Isaac Lab 기본)
```python
def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
```

**✅ 호환성**: **완전 호환**
- **동일한 계산 방식**: L2 squared penalty
- **적용 방법**: E2E의 `rew_ang_vel_xy`를 `mdp.ang_vel_xy_l2`로 교체 가능

---

### 10. Orientation

#### E2E_locomotion_v2
```python
def rew_orientation(env, asset_cfg: SceneEntityCfg):
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
```

#### rl_training (Isaac Lab 기본)
```python
def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
```

**✅ 호환성**: **완전 호환**
- **동일한 계산 방식**: L2 squared penalty
- **적용 방법**: E2E의 `rew_orientation`를 `mdp.flat_orientation_l2`로 교체 가능

---

### 11. Joint Position Limits

#### E2E_locomotion_v2
```python
def rew_dof_pos_limits(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)
```

#### rl_training (Isaac Lab 기본)
```python
def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = (
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    )
    out_of_limits = out_of_limits.clip_(min=0.0)
    return torch.sum(out_of_limits, dim=1)
```

**⚠️ 호환성**: **부분 호환** (계산 방식 약간 다름)
- **E2E 방식**: lower/upper limit을 각각 체크하고 합산
- **rl_training 방식**: 절댓값 사용하여 양쪽 limit을 동시에 체크
- **결과**: 거의 동일하지만 약간의 차이 가능
- **권장**: E2E 방식을 유지하거나 rl_training 방식으로 통일

---

## 📊 전체 호환성 요약

| Reward 함수 | E2E 이름 | rl_training 이름 | 호환성 | 비고 |
|------------|---------|-----------------|--------|------|
| **Velocity Tracking** | `rew_tracking_lin_vel` | `track_lin_vel_xy_exp` | ⚠️ 부분 | sigma vs std² 차이 |
| **Velocity Tracking** | `rew_tracking_ang_vel` | `track_ang_vel_z_exp` | ⚠️ 부분 | sigma vs std² 차이 |
| **Joint Torques** | `rew_torques` | `joint_torques_l2` | ✅ 완전 | 교체 가능 |
| **Joint Velocity** | `rew_dof_vel` | `joint_vel_l2` | ✅ 완전 | 교체 가능 |
| **Joint Acceleration** | `rew_dof_acc` | `joint_acc_l2` | ✅ 완전 | 교체 가능 |
| **Joint Power** | `rew_joint_power` | `joint_power` | ✅ 완전 | 교체 가능 |
| **Action Rate** | `rew_action_rate` | `action_rate_l2` | ✅ 완전 | 교체 가능 |
| **Base Height** | `rew_base_height` | `base_height_l2` | ✅ 완전 | 교체 가능 |
| **Linear Vel Z** | `rew_lin_vel_z` | `lin_vel_z_l2` | ✅ 완전 | 교체 가능 |
| **Angular Vel XY** | `rew_ang_vel_xy` | `ang_vel_xy_l2` | ✅ 완전 | 교체 가능 |
| **Orientation** | `rew_orientation` | `flat_orientation_l2` | ✅ 완전 | 교체 가능 |
| **Joint Pos Limits** | `rew_dof_pos_limits` | `joint_pos_limits` | ⚠️ 부분 | 계산 방식 약간 다름 |

---

## 🎯 적용 가능성 결론

### ✅ **완전 호환 (즉시 교체 가능)**
다음 함수들은 **Isaac Lab의 기본 함수로 직접 교체 가능**합니다:

1. `rew_torques` → `mdp.joint_torques_l2`
2. `rew_dof_vel` → `mdp.joint_vel_l2`
3. `rew_dof_acc` → `mdp.joint_acc_l2`
4. `rew_joint_power` → `mdp.joint_power`
5. `rew_action_rate` → `mdp.action_rate_l2`
6. `rew_base_height` → `mdp.base_height_l2` (flat terrain)
7. `rew_lin_vel_z` → `mdp.lin_vel_z_l2`
8. `rew_ang_vel_xy` → `mdp.ang_vel_xy_l2`
9. `rew_orientation` → `mdp.flat_orientation_l2`

**장점**:
- 코드 중복 제거
- Isaac Lab 표준 함수 사용
- 유지보수 용이

### ⚠️ **부분 호환 (수정 필요)**

#### 1. Velocity Tracking (`rew_tracking_lin_vel`, `rew_tracking_ang_vel`)
**문제**: 
- E2E: `exp(-error / sigma)` 
- rl_training: `exp(-error / std²)`

**해결 방법**:
- **Option A**: E2E 함수 유지 (현재 방식)
- **Option B**: rl_training 함수 사용 시 `std=sqrt(tracking_sigma)`로 변환
  - 예: `tracking_sigma=0.25` → `std=sqrt(0.25)=0.5`
- **Option C**: rl_training 함수를 E2E 형식으로 수정

#### 2. Joint Position Limits (`rew_dof_pos_limits`)
**문제**: 계산 방식이 약간 다름

**해결 방법**:
- **Option A**: E2E 방식 유지 (더 명확한 lower/upper 체크)
- **Option B**: rl_training 방식으로 통일 (더 간단)

---

## 💡 권장 사항

### 1. 즉시 교체 가능한 함수들
다음 함수들은 **Isaac Lab 기본 함수로 교체**하는 것을 권장합니다:

```python
# E2E_locomotion_v2/env_cfg.py 수정 예시
from isaaclab.envs.mdp import rewards as mdp_rewards

class RewardsCfg:
    # 기존 커스텀 함수 대신 Isaac Lab 기본 함수 사용
    torques = RewTerm(
        func=mdp_rewards.joint_torques_l2,  # mdp.rew_torques 대신
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[...])},
    )
    
    dof_vel = RewTerm(
        func=mdp_rewards.joint_vel_l2,  # mdp.rew_dof_vel 대신
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[...])},
    )
    
    # ... 등등
```

### 2. 유지해야 할 함수들
- `rew_tracking_lin_vel`, `rew_tracking_ang_vel`: 현재 형식 유지 (sigma 직접 사용)
- `rew_dof_pos_limits`: 현재 방식 유지 (더 명확한 limit 체크)

### 3. 추가 고려사항
- **Feet-related rewards**: rl_training에는 더 많은 발 관련 rewards가 있음
  - `feet_air_time`, `feet_gait`, `feet_height` 등
  - E2E에서 필요시 추가 고려

---

## 📝 마이그레이션 체크리스트

E2E_locomotion_v2를 rl_training 형식으로 마이그레이션할 때:

- [ ] `rew_torques` → `mdp_rewards.joint_torques_l2` 교체
- [ ] `rew_dof_vel` → `mdp_rewards.joint_vel_l2` 교체
- [ ] `rew_dof_acc` → `mdp_rewards.joint_acc_l2` 교체
- [ ] `rew_joint_power` → `mdp_rewards.joint_power` 교체
- [ ] `rew_action_rate` → `mdp_rewards.action_rate_l2` 교체
- [ ] `rew_base_height` → `mdp_rewards.base_height_l2` 교체 (sensor_cfg=None)
- [ ] `rew_lin_vel_z` → `mdp_rewards.lin_vel_z_l2` 교체
- [ ] `rew_ang_vel_xy` → `mdp_rewards.ang_vel_xy_l2` 교체
- [ ] `rew_orientation` → `mdp_rewards.flat_orientation_l2` 교체
- [ ] `rew_tracking_lin_vel` → `mdp.track_lin_vel_xy_exp` (std 변환 필요)
- [ ] `rew_tracking_ang_vel` → `mdp.track_ang_vel_z_exp` (std 변환 필요)
- [ ] `rew_dof_pos_limits` → 검토 후 결정

---

## 🔄 반대 방향 (rl_training → E2E)

rl_training의 함수들을 E2E 형식으로 사용하는 것도 **대부분 가능**합니다:

- ✅ `joint_torques_l2` → E2E의 `rew_torques`와 동일
- ✅ `joint_vel_l2` → E2E의 `rew_dof_vel`와 동일
- ✅ `joint_acc_l2` → E2E의 `rew_dof_acc`와 동일
- ✅ `joint_power` → E2E의 `rew_joint_power`와 동일
- ✅ `action_rate_l2` → E2E의 `rew_action_rate`와 동일
- ⚠️ `track_lin_vel_xy_exp` → E2E 형식으로 수정 필요 (std² → sigma)

---

## ✅ 최종 결론

**대부분의 reward 함수들은 같은 형식으로 적용 가능**합니다!

- **9개 함수**: 완전 호환 (즉시 교체 가능)
- **3개 함수**: 부분 호환 (약간의 수정 필요)

**권장**: Isaac Lab의 기본 함수들을 사용하여 코드 중복을 제거하고 표준화하는 것을 권장합니다.
