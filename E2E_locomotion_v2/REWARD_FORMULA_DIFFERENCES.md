# Reward 계산식 차이점: rl_training vs E2E_locomotion_v2

이 문서는 `rl_training`과 `E2E_locomotion_v2`에서 **동일한 목적의 reward**지만 **계산식이나 적용 방식이 다른** 경우를 정리합니다.

---

## 1. Velocity Tracking Rewards

### 1.1 track_lin_vel_xy_exp vs rew_tracking_lin_vel

**rl_training**:
```python
def track_lin_vel_xy_exp(env, std: float, command_name: str, ...):
    lin_vel_error = torch.sum(torch.square(cmd[:, :2] - vel[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std**2)  # std**2로 나눔
    return reward

# 사용: std=math.sqrt(0.5) → std**2 = 0.5로 나눔
```

**E2E_locomotion_v2**:
```python
def rew_tracking_lin_vel(env, tracking_sigma: float, command_name: str, ...):
    lin_vel_error = torch.sum(torch.square(cmd[:, :2] - vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)  # tracking_sigma로 나눔

# 사용: tracking_sigma=0.25 → 0.25로 나눔
```

**차이점**:
- **rl_training**: `std`를 받아서 `std**2`로 나눔 (예: std=√0.5 → 0.5로 나눔)
- **E2E**: `tracking_sigma`를 직접 받아서 그대로 나눔 (예: 0.25로 나눔)
- **영향**: E2E에서 `tracking_sigma=0.25`를 사용하면, rl_training에서 `std=0.5`를 사용한 것과 동일한 효과입니다.
- **주의**: E2E의 `tracking_sigma` 값이 `std**2` 형태로 전달되고 있는지 확인 필요. 현재 E2E에서 `tracking_sigma=0.25`를 사용하는데, rl_training에서 `std=math.sqrt(0.5)≈0.707`를 사용하면 `std**2=0.5`가 되어 **2배 차이**가 발생합니다.

**결론**: ⚠️ **계산식 차이** - E2E는 std 값을 직접 받지만 rl_training은 std**2로 나누므로, 동일한 효과를 얻으려면 E2E의 `tracking_sigma`는 rl_training의 `std**2` 값이어야 합니다.

---

### 1.2 track_ang_vel_z_exp vs rew_tracking_ang_vel

**rl_training**:
```python
def track_ang_vel_z_exp(env, std: float, command_name: str, ...):
    ang_vel_error = torch.square(cmd[:, 2] - vel[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)  # std**2로 나눔
    return reward
```

**E2E_locomotion_v2**:
```python
def rew_tracking_ang_vel(env, tracking_sigma: float, command_name: str, ...):
    ang_vel_error = torch.square(cmd_yaw - yaw_vel)
    return torch.exp(-ang_vel_error / tracking_sigma)  # tracking_sigma로 나눔
```

**차이점**: 위와 동일한 패턴 (std vs tracking_sigma)

---

## 2. Base Height Reward

### 2.1 base_height_l2 vs rew_base_height

**rl_training**:
```python
def base_height_l2(env, target_height: float, asset_cfg, sensor_cfg=None):
    if sensor_cfg is not None:
        # Rough terrain: sensor로 terrain 높이 조정
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]  # z 좌표만 추출 (높이)
        adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Flat terrain: 직접 사용
        adjusted_target_height = target_height
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return reward
```

**E2E_locomotion_v2**:
```python
def rew_base_height(env, target_height: float, asset_cfg):
    # 항상 고정된 target_height 사용 (terrain 조정 없음)
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)
```

**차이점**:
- **rl_training**: `sensor_cfg`가 제공되면 **rough terrain에 대해 target height를 자동으로 조정** (RayCaster 사용)
- **E2E**: 항상 고정된 `target_height` 사용, terrain 조정 없음

**영향**: Rough terrain에서 rl_training은 terrain 높이를 고려하지만, E2E는 그렇지 않습니다.

**결론**: ⚠️ **기능 차이** - Rough terrain 환경에서 동작이 다릅니다.

#### Sensor로 Terrain 높이 조정하는 방법 (상세 설명)

**RayCaster Sensor란?**
- RayCaster는 로봇의 base에서 **아래 방향으로 레이(광선)를 쏘아서** terrain mesh와의 충돌 지점을 측정하는 센서입니다.
- 여러 개의 레이를 grid pattern으로 쏴서, 로봇 주변의 terrain 높이를 측정합니다.

**예시 설정** (rl_training의 velocity_env_cfg.py):
```python
height_scanner_base = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",  # 로봇 base에 부착
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # base 위 20m에서 시작
    ray_alignment="yaw",  # 로봇의 yaw 방향에 맞춤
    pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),  # 0.1m x 0.1m 그리드
    mesh_prim_paths=["/World/ground"],  # ground mesh와 충돌 감지
)
```

**작동 원리**:

1. **레이 캐스팅**: 로봇 base에서 아래로 레이를 쏘아 ground mesh와의 충돌점을 찾습니다.
   - `ray_hits_w[..., 2]`는 각 레이가 ground와 충돌한 **세계 좌표계의 z 좌표(높이)**입니다.
   - 여러 개의 레이가 있으면 여러 점의 높이를 측정합니다.

2. **평균 높이 계산**: 
   - `torch.mean(ray_hits, dim=1)`: 여러 레이의 충돌점 높이의 평균을 계산
   - 예: 레이 4개가 각각 z=0.2, 0.3, 0.25, 0.35에 충돌 → 평균 = 0.275m

3. **Target Height 조정**:
   - Flat terrain: `target_height = 0.35m` (고정)
   - Rough terrain: `adjusted_target_height = target_height + 평균_terrain_높이`
     - 예: base height가 0.35m에서 시작하고, 현재 terrain이 평균 0.1m 높이에 있으면
     - `adjusted_target_height = 0.35 + 0.1 = 0.45m`

**왜 필요한가?**

- **Flat terrain**: 로봇이 항상 같은 높이(z=0)의 평지에 있으므로, target height는 고정값(예: 0.35m)으로 충분합니다.
- **Rough terrain**: 로봇이 언덕, 계곡 등 다양한 높이의 terrain 위에 있을 수 있습니다.
  - 예시 1: 언덕 위 (terrain 높이 +0.2m)
    - 고정 target: `base_height = 0.35m`로 유지하려고 하면 로봇이 언덕에서 떨어지려 할 수 있음
    - 조정된 target: `base_height = 0.35 + 0.2 = 0.55m`로 조정하면, 언덕 위에서도 올바른 높이 유지
  
  - 예시 2: 계곡 안 (terrain 높이 -0.1m)
    - 고정 target: `base_height = 0.35m`로 유지하려고 하면 로봇이 땅에 박히려 할 수 있음
    - 조정된 target: `base_height = 0.35 - 0.1 = 0.25m`로 조정하면, 계곡 안에서도 올바른 높이 유지

**시각적 예시**:

```
Flat Terrain:
        Robot (target: 0.35m)
           |
           |
    ─────────────── ground (z=0)
    
Rough Terrain (언덕):
        Robot (adjusted target: 0.55m)
           |
           |
      /|\ 언덕 (z=+0.2m)
     / | \
─────────────── reference ground (z=0)
```

**E2E_locomotion_v2의 경우**:
- 현재는 sensor를 사용하지 않으므로, rough terrain에서도 고정된 target height만 사용합니다.
- Rough terrain 환경을 사용한다면, 로봇이 terrain 높이 변화에 적응하기 어려울 수 있습니다.

---

## 3. Feet Air Time Reward

### 3.1 feet_air_time_including_ang_z vs rew_feet_air_time

**rl_training**:
```python
def feet_air_time_including_ang_z(env, command_name: str, sensor_cfg, threshold: float):
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # Command check: 모든 차원 확인 (xy + ang_z)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward
```

**E2E_locomotion_v2**:
```python
def rew_feet_air_time(env, command_name: str, sensor_cfg, threshold: float):
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # Command check: xy 차원만 확인
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward
```

**차이점**:
- **rl_training**: Command norm 계산 시 **모든 차원 (x, y, z, yaw)** 확인 → `command[:, :]` 또는 `command` 전체 사용
- **E2E**: Command norm 계산 시 **모든 차원** 확인 → 실제로는 동일함

**재확인 필요**: 
- rl_training의 `torch.norm(..., dim=1)`은 전체 command 벡터의 norm을 계산하므로 (x, y, yaw) 모두 포함
- E2E도 동일하게 `torch.norm(..., dim=1)`을 사용하므로 실제로는 **동일한 계산**입니다.

**함수명 차이**: 
- `feet_air_time_including_ang_z`라는 이름이지만, 실제로는 command norm 체크 시 모든 차원을 포함하므로 `feet_air_time`과 동일합니다.

**결론**: ✅ **실제 계산은 동일** (함수명만 다름)

---

## 4. Joint Power Reward

### 4.1 joint_power vs rew_joint_power

**rl_training**:
```python
def joint_power(env, asset_cfg):
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, joint_ids] * asset.data.applied_torque[:, joint_ids]),
        dim=1,
    )
    return reward
```

**E2E_locomotion_v2**:
```python
def rew_joint_power(env, asset_cfg):
    power = asset.data.applied_torque[:, joint_ids] * asset.data.joint_vel[:, joint_ids]
    return torch.sum(torch.abs(power), dim=1)
```

**차이점**: **계산 순서만 다름, 결과는 동일**
- rl_training: 곱셈 → abs → sum
- E2E: 곱셈 → abs → sum

**결론**: ✅ **동일한 계산식**

---

## 5. Gait Reward

### 5.1 GaitReward 클래스

**rl_training**:
```python
class GaitReward(ManagerTermBase):
    def __call__(self, ...):
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        return reward
```

**E2E_locomotion_v2**:
```python
class GaitReward(ManagerTermBase):
    def __call__(self, ...):
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        return reward
```

**차이점**: ✅ **동일한 구현**

---

## 6. Feet Air Time Variance

### 6.1 feet_air_time_variance_penalty vs rew_feet_air_time_variance

**rl_training**:
```python
def feet_air_time_variance_penalty(env, sensor_cfg):
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)
    return reward
```

**E2E_locomotion_v2**:
```python
def rew_feet_air_time_variance(env, sensor_cfg):
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)
    return reward
```

**차이점**: ✅ **동일한 계산식**

---

## 7. Contact Forces / Collision

### 7.1 contact_forces vs rew_collision

**rl_training**:
```python
# contact_forces 함수는 rewards.py에 직접 정의되어 있지 않음
# 아마도 base Isaac Lab의 함수를 사용하는 것으로 추정
```

**E2E_locomotion_v2**:
```python
def rew_collision(env, sensor_cfg, contact_force_threshold: float = 1.0):
    sensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_forces_w
    force_norm = torch.linalg.norm(forces_w, dim=-1)
    collision_mask = (force_norm > contact_force_threshold).float()
    collision_count = torch.sum(collision_mask, dim=1)
    return collision_count
```

**차이점**: 
- rl_training의 `contact_forces` 구현을 확인할 수 없으나, 일반적으로 동일한 방식일 가능성이 높습니다.
- **주의**: Sensor 이름이 다를 수 있음 (rl_training: `contact_forces`, E2E: `link_collision_sensor`)

**결론**: ⚠️ **구현 확인 불가** (rl_training 구현 없음)

---

## 8. Base Height 데이터 소스

### 8.1 root_pos_w vs root_link_pos_w

**rl_training** (base_height_l2):
```python
reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
```

**E2E_locomotion_v2** (rew_base_height):
```python
return torch.square(asset.data.root_pos_w[:, 2] - target_height)
```

**차이점**: ✅ **동일한 데이터 소스 사용** (`root_pos_w`)

---

## 9. Orientation Reward

### 9.1 flat_orientation_l2 vs rew_orientation

**rl_training**:
```python
def flat_orientation_l2(env, asset_cfg):
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return reward
```

**E2E_locomotion_v2**:
```python
def rew_orientation(env, asset_cfg):
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
```

**차이점**: ✅ **동일한 계산식**

---

## 요약: 계산식이 다른 Reward 항목

### ⚠️ 주의가 필요한 항목 (계산식 차이)

1. **track_lin_vel_xy_exp / track_ang_vel_z_exp**
   - **차이**: rl_training은 `std**2`로 나누지만, E2E는 `tracking_sigma`를 직접 나눔
   - **영향**: E2E에서 `tracking_sigma=0.25`를 사용하면, rl_training에서 `std=0.5`를 사용한 것과 동일 (std=√0.5≈0.707일 때는 std**2=0.5)
   - **권장**: E2E의 `tracking_sigma` 값을 rl_training의 `std**2` 값으로 설정해야 동일한 효과

2. **base_height_l2**
   - **차이**: rl_training은 rough terrain에서 sensor로 target height를 조정하지만, E2E는 고정값 사용
   - **영향**: Rough terrain 환경에서 동작이 다름
   - **권장**: Rough terrain을 사용한다면 E2E에도 sensor_cfg 기반 조정 로직 추가 고려

### ✅ 동일한 계산식 (함수명만 다름)

- `joint_power` / `rew_joint_power`
- `GaitReward` (클래스)
- `feet_air_time_variance_penalty` / `rew_feet_air_time_variance`
- `flat_orientation_l2` / `rew_orientation`
- `lin_vel_z_l2` / `rew_lin_vel_z`
- `ang_vel_xy_l2` / `rew_ang_vel_xy`

---

## 수정 권장사항

### 1. tracking_sigma 파라미터 수정

**E2E_locomotion_v2/env_cfg.py**에서:
```python
tracking_lin_vel = RewTerm(
    func=mdp.rew_tracking_lin_vel,
    weight=3.0,
    params={
        "command_name": "base_velocity",
        "tracking_sigma": 0.5,  # 0.25 → 0.5 (rl_training의 std**2 값)
        "asset_cfg": SceneEntityCfg("robot"),
    },
)
```

또는 **rew_tracking_lin_vel 함수 수정**:
```python
def rew_tracking_lin_vel(env, tracking_sigma: float, ...):
    # tracking_sigma를 std로 해석하고 std**2로 나누도록 수정
    lin_vel_error = torch.sum(torch.square(cmd - vel), dim=1)
    return torch.exp(-lin_vel_error / (tracking_sigma ** 2))  # std**2로 나누기
```

### 2. base_height에 terrain 조정 추가 (선택사항)

Rough terrain을 사용한다면 `rew_base_height` 함수를 수정하여 sensor 기반 조정 기능을 추가할 수 있습니다.

---

## 참고

- rl_training rewards: `rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- E2E rewards: `E2E_locomotion_v2/mdp/rewards.py`
