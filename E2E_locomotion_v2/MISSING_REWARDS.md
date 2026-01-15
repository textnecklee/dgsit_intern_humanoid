# Rough Task에서 활성화되었지만 E2E_locomotion_v2에 없는 Reward 항목

rl_training의 rough task (`DeeproboticsLite3RoughEnvCfg`)에서 **활성화되어 있는 (weight != 0)** reward 항목 중 E2E_locomotion_v2에 없는 항목들을 정리합니다.

---

## 활성화된 항목 목록

### 1. **feet_slide** (weight = -0.05) ⚠️ **없음**

**rl_training 설정**:
```python
self.rewards.feet_slide.weight = -0.05
self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
```

**함수 구현** (rl_training/mdp/rewards.py):
```python
def feet_slide(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.
    
    This function penalizes the agent for sliding its feet on the ground. 
    The reward is computed as the norm of the linear velocity of the feet 
    multiplied by a binary contact sensor. This ensures that the agent is 
    penalized only when the feet are in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 발의 속도를 body frame으로 변환
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    # lateral velocity (xy 평면에서의 속도)만 계산
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2))
    # contact 중일 때만 penalty 적용
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward
```

**목적**: 발이 지면과 접촉 중일 때 발이 미끄러지는 것을 패널티로 줍니다.

---

### 2. **feet_height** (weight = -0.2) ⚠️ **없음**

**rl_training 설정**:
```python
self.rewards.feet_height.weight = -0.2
self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
self.rewards.feet_height.params["target_height"] = 0.05
```

**함수 구현** (rl_training/mdp/rewards.py):
```python
def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    reward = torch.sum(foot_z_target_error, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.2
    return reward
```

**목적**: 발이 특정 높이(target_height = 0.05m)를 넘지 못할 때 패널티를 줍니다. (스윙 단계에서 발을 충분히 올리도록 유도)

---

### 3. **feet_height_body** (weight = -2.5) ⚠️ **없음**

**rl_training 설정**:
```python
self.rewards.feet_height_body.weight = -2.5
self.rewards.feet_height_body.params["target_height"] = -0.35
self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
```

**함수 구현** (rl_training/mdp/rewards.py):
```python
def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # 발 위치를 body frame으로 변환
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    # body frame에서 z 높이 오차 계산
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height)
    # 발 속도에 따라 스케일링 (빠르게 움직일 때 더 큰 패널티)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward
```

**목적**: 발이 로봇의 body frame 기준으로 특정 높이(target_height = -0.35m)를 유지하도록 유도합니다. 속도가 빠를수록 더 큰 패널티를 줍니다.

---

### 4. **undesired_contacts** (weight = -0.5) ⚠️ **없음**

**rl_training 설정**:
```python
self.rewards.undesired_contacts.weight = -0.5
self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
```

**함수 구현** (rl_training/mdp/rewards.py, isaaclab의 base 함수 사용):
```python
def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    return reward
```

**목적**: 발을 제외한 다른 body(예: 다리, 몸통 등)가 의도하지 않은 접촉을 할 때 패널티를 줍니다. 보행 중 로봇이 자신의 몸이나 다리로 땅에 닿는 것을 방지합니다.

---

### 5. **joint_mirror** (weight = -0.05) ⚠️ **없음**

**rl_training 설정**:
```python
self.rewards.joint_mirror.weight = -0.05
self.rewards.joint_mirror.params["mirror_joints"] = [
    ["FL_(HipX|HipY|Knee).*", "HR_(HipX|HipY|Knee).*"],
    ["FR_(HipX|HipY|Knee).*", "HL_(HipX|HipY|Knee).*"],
]
```

**함수 구현** (rl_training/mdp/rewards.py):
```python
def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
```

**목적**: 대칭적인 관절 쌍(예: FL과 HR, FR과 HL)이 거울처럼 대칭된 자세를 유지하도록 유도합니다. 좌우 대칭 보행을 장려합니다.

---

### 6. **feet_contact_without_cmd** (weight = 0.1) ⚠️ **없음**

**rl_training 설정**:
```python
self.rewards.feet_contact_without_cmd.weight = 0.1
self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
```

**함수 구현** (rl_training/mdp/rewards.py):
```python
def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.5
    return reward
```

**목적**: 명령이 없을 때(정지 상태) 발이 지면에 접촉하고 있으면 보상을 줍니다. 정지 상태에서 로봇이 안정적으로 서 있도록 유도합니다.

---

### 7. **joint_deviation_l1** (weight = -0.5) ⚠️ **없음**

**rl_training 설정**:
```python
self.rewards.joint_deviation_l1.weight = -0.5
self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*HipX.*"]
```

**함수 구현** (rl_training/mdp/rewards.py에서 주석 처리됨, isaaclab base에서 가져올 것으로 추정):
```python
def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)  # L1 norm
```

**목적**: 관절 위치가 기본 자세(default joint position)에서 벗어나는 것을 패널티로 줍니다. 특정 관절(HipX)만 적용하여 보행 시 기본 자세 유지에 도움을 줍니다.

---

## 요약

| Reward 항목 | Weight (rough) | 목적 | 중요도 |
|-------------|----------------|------|--------|
| **feet_slide** | -0.05 | 발 미끄러짐 방지 | ⭐⭐ |
| **feet_height** | -0.2 | 발을 최소 높이로 올리기 | ⭐⭐ |
| **feet_height_body** | -2.5 | body frame 기준 발 높이 유지 | ⭐⭐⭐ |
| **undesired_contacts** | -0.5 | 의도하지 않은 접촉 방지 | ⭐⭐⭐ |
| **joint_mirror** | -0.05 | 대칭 보행 유도 | ⭐ |
| **feet_contact_without_cmd** | 0.1 | 정지 시 안정성 | ⭐⭐ |
| **joint_deviation_l1** | -0.5 | 기본 자세 유지 | ⭐⭐ |

---

## 추가 고려사항

1. **feet_height_body**는 weight=-2.5로 가장 높은 중요도를 가지므로 우선적으로 추가하는 것을 고려해볼 수 있습니다.

2. **undesired_contacts**는 rough terrain에서 특히 중요합니다. 로봇이 다리나 몸통으로 땅에 닿는 것을 방지합니다.

3. **joint_mirror**와 **feet_contact_without_cmd**는 보행 품질을 향상시키는 데 도움이 됩니다.

4. 모든 함수는 rl_training의 구현을 그대로 사용하거나, E2E의 네이밍 컨벤션에 맞춰 `rew_` prefix를 추가할 수 있습니다.
