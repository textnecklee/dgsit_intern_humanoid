# Reward 최종 비교: rl_training (Rough Task) vs E2E_locomotion_v2

## 개요
이 문서는 현재 상태의 `rl_training` rough task와 `E2E_locomotion_v2`의 reward 설정을 비교합니다.
모든 weight 값은 rough task (`DeeproboticsLite3RoughEnvCfg`)에서 설정된 값을 기준으로 합니다.

---

## 1. General / Termination Rewards

| Reward 항목 | rl_training (Rough) | E2E_locomotion_v2 | 상태 |
|-------------|---------------------|-------------------|------|
| **is_terminated** | ❌ 없음 (또는 weight=0.0) | ❌ 없음 | - |
| **termination** | ❌ 없음 | ✅ `rew_termination` (weight=-3.0) | E2E만 있음 |
| **alive** | ❌ 없음 | ✅ `rew_alive` (weight=0.01) | E2E만 있음 |

---

## 2. Velocity Tracking Rewards

| Reward 항목 | rl_training (Rough) | E2E_locomotion_v2 | 상태 |
|-------------|---------------------|-------------------|------|
| **track_lin_vel_xy_exp** | ✅ weight=**3.5** | ✅ `rew_tracking_lin_vel` weight=**3.5** | ✅ **일치** |
| **track_ang_vel_z_exp** | ✅ weight=**1.5** | ✅ `rew_tracking_ang_vel` weight=**1.5** | ✅ **일치** |

**참고**: 계산식은 다르지만 (std vs tracking_sigma), rough task 기준으로 weight는 동일합니다.

---

## 3. Root/Body Penalties

| Reward 항목 | rl_training (Rough) | E2E_locomotion_v2 | 상태 |
|-------------|---------------------|-------------------|------|
| **lin_vel_z_l2** | ✅ weight=**-2.0** | ✅ `rew_lin_vel_z` weight=**-2.0** | ✅ **일치** |
| **ang_vel_xy_l2** | ✅ weight=**-0.05** | ✅ `rew_ang_vel_xy` weight=**-0.05** | ✅ **일치** |
| **flat_orientation_l2** | ✅ weight=**-5.0** | ✅ `rew_orientation` weight=**-5.0** | ✅ **일치** |
| **base_height_l2** | ✅ weight=**-10.0**, target=0.35 | ✅ `rew_base_height` weight=**-10.0**, target=**0.3536** | ⚠️ target 차이 (0.35 vs 0.3536) |
| **body_lin_acc_l2** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |

**참조 데이터**: 모두 `robot` root body 사용 → **겹침**

---

## 4. Joint Penalties

| Reward 항목 | rl_training (Rough) | E2E_locomotion_v2 | 상태 |
|-------------|---------------------|-------------------|------|
| **joint_torques_l2** | ✅ weight=**-2.5e-5** | ✅ `rew_torques` weight=**-2.5e-5** | ✅ **일치** |
| **joint_vel_l2** | ✅ weight=0.0 (비활성화) | ✅ `rew_dof_vel` weight=**-5.0e-4** | ⚠️ **차이** (rl_training 비활성화, E2E 활성화) |
| **joint_acc_l2** | ✅ weight=**-1e-8** | ✅ `rew_dof_acc` weight=**-1e-8** | ✅ **일치** |
| **joint_deviation_l1** | ✅ weight=**-0.5** (HipX만) | ✅ `rew_joint_deviation_l1` weight=**-0.5** (HipX만) | ✅ **일치** |
| **joint_power** | ✅ weight=**-2e-5** | ✅ `rew_joint_power` weight=**-2e-5** | ✅ **일치** |
| **joint_pos_limits** | ✅ weight=**-5.0** | ✅ `rew_dof_pos_limits` weight=**-5.0** | ✅ **일치** |
| **joint_vel_limits** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |

**참조 Joint**: 동일한 joint들 사용 → **겹침**

**주의**: 
- `joint_vel_l2`: rl_training은 비활성화, E2E는 활성화(-5.0e-4)

---

## 5. Joint Position Penalties (Command-dependent)

| Reward 항목 | rl_training (Rough) | E2E_locomotion_v2 | 상태 |
|-------------|---------------------|-------------------|------|
| **stand_still_without_cmd** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |
| **joint_pos_penalty** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |
| **stand_still** | ✅ weight=**-0.5** | ✅ `rew_stand_still` weight=**-0.5** | ✅ **일치** |

**주의**: rl_training의 `stand_still`은 `stand_still_joint_deviation_l1` 함수를 사용하지만, E2E의 `rew_stand_still`은 stub 함수(항상 0 반환)입니다. 실제 구현이 다를 수 있습니다.

---

## 6. Mirror/Sync Rewards

| Reward 항목 | rl_training (Rough) | E2E_locomotion_v2 | 상태 |
|-------------|---------------------|-------------------|------|
| **joint_mirror** | ✅ weight=**-0.05** | ✅ `rew_joint_mirror` weight=**-0.05** | ✅ **일치** |
| **action_mirror** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |
| **action_sync** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |

---

## 7. Action Penalties

| Reward 항목 | rl_training (Rough) | E2E_locomotion_v2 | 상태 |
|-------------|---------------------|-------------------|------|
| **action_rate_l2** | ✅ weight=**-0.02** | ✅ `rew_action_rate` weight=**-0.02** | ✅ **일치** |
| **applied_torque_limits** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |

**참조**: 둘 다 action space 사용 → **겹침**

---

## 8. Contact/Feet Rewards

| Reward 항목 | rl_training (Rough) | E2E_locomotion_v2 | 상태 |
|-------------|---------------------|-------------------|------|
| **feet_air_time** | ✅ weight=**5.0**, threshold=0.5 | ✅ `rew_feet_air_time` weight=**5.0**, threshold=**0.5** | ✅ **일치** |
| **feet_air_time_variance** | ✅ weight=**-8.0** | ✅ `rew_feet_air_time_variance` weight=**-8.0** | ✅ **일치** |
| **feet_gait** | ✅ weight=**0.5** | ✅ `GaitReward` weight=**0.5** | ✅ **일치** |
| **feet_slide** | ✅ weight=**-0.05** | ✅ `rew_feet_slide` weight=**-0.05** | ✅ **일치** |
| **feet_height** | ✅ weight=**-0.2**, target=0.05 | ✅ `rew_feet_height` weight=**-0.2**, target=**0.05** | ✅ **일치** |
| **feet_height_body** | ✅ weight=**-2.5**, target=-0.35 | ✅ `rew_feet_height_body` weight=**-2.5**, target=**-0.35** | ✅ **일치** |
| **feet_contact** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |
| **feet_contact_without_cmd** | ✅ weight=**0.1** | ✅ `rew_feet_contact_without_cmd` weight=**0.1** | ✅ **일치** |
| **feet_stumble** | ✅ weight=0.0 (비활성화) | ✅ `rew_feet_stumble` weight=**-0.0** (비활성화) | ✅ **일치** (둘 다 비활성화) |
| **contact_forces** | ✅ weight=**-0.1**, feet만 | ✅ `rew_collision` weight=**-0.1**, link_collision_sensor | ⚠️ **부분 겹침** (센서/대상 다름) |
| **undesired_contacts** | ✅ weight=**-0.5** | ❌ 없음 | rl_training만 있음 (의도적으로 제외) |

**주의사항**:
- `contact_forces` vs `collision`: 
  - rl_training: 발만 감지 (contact_forces sensor, body_names로 필터링)
  - E2E: 다리/몸통 감지 (link_collision_sensor, sensor에서 필터링)
  - 목적과 대상이 다르지만 weight는 동일 (-0.1)
- `undesired_contacts`: rl_training에는 있지만 E2E에는 의도적으로 추가하지 않음

---

## 9. 기타 Rewards

| Reward 항목 | rl_training (Rough) | E2E_locomotion_v2 | 상태 |
|-------------|---------------------|-------------------|------|
| **straight_line_motion** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |
| **feet_distance_y_exp** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |
| **upward** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (비활성화) |
| **wheel_vel_penalty** | ✅ weight=0.0 (비활성화) | ❌ 없음 | rl_training만 있음 (wheeled robot용) |

---

## 10. Weight 비교 요약

### 10.1 활성화된 Reward 항목 비교

| 항목 | rl_training (Rough) | E2E_locomotion_v2 | 상태 |
|------|---------------------|-------------------|------|
| **track_lin_vel_xy_exp** | 3.5 | 3.5 | ✅ 일치 |
| **track_ang_vel_z_exp** | 1.5 | 1.5 | ✅ 일치 |
| **lin_vel_z_l2** | -2.0 | -2.0 | ✅ 일치 |
| **ang_vel_xy_l2** | -0.05 | -0.05 | ✅ 일치 |
| **flat_orientation_l2** | -5.0 | -5.0 | ✅ 일치 |
| **base_height_l2** | -10.0 | -10.0 | ✅ 일치 |
| **target_height** | 0.35 | 0.3536 | ⚠️ 차이 (사용자 수정) |
| **joint_torques_l2** | -2.5e-5 | -2.5e-5 | ✅ 일치 |
| **joint_acc_l2** | -1e-8 | -1e-8 | ✅ 일치 |
| **joint_deviation_l1** | -0.5 | -0.5 | ✅ 일치 |
| **joint_power** | -2e-5 | -2e-5 | ✅ 일치 |
| **joint_pos_limits** | -5.0 | -5.0 | ✅ 일치 |
| **action_rate_l2** | -0.02 | -0.02 | ✅ 일치 |
| **feet_air_time** | 5.0 | 5.0 | ✅ 일치 |
| **feet_air_time_variance** | -8.0 | -8.0 | ✅ 일치 |
| **feet_gait** | 0.5 | 0.5 | ✅ 일치 |
| **feet_slide** | -0.05 | -0.05 | ✅ 일치 |
| **stand_still** | -0.5 | -0.5 | ✅ 일치 |
| **feet_height_body** | -2.5 | -2.5 | ✅ 일치 |
| **feet_height** | -0.2 | -0.2 | ✅ 일치 |
| **feet_contact_without_cmd** | 0.1 | 0.1 | ✅ 일치 |
| **contact_forces/collision** | -0.1 | -0.1 | ✅ 일치 (대상 다름) |
| **undesired_contacts** | -0.5 | 없음 | ❌ E2E에 없음 (의도적 제외) |

### 10.2 차이점

1. **base_height target_height**: 0.35 vs 0.3536 (사용자가 수동으로 변경)
2. **joint_vel_l2**: rl_training은 비활성화(0.0), E2E는 활성화(-5.0e-4)
3. **undesired_contacts**: rl_training에는 있지만 E2E에는 의도적으로 추가하지 않음
4. **contact_forces vs collision**: 같은 weight이지만 대상과 센서가 다름

---

## 11. rl_training에만 있는 Reward 항목 (비활성화 포함)

다음 항목들은 rl_training (rough)에만 존재하며 E2E_locomotion_v2에는 없습니다:

1. **body_lin_acc_l2** (weight=0.0, 비활성화)
2. **joint_vel_limits** (weight=0.0, 비활성화)
3. **stand_still_without_cmd** (weight=0.0, 비활성화)
4. **joint_pos_penalty** 및 변형들 (weight=0.0, 비활성화)
5. **action_mirror** (weight=0.0, 비활성화)
6. **action_sync** (weight=0.0, 비활성화)
7. **applied_torque_limits** (weight=0.0, 비활성화)
8. **feet_contact** (weight=0.0, 비활성화)
9. **straight_line_motion** (weight=0.0, 비활성화)
10. **feet_distance_y_exp** (weight=0.0, 비활성화)
11. **upward** (weight=0.0, 비활성화)
12. **undesired_contacts** (weight=-0.5) ⭐ **활성화됨, E2E에는 의도적으로 없음**

---

## 12. E2E_locomotion_v2에만 있는 Reward 항목

다음 항목들은 E2E_locomotion_v2에만 존재하며 rl_training (rough)에는 없습니다:

1. **termination** (weight=-3.0) ⭐ **활성화됨**
2. **alive** (weight=0.01) ⭐ **활성화됨**
3. **dof_vel** (weight=-5.0e-4) ⭐ **활성화됨** (rl_training은 비활성화)

---

## 13. 함수명 매핑

| 기능 | rl_training 함수명 | E2E_locomotion_v2 함수명 | 상태 |
|------|-------------------|-------------------------|------|
| Linear vel xy tracking | `track_lin_vel_xy_exp` | `rew_tracking_lin_vel` | ✅ |
| Angular vel z tracking | `track_ang_vel_z_exp` | `rew_tracking_ang_vel` | ✅ |
| Linear vel z penalty | `lin_vel_z_l2` | `rew_lin_vel_z` | ✅ |
| Angular vel xy penalty | `ang_vel_xy_l2` | `rew_ang_vel_xy` | ✅ |
| Orientation penalty | `flat_orientation_l2` | `rew_orientation` | ✅ |
| Base height penalty | `base_height_l2` | `rew_base_height` | ✅ |
| Joint torques penalty | `joint_torques_l2` | `rew_torques` | ✅ |
| Joint velocity penalty | `joint_vel_l2` | `rew_dof_vel` | ✅ |
| Joint acceleration penalty | `joint_acc_l2` | `rew_dof_acc` | ✅ |
| Joint power penalty | `joint_power` | `rew_joint_power` | ✅ |
| Joint position limits | `joint_pos_limits` | `rew_dof_pos_limits` | ✅ |
| Action rate penalty | `action_rate_l2` | `rew_action_rate` | ✅ |
| Feet air time | `feet_air_time_including_ang_z` | `rew_feet_air_time` | ✅ |
| Feet air time variance | `feet_air_time_variance_penalty` | `rew_feet_air_time_variance` | ✅ |
| Feet gait | `GaitReward` | `GaitReward` | ✅ (동일) |
| Feet slide | `feet_slide` | `rew_feet_slide` | ✅ |
| Feet height | `feet_height` | `rew_feet_height` | ✅ |
| Feet height body | `feet_height_body` | `rew_feet_height_body` | ✅ |
| Feet contact without cmd | `feet_contact_without_cmd` | `rew_feet_contact_without_cmd` | ✅ |
| Contact forces | `contact_forces` | `rew_collision` | ⚠️ (대상 다름) |
| Stand still | `stand_still_joint_deviation_l1` | `rew_stand_still` | ⚠️ (구현 다를 수 있음) |
| Joint mirror | `joint_mirror` | `rew_joint_mirror` | ✅ |
| Joint deviation L1 | `joint_deviation_l1` | `rew_joint_deviation_l1` | ✅ |

---

## 14. 주요 차이점 및 주의사항

### 14.1 완전히 일치하는 항목 (22개)

대부분의 활성화된 reward 항목이 weight 값에서 일치합니다:
- Velocity tracking (2개)
- Root/Body penalties (3개)
- Joint penalties (5개)
- Action penalties (1개)
- Feet rewards (10개)
- Mirror reward (1개)

### 14.2 차이점 (4개)

1. **base_height target_height**: 0.35 vs 0.3536 (사용자가 수동 수정)
2. **joint_vel_l2**: rl_training은 비활성화, E2E는 활성화(-5.0e-4)
3. **undesired_contacts**: rl_training에만 있음 (weight=-0.5)
4. **contact_forces vs collision**: 동일한 weight(-0.1)이지만 대상이 다름

### 14.3 E2E에만 있는 항목

- **termination** (weight=-3.0): E2E만 활성화
- **alive** (weight=0.01): E2E만 활성화

### 14.4 구현 차이 가능성

- **stand_still**: rl_training은 `stand_still_joint_deviation_l1` 함수 사용, E2E는 stub 함수(항상 0) 사용 가능
- **contact_forces vs collision**: 같은 목적이지만 센서와 대상이 다름

---

## 15. 결론

### 15.1 Overall 상태

**활성화된 reward 항목**: 거의 동일하게 설정되어 있습니다.

**Weight 값 일치율**: 약 **95%** (23개 중 22개 일치)

### 15.2 남은 차이점

1. **base_height target_height**: 0.35 vs 0.3536 (사용자 수동 수정, 의도적으로 다를 수 있음)
2. **joint_vel_l2**: E2E만 활성화 (-5.0e-4)
3. **undesired_contacts**: rl_training에만 있음 (의도적으로 E2E에 추가하지 않음)
4. **termination/alive**: E2E에만 있음 (E2E 특화)

### 15.3 권장사항

1. **stand_still** 함수 구현 확인: E2E의 `rew_stand_still`이 실제로 동작하는지 확인 필요
2. **undesired_contacts** 추가 여부: 필요하다면 E2E에도 추가 가능 (의도적으로 제외했으므로 재검토)

---

## 참고 자료

- rl_training rough task config: `rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py`
- rl_training base config: `rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- E2E_locomotion_v2 config: `E2E_locomotion_v2/env_cfg.py`
