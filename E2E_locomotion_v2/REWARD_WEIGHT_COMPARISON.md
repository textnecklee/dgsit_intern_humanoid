# 리워드 가중치 비교: rl_training vs E2E_locomotion_v2

## 개요
이 문서는 `rl_training`의 rough 환경(`DeeproboticsLite3RoughEnvCfg`)과 `E2E_locomotion_v2`의 리워드 가중치를 비교합니다.

---

## 리워드 가중치 비교표

| 리워드 항목 | rl_training (rough) | E2E_locomotion_v2 | 차이 | 비고 |
|------------|---------------------|-------------------|------|------|
| **Tracking Rewards** |
| `track_lin_vel_xy_exp` / `tracking_lin_vel` | 3.5 | 3.5 | 0.0 | ✅ 동일 |
| `track_ang_vel_z_exp` / `tracking_ang_vel` | 1.5 | 1.5 | 0.0 | ✅ 동일 |
| **Root Penalties** |
| `lin_vel_z_l2` / `lin_vel_z` | -2.0 | -4.0 | -2.0 | ⚠️ E2E가 더 강한 패널티 |
| `ang_vel_xy_l2` / `ang_vel_xy` | -0.05 | -0.2 | -0.15 | ⚠️ E2E가 더 강한 패널티 |
| `flat_orientation_l2` / `orientation` | -5.0 | -5.0 | 0.0 | ✅ 동일 |
| `base_height_l2` / `base_height` | -10.0 | -10.0 | 0.0 | ✅ 동일 |
| `base_height_l2.target_height` | 0.35 | 0.3536 | 0.0036 | ⚠️ 미세 차이 |
| **Joint Penalties** |
| `joint_torques_l2` / `torques` | -2.5e-5 | -2.5e-5 | 0.0 | ✅ 동일 |
| `joint_acc_l2` / `dof_acc` | -1e-8 | -1e-8 | 0.0 | ✅ 동일 |
| `joint_power` | -2e-5 | -2e-5 | 0.0 | ✅ 동일 |
| `joint_deviation_l1` | -0.5 | -0.5 | 0.0 | ✅ 동일 |
| `joint_pos_limits` / `dof_pos_limits` | -5.0 | -5.0 | 0.0 | ✅ 동일 |
| `joint_mirror` | -0.05 | -0.05 | 0.0 | ✅ 동일 |
| **Action Smoothness** |
| `action_rate_l2` / `action_rate` | -0.02 | -0.02 | 0.0 | ✅ 동일 |
| **Feet Rewards** |
| `feet_air_time` | 5.0 | 5.0 | 0.0 | ✅ 동일 |
| `feet_air_time.threshold` | 0.5 | 0.5 | 0.0 | ✅ 동일 |
| `feet_air_time_variance` | -8.0 | -8.0 | 0.0 | ✅ 동일 |
| `feet_slide` | -0.05 | -0.05 | 0.0 | ✅ 동일 |
| `feet_height` | -0.2 | -0.2 | 0.0 | ✅ 동일 |
| `feet_height.target_height` | 0.05 | 0.05 | 0.0 | ✅ 동일 |
| `feet_height_body` | -2.5 | -2.5 | 0.0 | ✅ 동일 |
| `feet_height_body.target_height` | -0.35 | -0.3536 | -0.0036 | ⚠️ 미세 차이 |
| `feet_contact_without_cmd` | 0.1 | 0.1 | 0.0 | ✅ 동일 |
| **Contact Rewards** |
| `contact_forces` / `foot_contact_forces` | -1e-1 (-0.1) | -0.1 | 0.0 | ✅ 동일 |
| `contact_forces.threshold` | (기본값) | 200.0 | - | ⚠️ E2E에만 threshold 명시 |
| `undesired_contacts` | -0.5 | -0.5 | 0.0 | ✅ 동일 |
| `undesired_contacts.threshold` | 1.0 | 1.0 | 0.0 | ✅ 동일 |
| **Other Rewards** |
| `stand_still` | -0.5 | -0.5 | 0.0 | ✅ 동일 |
| `stand_still.command_threshold` | 0.1 | 0.1 | 0.0 | ✅ 동일 |
| `feet_gait` | 0.5 | (주석 처리됨) | - | ⚠️ E2E에서 비활성화 |
| `feet_stumble` | (없음) | 0.0 | - | ⚠️ E2E에만 존재 (비활성화) |
| `dof_vel` | (없음) | 0.0 | - | ⚠️ E2E에만 존재 (비활성화) |
| `termination` / `is_terminated` | 0.0 | 0.0 | 0.0 | ✅ 동일 |

---

## 주요 차이점 요약

### 1. **Root Penalties 차이**
- **`lin_vel_z`**: E2E가 **-4.0**으로 rl_training의 **-2.0**보다 2배 강한 패널티
  - E2E는 수직 속도(점프)를 더 강하게 억제
- **`ang_vel_xy`**: E2E가 **-0.2**로 rl_training의 **-0.05**보다 4배 강한 패널티
  - E2E는 roll/pitch 각속도를 더 강하게 억제

### 2. **Target Height 차이**
- **`base_height.target_height`**: 
  - rl_training: **0.35**
  - E2E: **0.3536** (0.0036 차이)
- **`feet_height_body.target_height`**: 
  - rl_training: **-0.35**
  - E2E: **-0.3536** (0.0036 차이)

### 3. **비활성화된 리워드**
- **`feet_gait`**: 
  - rl_training: **0.5** (활성화)
  - E2E: 주석 처리됨 (비활성화)
- **`feet_stumble`**: E2E에만 존재하지만 weight=0.0으로 비활성화
- **`dof_vel`**: E2E에만 존재하지만 weight=0.0으로 비활성화

### 4. **Threshold 차이**
- **`foot_contact_forces.threshold`**: 
  - rl_training: 기본값 사용 (명시되지 않음)
  - E2E: **200.0** 명시

---

## 영향 분석

### E2E가 더 강한 패널티를 가진 항목들:
1. **`lin_vel_z` (-4.0 vs -2.0)**: 
   - 점프/호핑을 더 강하게 억제
   - 로봇이 더 낮게 유지되도록 유도
   
2. **`ang_vel_xy` (-0.2 vs -0.05)**:
   - 로봇의 기울임을 더 강하게 억제
   - 더 안정적인 자세 유지 유도

### 비활성화된 리워드:
- **`feet_gait`**: E2E에서 비활성화되어 보행 패턴 리워드가 없음
  - rl_training은 trot 보행 패턴을 명시적으로 리워드
  - E2E는 다른 리워드들(예: `feet_air_time`, `feet_air_time_variance`)로 보행 패턴을 간접적으로 유도

---

## 결론

대부분의 리워드 가중치는 동일하지만, E2E는 **수직 속도와 각속도 패널티를 더 강하게** 설정하여:
- 더 낮은 자세 유지
- 더 안정적인 보행
- 점프/호핑 억제

를 목표로 하는 것으로 보입니다.

반면, rl_training은 **`feet_gait` 리워드를 통해 보행 패턴을 명시적으로 학습**시키는 반면, E2E는 이를 비활성화하고 다른 메커니즘에 의존합니다.
