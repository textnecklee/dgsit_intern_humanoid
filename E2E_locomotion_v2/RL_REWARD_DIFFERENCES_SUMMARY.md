# RL Reward 전체 비교: rl_training vs E2E_locomotion_v2

## 개요
이 문서는 `rl_training` (rough task)와 `E2E_locomotion_v2`의 **Reward 전체적인 차이점**을 분석합니다.

---

## 1. 전체 통계

| 항목 | rl_training (Rough) | E2E_locomotion_v2 | 비고 |
|------|---------------------|-------------------|------|
| **활성화된 Reward 개수** | ~23개 | ~24개 | E2E가 1개 더 많음 |
| **Weight 일치율** | - | **~95%** (23개 중 22개) | 거의 동일 |
| **E2E 전용 항목** | - | 3개 | termination, alive, dof_vel |
| **rl_training 전용 항목** | 1개 (활성화) | - | undesired_contacts |

---

## 2. 카테고리별 비교

### 2.1 Velocity Tracking Rewards ✅

**목적**: 명령 속도를 추적하도록 유도

| Reward | rl_training | E2E | 상태 |
|--------|-------------|-----|------|
| **track_lin_vel_xy_exp** | 3.5 | 3.5 | ✅ 완전 일치 |
| **track_ang_vel_z_exp** | 1.5 | 1.5 | ✅ 완전 일치 |

**차이점**: 없음. 완전히 동일하게 설정됨.

---

### 2.2 Root/Body Penalties ✅

**목적**: 로봇의 기본 자세/움직임 제어

| Reward | rl_training | E2E | 상태 |
|--------|-------------|-----|------|
| **lin_vel_z_l2** | -2.0 | -2.0 | ✅ 일치 |
| **ang_vel_xy_l2** | -0.05 | -0.05 | ✅ 일치 |
| **flat_orientation_l2** | -5.0 | -5.0 | ✅ 일치 |
| **base_height_l2** | -10.0, target=0.35 | -10.0, target=0.3536 | ⚠️ target 차이 |
| **body_lin_acc_l2** | 0.0 (비활성화) | 없음 | rl_training만 정의됨 |

**차이점**:
- `base_height_l2`: target_height가 0.35 vs 0.3536 (사용자가 수동 수정)
- `body_lin_acc_l2`: rl_training에만 정의되어 있지만 비활성화

---

### 2.3 Joint Penalties ✅

**목적**: 관절 토크/속도/가속도 제한

| Reward | rl_training | E2E | 상태 |
|--------|-------------|-----|------|
| **joint_torques_l2** | -2.5e-5 | -2.5e-5 | ✅ 일치 |
| **joint_vel_l2** | **0.0 (비활성화)** | **-5.0e-4** | ⚠️ **차이** |
| **joint_acc_l2** | -1e-8 | -1e-8 | ✅ 일치 |
| **joint_deviation_l1** | -0.5 (HipX만) | -0.5 (HipX만) | ✅ 일치 |
| **joint_power** | -2e-5 | -2e-5 | ✅ 일치 |
| **joint_pos_limits** | -5.0 | -5.0 | ✅ 일치 |
| **joint_vel_limits** | 0.0 (비활성화) | 없음 | rl_training만 정의됨 |

**차이점**:
- `joint_vel_l2`: **rl_training은 비활성화, E2E는 활성화(-5.0e-4)** ⚠️
  - E2E에서 추가로 관절 속도 패널티를 적용함
  - 이는 더 부드러운 움직임을 유도할 수 있음

---

### 2.4 Action Penalties ✅

**목적**: 액션 변화율 제한

| Reward | rl_training | E2E | 상태 |
|--------|-------------|-----|------|
| **action_rate_l2** | -0.02 | -0.02 | ✅ 일치 |

**차이점**: 없음.

---

### 2.5 Feet/Contact Rewards ✅

**목적**: 보행 패턴 및 접촉 제어

| Reward | rl_training | E2E | 상태 |
|--------|-------------|-----|------|
| **feet_air_time** | 5.0, threshold=0.5 | 5.0, threshold=0.5 | ✅ 일치 |
| **feet_air_time_variance** | -8.0 | -8.0 | ✅ 일치 |
| **feet_gait** | 0.5 | 0.5 | ✅ 일치 |
| **feet_slide** | -0.05 | -0.05 | ✅ 일치 |
| **feet_height** | -0.2, target=0.05 | -0.2, target=0.05 | ✅ 일치 |
| **feet_height_body** | -2.5, target=-0.35 | -2.5, target=-0.35 | ✅ 일치 |
| **feet_contact_without_cmd** | 0.1 | 0.1 | ✅ 일치 |
| **contact_forces** | -0.1 (발만) | -0.1 (collision, 다리/몸통) | ⚠️ **대상 다름** |
| **undesired_contacts** | **-0.5** | **없음** | ⚠️ **차이** |

**차이점**:
1. **`contact_forces` vs `collision`**:
   - rl_training: `contact_forces` - 발만 감지 (feet_contact_sensor)
   - E2E: `collision` - 다리/몸통 감지 (link_collision_sensor)
   - Weight는 동일(-0.1)하지만 **대상이 다름**

2. **`undesired_contacts`**:
   - rl_training: 활성화됨 (weight=-0.5) - 발이 아닌 부위의 접촉 패널티
   - E2E: 의도적으로 추가하지 않음

---

### 2.6 Command-dependent Rewards ✅

**목적**: 명령이 작을 때의 행동 제어

| Reward | rl_training | E2E | 상태 |
|--------|-------------|-----|------|
| **stand_still** | -0.5, command_threshold=0.1 | -0.5, command_threshold=0.1 | ✅ 일치 |

**차이점**: 없음. 완전히 동일.

---

### 2.7 Mirror/Sync Rewards ✅

**목적**: 대칭성 유도

| Reward | rl_training | E2E | 상태 |
|--------|-------------|-----|------|
| **joint_mirror** | -0.05 | -0.05 | ✅ 일치 |

**차이점**: 없음.

---

### 2.8 General/Termination Rewards ⚠️

**목적**: 에피소드 종료 및 생존 유도

| Reward | rl_training | E2E | 상태 |
|--------|-------------|-----|------|
| **termination** | 없음 | -3.0 | ⚠️ **E2E만 있음** |
| **alive** | 없음 | 0.01 | ⚠️ **E2E만 있음** |
| **is_terminated** | 0.0 (비활성화) | 없음 | rl_training만 정의됨 |

**차이점**:
- **E2E에만 있는 항목**:
  - `termination` (-3.0): 종료 조건 발생 시 패널티
  - `alive` (0.01): 생존 보너스

---

## 3. 주요 차이점 요약

### 3.1 완전히 다른 항목 (4개)

1. **`joint_vel_l2`**: 
   - rl_training: 비활성화 (0.0)
   - E2E: 활성화 (-5.0e-4)
   - **영향**: E2E에서 관절 속도 패널티가 추가로 적용됨

2. **`undesired_contacts`**:
   - rl_training: 활성화 (-0.5)
   - E2E: 없음
   - **영향**: rl_training에서 발 외 부위 접촉에 패널티, E2E는 없음

3. **`termination`**:
   - rl_training: 없음
   - E2E: 활성화 (-3.0)
   - **영향**: E2E에서 종료 조건 발생 시 패널티

4. **`alive`**:
   - rl_training: 없음
   - E2E: 활성화 (0.01)
   - **영향**: E2E에서 생존 보너스 제공

### 3.2 동일하지만 구현이 다른 항목 (1개)

1. **`contact_forces` vs `collision`**:
   - Weight: 동일 (-0.1)
   - **대상**: rl_training은 발만, E2E는 다리/몸통
   - **센서**: rl_training은 `contact_forces` sensor, E2E는 `link_collision_sensor`

### 3.3 파라미터 차이 (1개)

1. **`base_height_l2` target_height**:
   - rl_training: 0.35
   - E2E: 0.3536
   - **비고**: 사용자가 수동으로 수정한 값

---

## 4. Reward 설계 철학의 차이

### 4.1 rl_training의 특징

1. **더 엄격한 접촉 제어**:
   - `undesired_contacts` (-0.5)로 발 외 부위 접촉을 명시적으로 패널티
   
2. **관절 속도 자유도**:
   - `joint_vel_l2` 비활성화로 관절 속도 제한이 없음

3. **종료 조건 제어 없음**:
   - `termination`, `alive` reward 없음
   - Terminations만으로 제어

### 4.2 E2E_locomotion_v2의 특징

1. **더 부드러운 움직임 유도**:
   - `joint_vel_l2` (-5.0e-4) 활성화로 관절 속도 패널티
   
2. **생존/종료 명시적 보상**:
   - `termination` (-3.0): 종료 조건 패널티
   - `alive` (0.01): 생존 보너스
   - Terminations뿐만 아니라 reward로도 제어

3. **접촉 제어 방식 차이**:
   - `undesired_contacts` 없음
   - 대신 `collision`이 다리/몸통에 적용 (발은 `feet_slide`로 별도 제어)

---

## 5. Weight 값 비교표

| Reward 항목 | rl_training | E2E | 차이 |
|-------------|-------------|-----|------|
| track_lin_vel_xy_exp | 3.5 | 3.5 | ✅ |
| track_ang_vel_z_exp | 1.5 | 1.5 | ✅ |
| lin_vel_z_l2 | -2.0 | -2.0 | ✅ |
| ang_vel_xy_l2 | -0.05 | -0.05 | ✅ |
| flat_orientation_l2 | -5.0 | -5.0 | ✅ |
| base_height_l2 | -10.0 | -10.0 | ✅ |
| joint_torques_l2 | -2.5e-5 | -2.5e-5 | ✅ |
| joint_vel_l2 | **0.0** | **-5.0e-4** | ⚠️ |
| joint_acc_l2 | -1e-8 | -1e-8 | ✅ |
| joint_deviation_l1 | -0.5 | -0.5 | ✅ |
| joint_power | -2e-5 | -2e-5 | ✅ |
| joint_pos_limits | -5.0 | -5.0 | ✅ |
| action_rate_l2 | -0.02 | -0.02 | ✅ |
| feet_air_time | 5.0 | 5.0 | ✅ |
| feet_air_time_variance | -8.0 | -8.0 | ✅ |
| feet_gait | 0.5 | 0.5 | ✅ |
| feet_slide | -0.05 | -0.05 | ✅ |
| stand_still | -0.5 | -0.5 | ✅ |
| feet_height_body | -2.5 | -2.5 | ✅ |
| feet_height | -0.2 | -0.2 | ✅ |
| feet_contact_without_cmd | 0.1 | 0.1 | ✅ |
| contact_forces/collision | -0.1 | -0.1 | ✅ (대상 다름) |
| undesired_contacts | **-0.5** | **없음** | ⚠️ |
| termination | **없음** | **-3.0** | ⚠️ |
| alive | **없음** | **0.01** | ⚠️ |

---

## 6. 실제 학습에 미치는 영향

### 6.1 E2E가 더 부드러운 움직임을 학습할 가능성

- **이유**: `joint_vel_l2` (-5.0e-4) 활성화
- **효과**: 관절 속도가 높을수록 패널티, 더 부드러운 움직임 유도

### 6.2 rl_training이 더 엄격한 접촉 제어를 학습할 가능성

- **이유**: `undesired_contacts` (-0.5) 활성화
- **효과**: 발 외 부위 접촉 시 명시적 패널티, 더 깔끔한 접촉 패턴 학습

### 6.3 E2E가 종료 조건을 더 명확히 학습할 가능성

- **이유**: `termination` (-3.0), `alive` (0.01) 보상
- **효과**: 종료 조건 발생 시 패널티, 생존 시 보너스로 더 명확한 학습 신호

---

## 7. 결론

### 7.1 전체적인 평가

**높은 유사도** (~95%):
- 대부분의 reward 항목과 weight 값이 동일
- 같은 목적의 reward들이 거의 동일하게 설정됨

**주요 차이점** (5개):
1. `joint_vel_l2`: E2E만 활성화
2. `undesired_contacts`: rl_training만 활성화
3. `termination`: E2E만 있음
4. `alive`: E2E만 있음
5. `contact_forces` vs `collision`: 대상 다름

### 7.2 권장사항

1. **E2E에 `undesired_contacts` 추가 고려**:
   - rl_training과 동일한 접촉 제어를 원한다면 추가 가능
   - 현재는 `collision`으로 대체 역할을 하고 있음

2. **`joint_vel_l2` 비활성화 고려**:
   - rl_training과 동일하게 하려면 비활성화 가능
   - 다만 E2E에서는 부드러운 움직임이 목표일 수 있음

3. **`contact_forces` vs `collision` 통일 고려**:
   - 목적에 따라 발만 감지할지, 다리/몸통도 감지할지 결정
   - 현재는 다른 전략을 사용 중

---

## 참고 자료

- rl_training rough task: `rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py`
- E2E_locomotion_v2: `E2E_locomotion_v2/env_cfg.py`
- 상세 비교: `E2E_locomotion_v2/REWARD_COMPARISON_FINAL.md`
