# 점프 동작 문제 분석 및 해결 방안

## 문제 현상
로봇이 계속 점프하는 동작을 수행함

## 원인 분석

### 1. `feet_air_time` 보상이 과도함 ⚠️ **주요 원인**
- **현재 설정**: `weight=5.0`, `threshold=0.5`
- **문제점**: 발이 0.5초 이상 공중에 있으면 보상을 주는 구조가 점프를 유도
- **위치**: `env_cfg.py:488-496`

### 2. 보상 가중치 불균형
- `feet_air_time`: +5.0 (점프 유도)
- `lin_vel_z`: -2.0 (수직 속도 페널티) - 너무 약함
- `base_height`: -10.0 (높이 페널티) - 하지만 초기 높이(0.36)와 목표(0.35) 불일치

### 3. 초기 높이와 목표 높이 불일치
- 초기 높이: `z=0.36`
- 목표 높이: `0.35`
- 위치: `env_cfg.py:102`

### 4. `feet_air_time_variance` 보상의 한계
- 모든 발이 동시에 점프하면 분산이 작아 페널티가 약함
- 위치: `env_cfg.py:498-504`

## 해결 방안

### 방안 1: `feet_air_time` 보상 가중치 감소 (권장)
```python
feet_air_time = RewTerm(
    func=mdp.rew_feet_air_time,
    weight=1.0,  # 5.0 → 1.0으로 감소
    params={
        "threshold": 0.3,  # 0.5 → 0.3으로 감소 (더 짧은 공중 시간만 보상)
    },
)
```

### 방안 2: 수직 속도 페널티 강화
```python
lin_vel_z = RewTerm(
    func=mdp.rew_lin_vel_z,
    weight=-5.0,  # -2.0 → -5.0으로 강화
)
```

### 방안 3: 초기 높이와 목표 높이 일치
```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0, 0, 0.35),  # 0.36 → 0.35로 변경
    ...
)
```

### 방안 4: `base_height` 보상 조정
```python
base_height = RewTerm(
    func=mdp.rew_base_height,
    weight=-15.0,  # -10.0 → -15.0으로 강화
    params={
        "target_height": 0.36,  # 초기 높이와 일치시키거나
        # 또는 0.35로 유지하고 초기 높이를 0.35로 변경
    },
)
```

### 방안 5: `feet_air_time_variance` 보상 강화
```python
feet_air_time_variance = RewTerm(
    func=mdp.rew_feet_air_time_variance,
    weight=-12.0,  # -8.0 → -12.0으로 강화
)
```

## 권장 조합
1. `feet_air_time` weight: 5.0 → 1.0
2. `feet_air_time` threshold: 0.5 → 0.3
3. `lin_vel_z` weight: -2.0 → -5.0
4. 초기 높이: 0.36 → 0.35 (목표 높이와 일치)

## 테스트 순서
1. 먼저 `feet_air_time` 보상만 조정하여 테스트
2. 효과가 없으면 `lin_vel_z` 페널티 강화
3. 여전히 문제가 있으면 초기 높이 조정

---

## 왜 rl_training은 괜찮은가?

### 핵심 차이점 분석

rl_training과 E2E_locomotion_v2의 보상 설정을 비교한 결과, **동일한 가중치를 사용하고 있음에도** rl_training에서 점프 문제가 없는 이유는 다음과 같습니다:

#### 1. **환경 차이: Rough Terrain vs Flat Terrain** ⚠️ **가장 중요한 차이**

**rl_training (DeeproboticsLite3RoughEnvCfg)**:
- **Rough terrain 환경**: 불규칙한 지형 (random_rough, hf_pyramid_slope 등)
- 점프하면 불안정해지고 넘어질 위험이 높음
- 자연스럽게 점프를 억제하는 환경적 제약

**E2E_locomotion_v2**:
- **Flat terrain 환경**: 평평한 지면
- 점프해도 안정적으로 착지 가능
- 환경적 제약이 없어 점프가 유리한 전략이 될 수 있음

#### 2. **`contact_forces` 보상의 존재** ⚠️ **두 번째로 중요한 차이**

**rl_training**:
```python
self.rewards.contact_forces.weight = -1e-1  # -0.1
self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
```
- 발이 땅에 닿을 때 큰 힘이 가해지면 페널티
- 점프 후 착지 시 큰 충격력이 발생하면 강한 페널티
- 점프를 자연스럽게 억제

**E2E_locomotion_v2**:
```python
collision = RewTerm(
    func=mdp.rew_collision,
    weight=-0.1,
    params={
        "sensor_cfg": SceneEntityCfg("link_collision_sensor"),  # 발이 아닌 링크 충돌
    },
)
```
- `collision` 보상은 있지만, `link_collision_sensor`를 사용 (발 접촉과는 다름)
- 발 접촉 힘에 대한 직접적인 페널티가 없음
- 점프 후 착지 시 페널티가 약함

#### 3. **보상 항목 비교**

| 보상 항목 | rl_training | E2E_locomotion_v2 | 차이점 |
|---------|-------------|-------------------|--------|
| `feet_air_time` | 5.0 | 5.0 | 동일 |
| `base_height_l2` | -10.0 | -10.0 | 동일 |
| `lin_vel_z_l2` | -2.0 | -2.0 | 동일 |
| `contact_forces` | **-0.1** | ❌ 없음 | **차이** |
| `collision` | ❌ 없음 | -0.1 (link collision) | 다름 |
| `feet_slide` | -0.05 | -0.05 | 동일 |
| `stand_still` | -0.5 | -0.5 | 동일 |
| `feet_height_body` | -2.5 | -2.5 | 동일 |

#### 4. **환경적 제약의 효과**

**Rough Terrain의 자연스러운 억제 효과**:
- 점프하면 불규칙한 지형에 불안정하게 착지
- 넘어질 위험이 높아짐
- 정책이 자연스럽게 점프를 피하도록 학습

**Flat Terrain의 문제**:
- 점프해도 안정적으로 착지 가능
- `feet_air_time` 보상(+5.0)이 점프를 유도
- `contact_forces` 페널티가 없어 점프 후 착지 시 제약이 약함

### 결론: 왜 rl_training은 괜찮은가?

1. **환경적 제약**: Rough terrain이 점프를 자연스럽게 억제
2. **`contact_forces` 보상**: 발 접촉 힘에 대한 직접적인 페널티로 점프 후 착지 시 제약
3. **불안정성**: 점프하면 넘어질 위험이 높아 정책이 점프를 피하도록 학습

### E2E_locomotion_v2에서의 해결 방안

Flat terrain 환경에서는 환경적 제약이 없으므로, 다음 중 하나 이상을 적용해야 합니다:

#### 방안 A: `contact_forces` 보상 추가 (rl_training과 동일하게) ⭐ **가장 권장**

E2E_locomotion_v2에는 이미 `rew_collision` 함수가 있고, 이것이 `contact_forces`와 동일한 로직입니다. 
다만 현재는 `link_collision_sensor`를 사용하고 있어서 발 접촉과는 다릅니다.

**방법 1: 기존 `collision` 보상을 발에만 적용하도록 변경**
```python
collision = RewTerm(
    func=mdp.rew_collision,
    weight=-0.1,  # rl_training과 동일
    params={
        "threshold": 100.0,
        "sensor_cfg": SceneEntityCfg("feet_contact_sensor"),  # link_collision_sensor → feet_contact_sensor
    },
)
```

**방법 2: `contact_forces` 보상을 별도로 추가 (rl_training과 완전히 동일)**
```python
# isaaclab.envs.mdp.rewards의 contact_forces 함수 사용
from isaaclab.envs import mdp as isaaclab_mdp

contact_forces = RewTerm(
    func=isaaclab_mdp.contact_forces,  # 또는 mdp.rew_collision 사용 가능
    weight=-0.1,
    params={
        "sensor_cfg": SceneEntityCfg("feet_contact_sensor"),
        "threshold": 100.0,
    },
)
```

**참고**: `rew_collision`과 `contact_forces`는 동일한 로직입니다:
- 둘 다 `net_forces_w_history`를 사용
- 둘 다 `threshold`를 초과하는 힘의 양(violation amount)을 페널티
- 차이점: `rew_collision`은 E2E_locomotion_v2에 정의되어 있고, `contact_forces`는 isaaclab 기본 함수

#### 방안 B: `feet_air_time` 보상 감소 (이미 제안한 방안)
```python
feet_air_time = RewTerm(
    func=mdp.rew_feet_air_time,
    weight=1.0,  # 5.0 → 1.0
    params={
        "threshold": 0.3,  # 0.5 → 0.3
    },
)
```

#### 방안 C: `lin_vel_z` 페널티 강화
```python
lin_vel_z = RewTerm(
    func=mdp.rew_lin_vel_z,
    weight=-5.0,  # -2.0 → -5.0
)
```

### 권장 조합 (Flat Terrain용)

1. `contact_forces` 보상 추가: `-0.1` (rl_training과 동일)
2. `feet_air_time` weight 감소: `5.0 → 1.0`
3. `lin_vel_z` 페널티 강화: `-2.0 → -5.0`

이렇게 하면 rl_training과 유사한 보상 구조를 가지면서도 flat terrain 환경에서 점프를 억제할 수 있습니다.
