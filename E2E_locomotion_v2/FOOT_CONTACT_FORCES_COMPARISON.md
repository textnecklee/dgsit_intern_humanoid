# Foot Contact Forces 보상 비교: E2E_locomotion_v2 vs rl_training

## 비교 결과

### ✅ 함수 구현 - 완전히 동일

**E2E_locomotion_v2** (`rew_foot_contact_forces`):
```python
def rew_foot_contact_forces(env, threshold: float, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    return torch.sum(violation.clip(min=0.0), dim=1)
```

**rl_training** (`contact_forces`):
```python
def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    return torch.sum(violation.clip(min=0.0), dim=1)
```

**결론**: ✅ **완전히 동일한 로직**

---

## 설정 비교

### E2E_locomotion_v2 (현재 주석 처리됨)

```python
# foot_contact_forces = RewTerm(
#     func=mdp.rew_foot_contact_forces,
#     weight=-0.5,  # ⚠️ rl_training과 다름
#     params={
#         "threshold": 100.0,  # ✅ rl_training과 동일
#         "sensor_cfg": SceneEntityCfg("feet_contact_sensor", body_names=[".*foot.*"]),  # ✅ 발만
#     },
# )
```

### rl_training

```python
self.rewards.contact_forces.weight = -1e-1  # -0.1
self.rewards.contact_forces.params["threshold"] = 100.0
self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]  # 발만
```

---

## 차이점 요약

| 항목 | E2E_locomotion_v2 | rl_training | 차이 |
|------|-------------------|-------------|------|
| **함수 구현** | `rew_foot_contact_forces` | `contact_forces` | ✅ 동일한 로직 |
| **weight** | `-0.5` (주석 처리) | `-0.1` | ⚠️ **5배 차이** |
| **threshold** | `100.0` | `100.0` | ✅ 동일 |
| **sensor** | `feet_contact_sensor` | `contact_forces` | ⚠️ 센서 이름만 다름 |
| **body_names** | `[".*foot.*"]` | `[foot_link_name]` | ✅ 발만 (동일) |
| **활성화 상태** | ❌ 주석 처리됨 | ✅ 활성화 | ⚠️ **차이** |

---

## 상세 분석

### 1. 함수 로직 - 완전히 동일 ✅

**계산 방식**:
1. `net_forces_w_history`에서 힘 히스토리 가져오기
2. 각 발의 힘 크기(norm) 계산
3. 히스토리에서 최대 힘 찾기
4. `violation = 최대 힘 - threshold` 계산
5. violation이 양수인 경우만 합산

**예시**:
- 발 힘: [120N, 80N, 150N, 90N]
- 최대 힘: 150N
- violation = 150 - 100 = 50
- 최종 반환값: 50

### 2. Weight 차이 - ⚠️ 중요

**E2E_locomotion_v2**: `-0.5`
- violation이 50이면 → 보상 = -0.5 × 50 = **-25.0**

**rl_training**: `-0.1`
- violation이 50이면 → 보상 = -0.1 × 50 = **-5.0**

**영향**:
- E2E_locomotion_v2가 **5배 더 강한 페널티**
- 발 접촉 힘에 대해 더 엄격하게 제한
- 점프 후 착지 시 더 큰 페널티

### 3. 센서 차이

**E2E_locomotion_v2**: `feet_contact_sensor`
- 발만 감지하는 전용 센서
- `prim_path="{ENV_REGEX_NS}/Robot/.*foot"`

**rl_training**: `contact_forces` 센서
- 모든 링크를 감지하지만 `body_names`로 발만 필터링
- `prim_path="{ENV_REGEX_NS}/Robot/.*"`

**영향**: 기능적으로는 동일 (둘 다 발만 측정)

### 4. 활성화 상태

**E2E_locomotion_v2**: ❌ 주석 처리됨
- 현재는 발 접촉 힘 페널티가 없음
- 점프 문제의 원인 중 하나일 수 있음

**rl_training**: ✅ 활성화됨
- 발 접촉 힘 페널티가 작동 중

---

## 권장 사항

### rl_training과 동일하게 설정하려면:

```python
foot_contact_forces = RewTerm(
    func=mdp.rew_foot_contact_forces,
    weight=-0.1,  # -0.5 → -0.1로 변경 (rl_training과 동일)
    params={
        "threshold": 100.0,
        "sensor_cfg": SceneEntityCfg("feet_contact_sensor", body_names=[".*foot.*"]),
    },
)
```

**또는 isaaclab의 기본 함수 사용**:
```python
foot_contact_forces = RewTerm(
    func=isaaclab_mdp.contact_forces,  # isaaclab 기본 함수
    weight=-0.1,
    params={
        "threshold": 100.0,
        "sensor_cfg": SceneEntityCfg("feet_contact_sensor", body_names=[".*foot.*"]),
    },
)
```

---

## 결론

1. **함수 로직**: ✅ 완전히 동일
2. **Weight**: ⚠️ E2E_locomotion_v2가 5배 더 강함 (-0.5 vs -0.1)
3. **활성화 상태**: ⚠️ E2E_locomotion_v2는 주석 처리됨
4. **센서**: 기능적으로 동일 (발만 측정)

**현재 E2E_locomotion_v2에는 발 접촉 힘 페널티가 없어서, 점프 후 착지 시 제약이 약할 수 있습니다.**

rl_training과 동일하게 하려면:
1. 주석 해제
2. weight를 -0.1로 변경
