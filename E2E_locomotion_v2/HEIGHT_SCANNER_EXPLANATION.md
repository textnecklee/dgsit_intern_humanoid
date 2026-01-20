# Height Scanner 파라미터 설명

## 개요

Rough terrain 환경에서 로봇이 지형의 높이를 감지하기 위해 사용하는 RayCaster 센서입니다. 두 개의 센서가 서로 다른 목적으로 사용됩니다.

---

## 1. `height_scanner` (관측용)

**목적**: 정책(Policy)이 지형 정보를 관측하기 위한 센서

```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",  # base link에 부착
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # base link 위 20m에서 시작
    ray_alignment="yaw",  # yaw 회전만 추적 (pitch/roll 무시)
    pattern_cfg=patterns.GridPatternCfg(
        resolution=0.1,      # 그리드 해상도: 0.1m 간격
        size=[1.6, 1.0]     # 그리드 크기: 1.6m x 1.0m (전방 x 좌우)
    ),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],  # 지형 메시에 레이 캐스팅
)
```

### 파라미터 상세 설명

#### `prim_path`
- **의미**: 센서가 부착될 링크의 경로
- **값**: `"{ENV_REGEX_NS}/Robot/base_link"`
- **설명**: 로봇의 base link에 센서를 부착. `{ENV_REGEX_NS}`는 환경별 네임스페이스로 자동 치환됨

#### `offset`
- **의미**: 센서 프레임의 부모 프레임(base link)으로부터의 오프셋
- **값**: `pos=(0.0, 0.0, 20.0)`
- **설명**: 
  - `(0.0, 0.0, 20.0)`: base link의 위치에서 **위로 20m 올라간 지점**에서 레이를 발사
  - 레이는 **위에서 아래로** (`direction=(0.0, 0.0, -1.0)`) 쏘아 지형 높이를 측정
  - **왜 20m인가?**
    1. **안전 마진**: 로봇이 점프하거나 높은 지형 위에 있어도 레이가 지형을 확실히 감지
    2. **레이 방향**: 위에서 아래로 쏘므로, 시작점이 높을수록 더 넓은 범위를 커버
    3. **계산 효율**: 너무 높으면 불필요하지만, 20m는 합리적인 값 (실제로는 5-10m도 가능)
    4. **관례**: Isaac Lab의 모든 예제에서 20m를 표준으로 사용
    5. **max_distance**: 기본값이 1e6m이므로 레이 거리 제한은 없지만, 시작점이 높으면 더 안전
  - **더 낮은 값도 가능**: 5m, 10m 등으로 줄일 수 있지만, 높은 지형이나 점프 시 감지 실패 가능

#### `ray_alignment`
- **의미**: 레이의 정렬 방식
- **값**: `"yaw"`
- **옵션**:
  - `"base"`: 전체 자세(roll, pitch, yaw) 추적
  - `"yaw"`: yaw 회전만 추적 (pitch/roll 무시) - **지형 스캔에 적합**
  - `"world"`: 월드 프레임 고정
- **설명**: 로봇이 기울어져도 레이 패턴이 수평을 유지하도록 yaw만 추적

#### `pattern_cfg` (GridPatternCfg)
- **의미**: 레이를 발사하는 패턴 설정

##### `resolution`
- **의미**: 그리드 해상도 (레이 간격)
- **값**: `0.1` (10cm)
- **설명**: 10cm 간격으로 레이를 배치. 해상도가 높을수록 더 정밀하지만 계산 비용 증가

##### `size`
- **의미**: 그리드 크기
- **값**: `[1.6, 1.0]` (길이 x 너비, 단위: m)
- **설명**: 
  - `1.6m`: 전방 방향 범위
  - `1.0m`: 좌우 방향 범위
  - 로봇 전방의 넓은 영역을 스캔하여 지형 정보 제공

#### `mesh_prim_paths`
- **의미**: 레이 캐스팅 대상 메시 경로
- **값**: `["/World/ground"]`
- **설명**: 지형 메시에 레이를 쏘아 충돌 지점의 높이를 측정

---

## 2. `height_scanner_base` (보상 계산용)

**목적**: `base_height` 보상 계산 시 로봇 바로 아래 지형 높이를 정밀하게 측정

```python
height_scanner_base = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",  # base link에 부착
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # base link 위 20m에서 시작
    ray_alignment="yaw",  # yaw 회전만 추적
    pattern_cfg=patterns.GridPatternCfg(
        resolution=0.05,     # 그리드 해상도: 0.05m 간격 (더 정밀)
        size=(0.1, 0.1)      # 그리드 크기: 0.1m x 0.1m (작은 영역)
    ),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)
```

### `height_scanner`와의 차이점

| 파라미터 | height_scanner | height_scanner_base | 차이점 |
|---------|----------------|-------------------|--------|
| **목적** | 관측용 (정책 입력) | 보상 계산용 | 다름 |
| **resolution** | 0.1m | 0.05m | **2배 더 정밀** |
| **size** | 1.6m x 1.0m | 0.1m x 0.1m | **16배 작은 영역** |

### 왜 두 개가 필요한가?

1. **`height_scanner` (넓은 범위)**
   - 정책이 전방 지형을 미리 파악하여 보행 계획 수립
   - 넓은 영역(1.6m x 1.0m)을 스캔하여 장애물, 경사 등을 감지
   - 관측치(observation)로 사용되어 정책 네트워크에 입력

2. **`height_scanner_base` (정밀 측정)**
   - 로봇 **바로 아래** 지형 높이만 정밀하게 측정
   - `base_height` 보상 계산 시 사용:
     ```python
     # rew_base_height 함수에서
     adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
     # 로봇 높이 = 목표 높이 + 지형 높이 보정
     ```
   - 작은 영역(0.1m x 0.1m)만 스캔하여 계산 효율성 확보

---

## 3. 보상 계산에서의 사용

### `base_height` 보상

```python
base_height = RewTerm(
    func=mdp.rew_base_height,
    weight=-10.0,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "target_height": 0.3536,  # 목표 높이 (평지 기준)
        "sensor_cfg": SceneEntityCfg("height_scanner_base"),  # 지형 높이 보정
    },
)
```

**동작 방식**:
1. `height_scanner_base`가 로봇 바로 아래 지형 높이를 측정
2. `target_height`에 지형 높이를 더하여 조정된 목표 높이 계산
3. 로봇의 실제 높이와 조정된 목표 높이의 차이를 페널티

**예시**:
- 평지: 목표 높이 = 0.3536m
- 경사면(지형 높이 +0.1m): 목표 높이 = 0.3536 + 0.1 = 0.4536m
- 로봇이 경사면에 맞춰 높이를 조절하면 페널티 없음

---

## 4. 파라미터 튜닝 가이드

### `height_scanner` (관측용)

- **`resolution` 감소 (0.1 → 0.05)**: 더 정밀한 지형 정보, 하지만 계산 비용 증가
- **`size` 증가 ([1.6, 1.0] → [2.0, 1.5])**: 더 넓은 전방 시야, 하지만 관측치 차원 증가
- **`offset` z 값 감소 (20.0 → 10.0)**: 계산 효율 향상, 하지만 높은 지형 감지 실패 가능

### `height_scanner_base` (보상용)

- **`resolution` 감소 (0.05 → 0.02)**: 더 정밀한 높이 측정, 하지만 보상 계산 비용 증가
- **`size` 증가 ((0.1, 0.1) → (0.2, 0.2))**: 평균화 효과로 안정적, 하지만 지형 변화 반영 느림

---

## 5. 주의사항

1. **RayCaster는 정규식 패턴을 허용하지 않음**
   - `prim_path`에 `".*torso.*"` 같은 정규식 사용 불가
   - 구체적인 링크 이름 사용: `"{ENV_REGEX_NS}/Robot/base_link"`

2. **`__post_init__`에서 `prim_path` 설정**
   - 초기값은 placeholder로 설정
   - `__post_init__`에서 실제 base link 이름으로 변경

3. **`mesh_prim_paths`는 지형 메시 경로**
   - Rough terrain: `["/World/ground"]`
   - Flat terrain: 사용하지 않음 (센서 자체가 없음)

---

## 참고

- rl_training과 동일한 설정 사용
- `height_scanner`: 관측용 (넓은 범위)
- `height_scanner_base`: 보상 계산용 (정밀 측정)
