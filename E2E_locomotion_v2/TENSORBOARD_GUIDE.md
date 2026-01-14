# TensorBoard 그래프 확인 가이드

## ❌ pt 파일만으로는 안됩니다!

TensorBoard는 **이벤트 파일(events.out.tfevents.*)**을 읽어서 그래프를 표시합니다.
- `model_XXXXX.pt`: 모델 체크포인트 (정책 실행용)
- `events.out.tfevents.*`: TensorBoard 이벤트 파일 (그래프 표시용)

## ✅ 필요한 파일들

TensorBoard를 실행하려면 **전체 log 디렉토리**가 필요합니다:

```
runs/legbot_rslrl/YYYY-MM-DD_HH-MM-SS/
├── model_0.pt              # 체크포인트 (필수 아님)
├── model_300.pt            # 체크포인트 (필수 아님)
├── model_600.pt            # 체크포인트 (필수 아님)
├── events.out.tfevents.*    # ⭐ TensorBoard 이벤트 파일 (필수!)
├── config.yaml              # 설정 파일 (선택)
└── ... (기타 파일들)
```

## 📊 TensorBoard 실행 방법

### 방법 1: 특정 학습 실행 디렉토리 지정

```bash
cd /home/teamquad/Desktop/Intern/IsaacLab/E2E_locomotion_v2

# 특정 실행의 TensorBoard 보기
tensorboard --logdir runs/legbot_rslrl/2026-01-12_21-17-39 --port 6006
```

### 방법 2: 모든 학습 실행 비교 (권장)

```bash
cd /home/teamquad/Desktop/Intern/IsaacLab/E2E_locomotion_v2

# 모든 실행을 한번에 비교
tensorboard --logdir runs/legbot_rslrl --port 6006
```

이렇게 하면 여러 학습 실행을 비교할 수 있습니다.

## 🔍 확인할 수 있는 그래프

TensorBoard에서 다음을 확인할 수 있습니다:

1. **Train/mean_reward**: 평균 보상 (학습 진행도)
2. **Train/mean_episode_length**: 평균 에피소드 길이
3. **Loss/value_loss**: 가치 함수 손실
4. **Loss/surrogate_loss**: 정책 손실
5. **Loss/learning_rate**: 학습률
6. **Perf/total_fps**: 성능 (FPS)
7. **Metrics/**: 커스텀 메트릭들

## ⚠️ 주의사항

1. **이벤트 파일이 없으면 그래프를 볼 수 없습니다**
   - pt 파일만 있어서는 TensorBoard 그래프를 볼 수 없음
   - 학습 중에 자동으로 생성되는 `events.out.tfevents.*` 파일이 필요

2. **학습이 완료되어야 이벤트 파일이 생성됩니다**
   - 학습 중에도 실시간으로 업데이트됨
   - 학습이 중단되어도 지금까지의 로그는 확인 가능

3. **log_dir 전체 디렉토리가 필요합니다**
   - pt 파일만 복사해서는 안됨
   - 전체 디렉토리를 보관해야 함

## 📁 파일 구조 예시

```
runs/legbot_rslrl/
└── 2026-01-12_21-17-39/          # 학습 실행 디렉토리
    ├── events.out.tfevents.1234567890.12345  # ⭐ TensorBoard 이벤트 파일
    ├── model_0.pt
    ├── model_300.pt
    ├── model_600.pt
    └── ...
```

## 🚀 빠른 확인 방법

```bash
# 1. 이벤트 파일이 있는지 확인
ls -lh runs/legbot_rslrl/*/events.out.tfevents.*

# 2. TensorBoard 실행
tensorboard --logdir runs/legbot_rslrl --port 6006

# 3. 브라우저에서 접속
# http://localhost:6006
```

## 💡 팁

- **여러 실행 비교**: `--logdir runs/legbot_rslrl`로 모든 실행을 한번에 비교
- **특정 실행만**: `--logdir runs/legbot_rslrl/2026-01-12_21-17-39`로 특정 실행만 보기
- **원격 접속**: `--host 0.0.0.0` 옵션으로 원격에서도 접속 가능
