# 학습 결과 확인 방법

## 1. TensorBoard로 학습 곡선 확인

학습 중에 TensorBoard 로그가 자동으로 저장됩니다. 다음 명령어로 확인할 수 있습니다:

```bash
# 프로젝트 루트에서 실행
cd /home/teamquad/Desktop/Intern/IsaacLab/E2E_locomotion_v2

# TensorBoard 실행
tensorboard --logdir runs/legbot_rslrl --port 6006
```

브라우저에서 `http://localhost:6006`으로 접속하면 다음을 확인할 수 있습니다:
- **Reward**: 에피소드별 평균 보상
- **Episode Length**: 에피소드 길이
- **Policy Loss**: 정책 손실
- **Value Loss**: 가치 함수 손실
- **Learning Rate**: 학습률 변화

## 2. 학습된 정책 실행 및 시각화

`eval_policy.py` 스크립트를 사용하여 학습된 정책을 실행하고 시각화할 수 있습니다:

```bash
# 기본 실행 (GUI로 시각화)
python E2E_locomotion_v2/scripts/eval_policy.py \
  --checkpoint runs/legbot_rslrl/YYYY-MM-DD_HH-MM-SS/model_XXXXX.pt \
  --num_envs 1 \
  --rollout_steps 0 \
  --device cuda

# 헤드리스 모드 (시각화 없이 실행)
python E2E_locomotion_v2/scripts/eval_policy.py \
  --checkpoint runs/legbot_rslrl/YYYY-MM-DD_HH-MM-SS/model_XXXXX.pt \
  --num_envs 200 \
  --rollout_steps 10000 \
  --device cuda \
  --headless
```

**파라미터 설명:**
- `--checkpoint`: 학습된 모델 체크포인트 경로 (필수)
- `--num_envs`: 병렬 환경 개수 (시각화: 1, 평가: 200+)
- `--rollout_steps`: 실행할 스텝 수 (0이면 무한 실행, 창을 닫으면 종료)
- `--device`: 사용할 디바이스 (cuda 또는 cpu)
- `--headless`: GUI 없이 실행 (선택)

## 3. 체크포인트 파일 확인

학습 중에 `save_interval`마다 체크포인트가 저장됩니다:

```bash
# 체크포인트 파일 목록 확인
ls -lh runs/legbot_rslrl/*/model_*.pt

# 최신 체크포인트 찾기
find runs/legbot_rslrl -name "model_*.pt" -type f | sort -V | tail -5
```

## 4. 학습 통계 확인

각 학습 실행 디렉토리에는 다음 파일들이 포함됩니다:
- `model_XXXXX.pt`: 모델 체크포인트
- `events.out.tfevents.*`: TensorBoard 이벤트 파일
- `config.yaml`: 학습 설정 파일

## 5. 여러 체크포인트 비교

여러 에포크의 체크포인트를 비교하여 최적의 모델을 찾을 수 있습니다:

```bash
# 여러 체크포인트를 순차적으로 평가
for checkpoint in runs/legbot_rslrl/YYYY-MM-DD_HH-MM-SS/model_*.pt; do
  echo "Evaluating: $checkpoint"
  python E2E_locomotion_v2/scripts/eval_policy.py \
    --checkpoint "$checkpoint" \
    --num_envs 50 \
    --rollout_steps 5000 \
    --device cuda \
    --headless
done
```

## 주의사항

1. **환경 설정 일치**: 평가 시 `env_cfg.py`가 학습 시와 동일해야 합니다.
2. **최적 체크포인트**: 마지막 체크포인트가 항상 최고 성능은 아닙니다. 여러 체크포인트를 테스트하세요.
3. **시각화 모드**: GUI 모드에서는 `--num_envs 1`을 권장합니다 (성능 문제 방지).
