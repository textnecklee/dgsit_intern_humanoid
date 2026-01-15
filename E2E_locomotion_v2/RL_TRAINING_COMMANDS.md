# rl_training 실행 코드

## 1. 학습 (Training)

### 기본 명령어:
```bash
cd /home/teamquad/Desktop/Intern/IsaacLab/rl_training
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Rough-Deeprobotics-Lite3-v0
```

### 주요 옵션:
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --seed 42 \
    --experiment_name my_experiment \
    --run_name test_run
```

### 옵션 설명:
- `--task`: Task 이름 (필수)
  - `Rough-Deeprobotics-Lite3-v0` (Rough terrain) ✅ **올바른 이름**
  - `Flat-Deeprobotics-Lite3-v0` (Flat terrain) ✅ **올바른 이름**
- `--num_envs`: 환경 개수 (기본값: config에서 설정)
- `--max_iterations`: 최대 학습 iteration 수
- `--seed`: 랜덤 시드
- `--experiment_name`: 실험 폴더 이름
- `--run_name`: 실행 이름 (로그 폴더에 추가됨)
- `--video`: 비디오 녹화 활성화
- `--resume`: 체크포인트에서 재개
- `--load_run`: 재개할 run 이름
- `--checkpoint`: 재개할 체크포인트 파일

---

## 2. 실행/테스트 (Play)

### 기본 명령어 (방법 1: run 이름과 체크포인트 이름 지정):
```bash
cd /home/teamquad/Desktop/Intern/IsaacLab/rl_training
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --load_run 2025-01-15_10-30-45_test_run \
    --checkpoint model_1000.pt
```

### 기본 명령어 (방법 2: 체크포인트 파일 경로 직접 지정):
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --checkpoint /path/to/model_1000.pt
```

### 기본 명령어 (방법 3: 가장 최근 체크포인트 자동 선택):
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --load_run .* \
    --checkpoint .*
```

### 주요 옵션 예시:
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --load_run 2025-01-15_10-30-45_test_run \
    --checkpoint model_1000.pt \
    --num_envs 10 \
    --real-time \
    --keyboard
```

### 옵션 설명:
- `--task`: Task 이름 (필수)
  - `Rough-Deeprobotics-Lite3-v0` (Rough terrain)
  - `Flat-Deeprobotics-Lite3-v0` (Flat terrain)
- `--load_run`: 로드할 run 이름 (로그 폴더 이름, 예: `2025-01-15_10-30-45_test_run`)
  - `.*` 사용 시 가장 최근 run 자동 선택
- `--checkpoint`: 체크포인트 파일 이름 또는 전체 경로
  - 파일 이름만: `model_1000.pt` (run 폴더 내에서 찾음)
  - 전체 경로: `/path/to/model_1000.pt` (직접 지정)
  - `.*` 사용 시 가장 최근 체크포인트 자동 선택
- `--experiment_name`: 실험 이름 (기본값: `deeprobotics_lite3_rough` 또는 `deeprobotics_lite3_flat`)
- `--num_envs`: 환경 개수 (기본값: 50)
- `--real-time`: 실시간 실행
- `--keyboard`: 키보드로 제어 (단일 환경만, num_envs=1로 자동 설정)
- `--video`: 비디오 녹화
- `--use_pretrained_checkpoint`: 사전 학습된 체크포인트 사용

---

## 3. Task 이름 목록

### 등록된 Task:
- `Rough-Deeprobotics-Lite3-v0` ✅
  - Rough terrain 환경
  - 다양한 지형이 섞여있음
  
- `Flat-Deeprobotics-Lite3-v0` ✅
  - Flat terrain 환경
  - 평평한 지면

### Task 이름 형식:
```
{TerrainType}-{RobotName}-v0
```

---

## 4. 로그 위치

### 학습 로그:
```
logs/rsl_rl/{experiment_name}/{timestamp}_{run_name}/
├── model_*.pt          # 체크포인트 파일
├── params/
│   ├── env.yaml       # 환경 설정
│   └── agent.yaml     # 에이전트 설정
└── videos/            # 비디오 (--video 옵션 사용 시)
```

### 예시:
```
logs/rsl_rl/my_experiment/2025-01-15_10-30-45_test_run/
├── model_1000.pt
├── model_2000.pt
├── params/
│   ├── env.yaml
│   └── agent.yaml
└── videos/
    └── train/
        └── rl-video-episode-0.mp4
```

---

## 5. 실제 사용 예시

### 학습 시작:
```bash
cd /home/teamquad/Desktop/Intern/IsaacLab/rl_training
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --experiment_name deeprobotics_rough \
    --run_name baseline
```

### 학습 재개:
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --resume \
    --load_run 2025-01-15_10-30-45_baseline \
    --checkpoint model_1500.pt
```

### 학습된 모델 테스트:
```bash
# 방법 1: run 이름과 체크포인트 이름 지정
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --load_run 2026-01-15_20-34-18 \
    --checkpoint model_0.pt \
    --num_envs 10 \
    --real-time

# 방법 2: 가장 최근 체크포인트 자동 선택
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --load_run .* \
    --checkpoint .* \
    --num_envs 10 \
    --real-time

# 방법 3: 체크포인트 전체 경로 직접 지정
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --checkpoint /home/teamquad/Desktop/Intern/IsaacLab/rl_training/logs/rsl_rl/deeprobotics_lite3_rough/2026-01-15_20-34-18/model_0.pt \
    --num_envs 10 \
    --real-time
```

### 키보드로 제어:
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --load_run 2025-01-15_10-30-45_baseline \
    --checkpoint model_3000.pt \
    --keyboard
```

### 체크포인트 경로 구조:
```
logs/rsl_rl/{experiment_name}/{run_name}/model_*.pt
```

예시:
```
logs/rsl_rl/deeprobotics_lite3_rough/2025-01-15_10-30-45_baseline/model_3000.pt
```

따라서:
- `--experiment_name`: `deeprobotics_lite3_rough` (기본값, 생략 가능)
- `--load_run`: `2025-01-15_10-30-45_baseline`
- `--checkpoint`: `model_3000.pt`

---

## 6. 주의사항

1. **Isaac Sim 실행 필요**: 
   - 학습/테스트 전에 Isaac Sim Simulator를 먼저 실행해야 함
   - 코드 주석에 "Launch Isaac Sim Simulator first." 라고 명시됨

2. **작업 디렉토리**:
   - `rl_training` 폴더에서 실행해야 함
   - 상대 경로가 올바르게 작동함

3. **체크포인트 경로**:
   - `--load_run`: 로그 폴더 이름 (타임스탬프 포함)
   - `--checkpoint`: 체크포인트 파일 이름 (예: `model_1000.pt`)

4. **환경 개수**:
   - 학습: 많은 환경 (4096 등) 권장
   - 테스트: 적은 환경 (1~50) 권장
