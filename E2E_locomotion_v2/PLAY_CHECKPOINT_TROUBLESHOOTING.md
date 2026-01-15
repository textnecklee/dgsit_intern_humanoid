# Play 명령어 체크포인트 찾기 문제 해결

## 에러 메시지
```
FileNotFoundError: Unable to find the file: model_1000.pt
```

## 원인
`--checkpoint model_1000.pt`를 지정했지만, 실제로는 해당 파일이 존재하지 않습니다.

## 해결 방법

### 1. 실제 체크포인트 파일 확인
```bash
# 로그 디렉터리 확인
ls -la /home/teamquad/Desktop/Intern/IsaacLab/rl_training/logs/rsl_rl/deeprobotics_lite3_rough/

# run 폴더 확인
ls -la /home/teamquad/Desktop/Intern/IsaacLab/rl_training/logs/rsl_rl/deeprobotics_lite3_rough/2026-01-15_20-34-18/

# 체크포인트 파일 찾기
find /home/teamquad/Desktop/Intern/IsaacLab/rl_training/logs/rsl_rl/deeprobotics_lite3_rough/ -name "*.pt"
```

### 2. 올바른 명령어 사용

#### 방법 1: run 이름과 체크포인트 이름 지정 (권장)
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --load_run 2026-01-15_20-34-18 \
    --checkpoint model_0.pt
```

#### 방법 2: 가장 최근 체크포인트 자동 선택
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --load_run .* \
    --checkpoint .*
```

#### 방법 3: 체크포인트 전체 경로 직접 지정
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --checkpoint /home/teamquad/Desktop/Intern/IsaacLab/rl_training/logs/rsl_rl/deeprobotics_lite3_rough/2026-01-15_20-34-18/model_0.pt
```

### 3. play.py의 체크포인트 찾기 로직

```python
if args_cli.use_pretrained_checkpoint:
    # 사전 학습된 체크포인트 사용
    resume_path = get_published_pretrained_checkpoint(...)
elif args_cli.checkpoint:
    # --checkpoint가 지정되면: 전체 경로 또는 파일 이름만
    resume_path = retrieve_file_path(args_cli.checkpoint)
    # 이 함수는 절대 경로 또는 상대 경로를 찾음
    # 파일 이름만 지정하면 현재 디렉터리에서 찾으려고 함 → 실패!
else:
    # --checkpoint가 없으면: run 이름과 체크포인트 이름으로 찾기
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
```

### 4. 주의사항

- `--checkpoint model_1000.pt`만 지정하면:
  - `retrieve_file_path`가 현재 디렉터리에서 찾으려고 함
  - 실제 파일은 run 폴더 안에 있음 → 실패!

- 올바른 사용법:
  - `--load_run <run_name> --checkpoint model_0.pt` (run 폴더 내에서 찾음)
  - 또는 `--checkpoint /full/path/to/model_0.pt` (전체 경로)

### 5. 현재 상황에 맞는 명령어

실제 체크포인트 파일: `model_0.pt` (iteration 0)

```bash
cd /home/teamquad/Desktop/Intern/IsaacLab/rl_training
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --load_run 2026-01-15_20-34-18 \
    --checkpoint model_0.pt \
    --num_envs 10 \
    --real-time
```

또는 가장 최근 체크포인트 자동 선택:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --load_run .* \
    --checkpoint .*
```
