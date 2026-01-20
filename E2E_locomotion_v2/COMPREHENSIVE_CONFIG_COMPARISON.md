# E2E_locomotion_v2 vs rl_training (Rough) 전체 설정 비교

## 목차
1. [Simulation 설정](#1-simulation-설정)
2. [Scene 설정](#2-scene-설정)
3. [Robot Asset 설정](#3-robot-asset-설정)
4. [Observations 설정](#4-observations-설정)
5. [Actions 설정](#5-actions-설정)
6. [Commands 설정](#6-commands-설정)
7. [Rewards 설정](#7-rewards-설정)
8. [Terminations 설정](#8-terminations-설정)
9. [Events 설정](#9-events-설정)
10. [Curriculum 설정](#10-curriculum-설정)
11. [기타 설정](#11-기타-설정)

---

## 1. Simulation 설정

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **dt** | `1/120` ≈ `0.00833` | `0.005` | E2E가 더 큰 timestep (약 1.67배) |
| **decimation** | `2` | `4` | rl_training이 2배 더 큰 decimation |
| **step_dt** | `1/120 * 2 = 0.01667` | `0.005 * 4 = 0.02` | rl_training이 약간 더 큰 환경 step |
| **episode_length_s** | `20` | `20.0` | 동일 |
| **max_episode_length** | `20 / 0.01667 = 1200` steps | `20 / 0.02 = 1000` steps | E2E가 더 많은 step |
| **render_interval** | `2` | `4` | rl_training이 더 큰 render interval |
| **physx.max_position_iteration_count** | 설정 없음 (기본값) | `4` | rl_training만 명시적 설정 |
| **physx.max_velocity_iteration_count** | 설정 없음 (기본값) | `1` | rl_training만 명시적 설정 |
| **physx.gpu_max_rigid_patch_count** | 설정 없음 (기본값) | `10 * 2**15` | rl_training만 명시적 설정 |

**영향**:
- E2E는 더 작은 dt로 더 정밀한 물리 시뮬레이션
- rl_training은 더 큰 decimation으로 더 빠른 학습 (더 적은 step)

---

## 2. Scene 설정

### 2.1 Terrain

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **terrain_type** | `"plane"` (GroundPlaneCfg) | `"generator"` (TerrainImporterCfg) | E2E는 평지, rl_training은 rough terrain |
| **terrain_generator** | 없음 | `ROUGH_TERRAINS_CFG` | rl_training만 terrain generator 사용 |
| **max_init_terrain_level** | 없음 | `5` | rl_training만 curriculum terrain |
| **physics_material** | 없음 | `static_friction=1.0, dynamic_friction=1.0, restitution=1.0` | rl_training만 명시적 설정 |
| **visual_material** | 없음 | MdlFileCfg (마블 텍스처) | rl_training만 시각적 재질 |

### 2.2 Sensors

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **feet_contact_sensor** | `ContactSensorCfg` | `contact_forces` (이름 다름) | 이름만 다름 |
| **prim_path** | `"{ENV_REGEX_NS}/Robot/.*foot"` | `"{ENV_REGEX_NS}/Robot/.*"` | rl_training이 모든 링크 감지 |
| **history_length** | `3` | `3` | 동일 |
| **track_air_time** | `True` | `True` | 동일 |
| **force_threshold** | `200.0` | 기본값 (1.0) | E2E가 더 높은 threshold |
| **update_period** | `0.0` | `sim.dt` (0.005) | rl_training이 명시적 설정 |
| **link_collision_sensor** | 있음 (`.*(torso\|thigh\|shank).*`) | 없음 | E2E만 별도 collision sensor |
| **height_scanner** | 없음 | 있음 (`RayCasterCfg`) | rl_training만 height scanner |
| **height_scanner_base** | 없음 | 있음 (`RayCasterCfg`) | rl_training만 base height scanner |

### 2.3 Lights

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **dome_light** | `DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0)` | 없음 | E2E만 dome light |
| **sky_light** | 없음 | `DomeLightCfg(intensity=750.0, texture_file=...)` | rl_training만 sky light (HDR 텍스처) |

### 2.4 Environment Settings

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **num_envs** | `1` | `4096` | rl_training이 훨씬 많은 환경 |
| **env_spacing** | `2.0` | `2.5` | rl_training이 더 큰 간격 |

---

## 3. Robot Asset 설정

### 3.1 USD 파일

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **usd_path** | `/home/teamquad/Desktop/Intern/urdf/Quad_v2_serial_v4/Quad_v2_serial _v4.usd` | `ISAACLAB_ASSETS_DATA_DIR/Lite3/Lite3_usd/Lite3.usd` | 완전히 다른 로봇 모델 |
| **class_type** | `QuadArticulation` (커스텀) | 기본 `Articulation` | E2E는 커스텀 클래스 (joint limits 수정) |

### 3.2 Rigid Body Properties

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **disable_gravity** | `False` | `False` | 동일 |
| **max_depenetration_velocity** | `5.0` | `1.0` | E2E가 더 큰 값 |
| **retain_accelerations** | 설정 없음 | `False` | rl_training만 명시적 설정 |
| **linear_damping** | 설정 없음 | `0.0` | rl_training만 명시적 설정 |
| **angular_damping** | 설정 없음 | `0.0` | rl_training만 명시적 설정 |
| **max_linear_velocity** | 설정 없음 | `1000.0` | rl_training만 명시적 설정 |
| **max_angular_velocity** | 설정 없음 | `1000.0` | rl_training만 명시적 설정 |

### 3.3 Articulation Properties

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **enabled_self_collisions** | `False` | `False` | 동일 |
| **solver_position_iteration_count** | `8` | `4` | E2E가 더 많은 iteration |
| **solver_velocity_iteration_count** | `0` | `1` | E2E는 0, rl_training은 1 |

### 3.4 Initial State

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **pos (z)** | `0.36` | `0.35` | E2E가 1cm 더 높음 |
| **joint_pos (HipX)** | `0.0` | `0.0` | 동일 |
| **joint_pos (HipY)** | `0.785` (45°) | `-0.8` (-45.8°) | **반대 방향!** |
| **joint_pos (Knee)** | `1.571` (90°) | `1.6` (91.7°) | 거의 동일 |

### 3.5 Actuators

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **actuator_type** | `ImplicitActuatorCfg` | `DelayedPDActuatorCfg` | **완전히 다른 액추에이터!** |
| **stiffness** | `40` | `30.0` | E2E가 더 높은 stiffness |
| **damping** | `1` | `1.0` | 동일 |
| **effort_limit_sim** | `100.0` | `24.0` (Hip), `36.0` (Knee) | E2E가 훨씬 큰 effort limit |
| **velocity_limit_sim** | `100.0` | `26.2` (Hip), `17.3` (Knee) | E2E가 훨씬 큰 velocity limit |
| **min_delay / max_delay** | 없음 | `0 / 5` | rl_training만 delay (실제 하드웨어 모델링) |
| **friction** | 없음 | `0.0` | rl_training만 명시적 설정 |
| **armature** | 없음 | `0.0` | rl_training만 명시적 설정 |

### 3.6 Joint Names

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **naming** | `FLHAA`, `FLHIP`, `FLKNEE` | `FL_HipX_joint`, `FL_HipY_joint`, `FL_Knee_joint` | **다른 네이밍 컨벤션** |

---

## 4. Observations 설정

### 4.1 Observation Terms

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **base_lin_vel** | 있음 | `None` (비활성) | E2E만 활성 |
| **base_ang_vel** | 있음 | 있음 | 둘 다 활성 |
| **base_ang_vel.scale** | 없음 (기본값 1.0) | `0.25` | rl_training이 0.25배 스케일 |
| **projected_gravity** | 없음 | 있음 | rl_training만 활성 |
| **base_z_pos** | 있음 | 없음 | E2E만 활성 |
| **joint_pos** | 있음 (`joint_pos_rel`) | 있음 (`joint_pos_rel`) | 둘 다 활성 |
| **joint_pos.scale** | 없음 (기본값 1.0) | `1.0` | 동일 |
| **joint_pos.noise** | 없음 | `Unoise(n_min=-0.01, n_max=0.01)` | rl_training만 노이즈 |
| **joint_vel** | 있음 (`joint_vel_rel`) | 있음 (`joint_vel_rel`) | 둘 다 활성 |
| **joint_vel.scale** | 없음 (기본값 1.0) | `0.05` | rl_training이 0.05배 스케일 |
| **joint_vel.noise** | 없음 | `Unoise(n_min=-1.5, n_max=1.5)` | rl_training만 노이즈 |
| **velocity_commands** | 있음 | 있음 | 둘 다 활성 |
| **actions** | 있음 | 있음 | 둘 다 활성 |
| **height_scan** | 없음 | 있음 (rough에서 `None`으로 비활성) | rl_training에 정의되어 있으나 rough에서는 비활성 |

### 4.2 Observation Settings

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **enable_corruption** | `False` | `True` | rl_training만 corruption 활성 |
| **concatenate_terms** | `True` | `True` | 동일 |
| **clip** | 없음 (대부분) | `(-100.0, 100.0)` (대부분) | rl_training만 clip 설정 |
| **noise** | 없음 (대부분) | 있음 (대부분) | rl_training만 노이즈 추가 |

---

## 5. Actions 설정

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **action_type** | `JointPositionActionCfg` | `JointPositionActionCfg` | 동일 |
| **scale (HipX)** | `0.125` | `0.125` | 동일 |
| **scale (HipY/Knee)** | `0.25` | `0.25` | 동일 |
| **clip** | 없음 | `(-100.0, 100.0)` | rl_training만 clip 설정 |
| **use_default_offset** | `True` | `True` | 동일 |
| **preserve_order** | `True` | `True` | 동일 |

---

## 6. Commands 설정

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **command_type** | `UniformVelocityCommandCfg` | `UniformThresholdVelocityCommandCfg` | **다른 클래스!** |
| **resampling_time_range** | `(10, 10)` | `(10.0, 10.0)` | 동일 |
| **rel_standing_envs** | `0.02` | `0.02` | 동일 |
| **rel_heading_envs** | `1.0` | `1.0` | 동일 |
| **heading_command** | `True` | `True` | 동일 |
| **heading_control_stiffness** | `0.5` | `0.5` | 동일 |
| **lin_vel_x** | `(-1.5, 1.5)` | `(-1.5, 1.5)` | 동일 (최근 수정됨) |
| **lin_vel_y** | `(-0.8, 0.8)` | `(-0.8, 0.8)` | 동일 (최근 수정됨) |
| **ang_vel_z** | `(-0.8, 0.8)` | `(-0.8, 0.8)` | 동일 (최근 수정됨) |
| **heading** | `(-π, π)` | `(-π, π)` | 동일 (최근 수정됨) |
| **debug_vis** | 없음 | `True` | rl_training만 시각화 |

---

## 7. Rewards 설정

### 7.1 Tracking Rewards

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **tracking_lin_vel** | `weight=3.5, std=sqrt(0.5)` | `weight=3.5, std=sqrt(0.5)` | 동일 |
| **tracking_ang_vel** | `weight=1.5, std=sqrt(0.5)` | `weight=1.5, std=sqrt(0.5)` | 동일 |

### 7.2 Base Stability Rewards

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **lin_vel_z** | `weight=-2.0` | `weight=-2.0` | 동일 |
| **ang_vel_xy** | `weight=-0.05` | `weight=-0.05` | 동일 |
| **orientation** | `weight=-5.0` | `weight=-5.0` (flat_orientation_l2) | 동일 |
| **base_height** | `weight=-10.0, target=0.3536` | `weight=-10.0, target=0.35` | **target 높이 약간 다름** |

### 7.3 Joint Rewards

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **torques** | `weight=-2.5e-5` | `weight=-2.5e-5` (joint_torques_l2) | 동일 |
| **dof_vel** | `weight=0.0` | 없음 | E2E만 정의 (비활성) |
| **dof_acc** | `weight=-1e-8` | `weight=-1e-8` (joint_acc_l2) | 동일 |
| **joint_power** | `weight=-2e-5` | `weight=-2e-5` | 동일 |
| **action_rate** | `weight=-0.02` | `weight=-0.02` (action_rate_l2) | 동일 |
| **dof_pos_limits** | `weight=-5.0` | `weight=-5.0` (joint_pos_limits) | 동일 |
| **joint_deviation_l1** | `weight=-0.5` (HipX만) | `weight=-0.5` (HipX만) | 동일 |

### 7.4 Feet Rewards

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **feet_air_time** | `weight=5.0, threshold=0.5` | `weight=5.0, threshold=0.5` | 동일 |
| **feet_air_time_variance** | `weight=-8.0` | `weight=-8.0` | 동일 |
| **feet_gait** | `weight=0.5, std=sqrt(0.5)` | `weight=0.5, std=sqrt(0.5)` | 동일 |
| **foot_contact_forces** | `weight=-0.1, threshold=200.0` | `weight=-0.1, threshold=100.0` | **threshold 2배 차이!** |
| **feet_slide** | `weight=-0.05` | `weight=-0.05` | 동일 |
| **feet_height** | `weight=-0.2, target=0.05` | `weight=-0.2, target=0.05` | 동일 |
| **feet_height_body** | `weight=-2.5, target=-0.3536` | `weight=-2.5, target=-0.35` | **target 높이 약간 다름** |
| **feet_contact_without_cmd** | `weight=0.1` | `weight=0.1` | 동일 |
| **feet_stumble** | `weight=0.0` | 없음 | E2E만 정의 (비활성) |

### 7.5 Contact Rewards

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **undesired_contacts** | `weight=-0.5, threshold=1.0` | `weight=-0.5, threshold=1.0` | 동일 |

### 7.6 Other Rewards

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **stand_still** | `weight=-0.5` | `weight=-0.5` (stand_still) | 동일 |
| **joint_mirror** | `weight=-0.05` | `weight=-0.05` | 동일 |
| **termination** | `weight=0.0` | 없음 (is_terminated) | E2E만 정의 (비활성) |

---

## 8. Terminations 설정

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **time_out** | 있음 | 있음 | 동일 |
| **body_height_oob** | 있음 (`minimum_height=0.15`) | 없음 | **E2E만 활성!** |
| **bad_orientation** | 있음 (`limit_angle=π/6`) | 없음 | **E2E만 활성!** |
| **terrain_out_of_bounds** | 없음 | 있음 (`distance_buffer=3.0`) | **rl_training만 활성!** |
| **illegal_contact** | 없음 | `None` (비활성) | 둘 다 비활성 |
| **bad_orientation_2** | 없음 | `None` (비활성) | 둘 다 비활성 |

**영향**:
- E2E는 더 엄격한 termination 조건 (높이, 자세)
- rl_training은 terrain 경계만 체크

---

## 9. Events 설정

### 9.1 Startup Events

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **randomize_rigid_body_material** | 없음 | 있음 | rl_training만 활성 |
| **randomize_rigid_body_mass** | 없음 | 있음 | rl_training만 활성 |
| **randomize_rigid_body_mass_base** | 없음 | `None` (비활성) | 둘 다 비활성 |
| **randomize_rigid_body_inertia** | 없음 | 있음 | rl_training만 활성 |
| **randomize_com_positions** | 없음 | 있음 | rl_training만 활성 |

### 9.2 Reset Events

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **reset_body** | 있음 (`reset_root_state_uniform`) | 있음 (`randomize_reset_base`) | 둘 다 있으나 설정 다름 |
| **reset_body.pose_range** | `x=(0,0), y=(0,0), z=(0,0), roll=(0,0), pitch=(0,0), yaw=(-π,π)` | `x=(-1,1), y=(-1,1), z=(0,0), roll=(-0.3,0.3), pitch=(-0.3,0.3), yaw=(-3.14,3.14)` | **rl_training이 더 넓은 범위!** |
| **reset_body.velocity_range** | 모두 `(0,0)` | `x=(-0.2,0.2), y=(-0.2,0.2), z=(-0.2,0.2), roll=(-0.05,0.05), pitch=(-0.05,0.05), yaw=(0,0)` | **rl_training이 초기 속도 추가!** |
| **reset_joints** | 있음 (`reset_joints_by_offset`) | 있음 (`randomize_reset_joints`) | 함수 다름 |
| **reset_joints.position_range** | `(0.0, 0.0)` | `(1.0, 1.0)` | **완전히 다름!** |
| **reset_joints.velocity_range** | `(0.0, 0.0)` | `(0.0, 0.0)` | 동일 |
| **randomize_apply_external_force_torque** | 없음 | `None` (비활성) | 둘 다 비활성 |
| **randomize_actuator_gains** | 없음 | 있음 | rl_training만 활성 |
| **randomize_push_robot** | 없음 | `None` (비활성) | 둘 다 비활성 |

---

## 10. Curriculum 설정

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **terrain_levels** | 없음 | 있음 (`terrain_levels_vel`) | rl_training만 활성 |
| **command_levels** | 없음 | `None` (비활성) | 둘 다 비활성 |

---

## 11. 기타 설정

| 항목 | E2E_locomotion_v2 | rl_training (rough) | 차이 |
|------|-------------------|---------------------|------|
| **viewer.eye** | `(8.0, 0.0, 5.0)` | 설정 없음 | E2E만 명시적 설정 |
| **disable_zero_weight_rewards()** | 있음 | 있음 | 둘 다 동일한 메서드 |

---

## 주요 차이점 요약

### 1. **Simulation 설정**
- E2E: `dt=1/120`, `decimation=2` → 더 정밀한 물리 시뮬레이션
- rl_training: `dt=0.005`, `decimation=4` → 더 빠른 학습

### 2. **Terrain**
- E2E: 평지 (GroundPlaneCfg)
- rl_training: Rough terrain (TerrainImporterCfg with generator)

### 3. **Robot Model**
- E2E: Quad_v2_serial_v4 (커스텀 모델)
- rl_training: Deeprobotics Lite3 (표준 모델)

### 4. **Actuators**
- E2E: `ImplicitActuatorCfg` (단순 PD)
- rl_training: `DelayedPDActuatorCfg` (하드웨어 지연 모델링)

### 5. **Observations**
- E2E: 노이즈 없음, clip 없음
- rl_training: 노이즈 있음, clip 있음, corruption 활성

### 6. **Terminations**
- E2E: `body_height_oob`, `bad_orientation` 활성 (엄격)
- rl_training: `terrain_out_of_bounds`만 활성 (관대)

### 7. **Events**
- E2E: 최소한의 randomization
- rl_training: 광범위한 randomization (mass, inertia, material, actuator gains 등)

### 8. **Rewards**
- 대부분 동일하나, 일부 target 값 차이:
  - `base_height`: E2E `0.3536` vs rl_training `0.35`
  - `feet_height_body`: E2E `-0.3536` vs rl_training `-0.35`
  - `contact_forces`: E2E `threshold=200.0` vs rl_training `threshold=100.0`

---

## 결론

E2E_locomotion_v2와 rl_training (rough)는 **근본적으로 다른 환경**입니다:

1. **로봇 모델**: 완전히 다른 로봇 (Quad_v2 vs Lite3)
2. **Terrain**: 평지 vs rough terrain
3. **Simulation**: 다른 dt/decimation 설정
4. **Actuators**: 단순 PD vs 지연 PD
5. **Terminations**: E2E가 더 엄격
6. **Events**: rl_training이 더 많은 randomization

이러한 차이점들이 학습 성능과 보행 패턴에 큰 영향을 미칩니다.
