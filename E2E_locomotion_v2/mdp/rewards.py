# Custom reward functions for E2E locomotion
import torch
from isaaclab.managers import SceneEntityCfg

# --------------------------------------------------------------------------- #
#  Reward functions : legged_gym-style
# --------------------------------------------------------------------------- #

# 0) episode alive / termination
def rew_termination(env) -> torch.Tensor:
    return env.termination_manager.terminated.float()


def rew_alive(env) -> torch.Tensor:
    return (~env.termination_manager.terminated).float()


# 1) tracking terms
def rew_tracking_lin_vel(
    env,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = asset.data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(cmd - vel), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)


def rew_tracking_ang_vel(
    env,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]
    yaw_vel = asset.data.root_ang_vel_b[:, 2]
    ang_vel_error = torch.square(cmd_yaw - yaw_vel)
    return torch.exp(-ang_vel_error / tracking_sigma)


# 2) base stability
def rew_lin_vel_z(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def rew_ang_vel_xy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def rew_orientation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def rew_base_height(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


# 3) joint / torque / smoothness
def rew_torques(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def rew_dof_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def rew_dof_acc(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def rew_joint_power(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mechanical power usage: |torque * joint velocity| summed across joints."""
    asset = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(power), dim=1)


def rew_action_rate(env) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )


def rew_dof_pos_limits(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# 4) feet air time
def rew_feet_air_time(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.3,
    max_air_time: float = 0.4,
) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]

    first_contact = sensor.compute_first_contact(env.step_dt)
    last_air_time = sensor.data.last_air_time

    adv = torch.clamp(last_air_time - threshold, min=0.0, max=max_air_time)
    rew = torch.sum(adv * first_contact.float(), dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_xy_norm = torch.norm(cmd[:, :2], dim=1)
    rew *= (cmd_xy_norm > 0.1).float()

    rew *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 1.0)

    return rew


# 5) collision / feet_stumble / stand_still
def rew_collision(
    env,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]
    forces_w = sensor.data.net_forces_w
    force_norm = torch.linalg.norm(forces_w, dim=-1)
    collision_mask = (force_norm > contact_force_threshold).float()
    collision_count = torch.sum(collision_mask, dim=1)
    return collision_count


def rew_feet_stumble(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def rew_stand_still(env, command_name: str) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)

