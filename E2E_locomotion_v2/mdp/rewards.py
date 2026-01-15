# Custom reward functions for E2E locomotion
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg as RewTerm
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

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
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward


# 5) collision / feet_stumble / stand_still
def rew_collision(
    env,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_forces_w
    force_norm = torch.linalg.norm(forces_w, dim=-1)
    collision_mask = (force_norm > contact_force_threshold).float()
    collision_count = torch.sum(collision_mask, dim=1)
    return collision_count


def rew_feet_stumble(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def rew_stand_still(env, command_name: str) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


# 6) feet air time variance (보행 패턴: 발들 간 균형)
def rew_feet_air_time_variance(
    env,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other.
    
    This reward encourages feet to move in a synchronized manner by penalizing
    differences in air time and contact time between feet.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor. 
                   Use body_names to filter specific feet (e.g., [".*foot"]).
                   If not specified, all detected bodies are used.
                   
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    
    # Compute variance across feet (dim=1 is the foot dimension)
    # Clip to prevent extreme values from dominating
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    
    return reward


# 7) gait reward (보행 패턴: 특정 보행 강제)
class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: "ManagerBasedRLEnv"):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        return reward

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

