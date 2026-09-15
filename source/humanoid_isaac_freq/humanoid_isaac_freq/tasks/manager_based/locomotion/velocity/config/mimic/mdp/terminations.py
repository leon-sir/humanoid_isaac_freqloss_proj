"""Independent, reset-safe hip-order timing for mimic continuation tasks."""
import torch
from isaaclab.managers import ManagerTermBase


class SwapTimeTooShort(ManagerTermBase):
    """Terminate repeated rapid hip-pitch order changes, not foot contacts.

    A Schmitt trigger ignores small fluctuations around equal angles. The first
    crossing only starts timing. Standing resets state; no state is shared with
    swap_time_too_long, so termination evaluation order cannot change results.
    swap_time_threshold is measured in control steps, just like too-long:
    20 steps at step_dt=0.02 seconds means a minimum 0.4-second swap interval.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        if p.get('swap_time_threshold', 20) <= 0 or p.get('angle_hysteresis_rad', 0.02) <= 0:
            raise ValueError('Swap interval and angle hysteresis must be positive')
        count = p.get('consecutive_short_swaps', 2)
        if not isinstance(count, int) or count < 1:
            raise ValueError('consecutive_short_swaps must be a positive integer')
        self.robot = env.scene.articulations['robot']
        self.left = self.robot.find_joints('left_hip_pitch_joint')[0][0]
        self.right = self.robot.find_joints('right_hip_pitch_joint')[0][0]
        self.command = env.command_manager.get_term(p.get('command_name', 'base_velocity'))
        self.side = torch.zeros(env.num_envs, dtype=torch.int8, device=env.device)
        self.elapsed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.short_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.side[ids] = 0
        self.elapsed[ids] = 0
        self.started[ids] = False
        self.short_count[ids] = 0

    def __call__(self, env, swap_time_threshold=20, angle_hysteresis_rad=0.02,
                 consecutive_short_swaps=2, command_name='base_velocity', command_threshold=0.1):
        moving = (~self.command.is_standing_env
                  & (env.command_manager.get_command(command_name)[:, :3].norm(dim=-1) > command_threshold))
        self.reset((~moving).nonzero(as_tuple=False).flatten())
        delta = self.robot.data.joint_pos[:, self.left] - self.robot.data.joint_pos[:, self.right]
        side = torch.where(delta > angle_hysteresis_rad, 1,
                           torch.where(delta < -angle_hysteresis_rad, -1, self.side)).to(torch.int8)
        changed = moving & (self.side != 0) & (side != self.side)
        self.elapsed += moving.long()
        measured = changed & self.started
        short = measured & (self.elapsed < swap_time_threshold)
        self.short_count = torch.where(measured, torch.where(short, self.short_count + 1, 0), self.short_count)
        result = moving & (self.short_count >= consecutive_short_swaps)
        self.elapsed[changed] = 0
        self.started[changed] = True
        self.side.copy_(torch.where(moving, side, 0))
        return result
