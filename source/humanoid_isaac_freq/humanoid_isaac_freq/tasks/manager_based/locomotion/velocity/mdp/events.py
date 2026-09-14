# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    # from humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.config.humanoid.ymboy_flip.env.manager_based_rl_env import MyManagerBasedRLYMEnv


def reset_root_state_uniform_body_frame(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset pose as usual, but sample linear/angular velocity in the new body frame.

    Ported from Dash skating. Pose translation remains in world coordinates.
    Use the freshly sampled quaternion directly, independent of simulator caches.
    Default root velocity components are interpreted in body coordinates.
    """
    asset = env.scene[asset_cfg.name]
    state = asset.data.default_root_state[env_ids].clone()
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    ranges = torch.tensor([pose_range.get(k, (0., 0.)) for k in keys], device=asset.device)
    pose = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    position = state[:, :3] + env.scene.env_origins[env_ids] + pose[:, :3]
    orientation = math_utils.quat_mul(
        state[:, 3:7], math_utils.quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]))
    ranges = torch.tensor([velocity_range.get(k, (0., 0.)) for k in keys], device=asset.device)
    velocity = state[:, 7:13] + math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocity_w = torch.cat((math_utils.quat_apply(orientation, velocity[:, :3]),
                            math_utils.quat_apply(orientation, velocity[:, 3:])), dim=-1)
    asset.write_root_pose_to_sim(torch.cat((position, orientation), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocity_w, env_ids=env_ids)


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias[env_ids[:, None], body_ids]

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


def randomize_implicit_actuator_strength(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    strength_distribution_params: tuple[float, float],
):
    """Scale implicit-actuator PD gains with one correlated motor-strength factor per joint.

    Isaac Gym's Go1 task multiplies each motor's final PD torque by a fixed strength factor. For an implicit PD
    actuator with no feed-forward effort, multiplying stiffness and damping by the same factor is equivalent before
    the simulator applies the unchanged effort limit. This term intentionally multiplies the current gains so it can
    be composed after independent stiffness/damping randomization.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(device=asset.device)

    for actuator in asset.actuators.values():
        if not isinstance(actuator, ImplicitActuator):
            continue

        if isinstance(asset_cfg.joint_ids, slice):
            actuator_indices: slice | torch.Tensor = slice(None)
        else:
            actuator_joint_ids = torch.as_tensor(actuator.joint_indices, device=asset.device)
            selected_joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=asset.device)
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_ids, selected_joint_ids)).flatten()
            if actuator_indices.numel() == 0:
                continue

        stiffness = actuator.stiffness[env_ids].clone()
        damping = actuator.damping[env_ids].clone()
        selected_stiffness = stiffness[:, actuator_indices]
        strength = math_utils.sample_uniform(
            *strength_distribution_params,
            selected_stiffness.shape,
            device=asset.device,
        )
        stiffness[:, actuator_indices] *= strength
        damping[:, actuator_indices] *= strength

        actuator.stiffness[env_ids] = stiffness
        actuator.damping[env_ids] = damping
        asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
        asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data




def reset_flip_stage_buf(
    env: MyManagerBasedRLYMEnv,
    env_ids: torch.Tensor | None,
):
    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    env.flip_state_buff[env_ids, :, :] = 0.0
    env.flip_state_buff[env_ids, 0, 0] = 1.0
    env.is_half_turn_buf[env_ids] = 0.0
    env.is_one_turn_buf[env_ids] = 0.0


def push_by_setting_velocity_body_frame(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # in body frame
    # note: this event likes a curriculum
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    # vel_w = asset.data.root_vel_w[env_ids]

    vel_w = asset.data.root_vel_w[env_ids].clone()
    ang_vel_w = vel_w[:, 3:]


    lin_vel_b = asset.data.root_lin_vel_b[env_ids].clone()
    # sample random linear velocities
    range_lin_vel_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    # sample random angular velocities
    range_ang_vel_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    lin_vel_ranges = torch.tensor(range_lin_vel_list, device=asset.device)
    ang_vel_ranges = torch.tensor(range_ang_vel_list, device=asset.device)

    ang_vel_w += math_utils.sample_uniform(ang_vel_ranges[:, 0], ang_vel_ranges[:, 1], ang_vel_w.shape, device=asset.device)
    lin_vel_b += math_utils.sample_uniform(lin_vel_ranges[:, 0], lin_vel_ranges[:, 1], lin_vel_b.shape, device=asset.device)

    # lin_vel_b[:,0] = env.command_manager.get_command("base_velocity")[env_ids, 0]

    lin_vel_w = math_utils.quat_apply(asset.data.root_link_quat_w[env_ids], lin_vel_b)
    new_vel_w = torch.cat([lin_vel_w, ang_vel_w], dim=-1)

    # 
    # vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)

    
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(new_vel_w, env_ids=env_ids)


def push_joints_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_range: tuple[float, float],
    operation: Literal["scale", "offset"] = "offset",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Perturb current joint velocities with independently sampled scales or offsets.

    Unlike the joint-reset events, this function starts from the current joint velocities rather than the default
    joint state and writes only joint velocities back to simulation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_vel = asset.data.joint_vel[env_ids].clone()
        joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    else:
        joint_vel = asset.data.joint_vel[env_ids[:, None], asset_cfg.joint_ids].clone()
        joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids[:, None], asset_cfg.joint_ids]

    random_samples = math_utils.sample_uniform(*velocity_range, joint_vel.shape, device=asset.device)
    if operation == "scale":
        joint_vel *= random_samples
    elif operation == "offset":
        joint_vel += random_samples
    else:
        raise ValueError(
            f"Unsupported joint velocity perturbation operation '{operation}'. Expected 'scale' or 'offset'."
        )

    joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    asset.write_joint_velocity_to_sim(joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def push_by_setting_desired_velocity_body_frame(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.5,
):
    # in body frame
    # note: this event likes a curriculum
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    # vel_w = asset.data.root_vel_w[env_ids]

    vel_w = asset.data.root_vel_w[env_ids].clone()
    ang_vel_w = vel_w[:, 3:]


    lin_vel_b = asset.data.root_lin_vel_b[env_ids].clone()
    # sample random linear velocities
    range_lin_vel_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    # sample random angular velocities
    range_ang_vel_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    lin_vel_ranges = torch.tensor(range_lin_vel_list, device=asset.device)
    ang_vel_ranges = torch.tensor(range_ang_vel_list, device=asset.device)

    ang_vel_w += math_utils.sample_uniform(ang_vel_ranges[:, 0], ang_vel_ranges[:, 1], ang_vel_w.shape, device=asset.device)
    # lin_vel_b += math_utils.sample_uniform(lin_vel_ranges[:, 0], lin_vel_ranges[:, 1], lin_vel_b.shape, device=asset.device)
    lin_vel_b[:, 0] = lin_vel_b[:, 0] + (env.command_manager.get_command("base_velocity")[env_ids, 0] - lin_vel_b[:, 0]) * scale

    # lin_vel_b[:,0] = env.command_manager.get_command("base_velocity")[env_ids, 0]

    lin_vel_w = math_utils.quat_apply(asset.data.root_link_quat_w[env_ids], lin_vel_b)
    new_vel_w = torch.cat([lin_vel_w, ang_vel_w], dim=-1)

    # 
    # vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)

    
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(new_vel_w, env_ids=env_ids)


def reset_root_state_uniform_body_frame(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    
    """                     !! only velocity in body frame !!

    Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # set pose into the physics simulation first
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)

    # velocities
    # range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]

    range_lin_vel_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    # sample random angular velocities
    range_ang_vel_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    lin_vel_ranges = torch.tensor(range_lin_vel_list, device=asset.device)
    ang_vel_ranges = torch.tensor(range_ang_vel_list, device=asset.device)

    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # root_state[:, 7:10]: linear velocity in world frame, root_state[:, 10:13]: angular velocity in world frame
    lin_vel_b = root_states[:, 7:10] + math_utils.sample_uniform(lin_vel_ranges[:, 0], lin_vel_ranges[:, 1], (len(env_ids), 3), device=asset.device)
    lin_vel_w = math_utils.quat_apply(asset.data.root_link_quat_w[env_ids], lin_vel_b)

    ang_vel_b = root_states[:, 10:13] + math_utils.sample_uniform(ang_vel_ranges[:, 0], ang_vel_ranges[:, 1], (len(env_ids), 3), device=asset.device)
    ang_vel_w = math_utils.quat_apply(asset.data.root_link_quat_w[env_ids], ang_vel_b)
    
    new_vel_w = torch.cat([lin_vel_w, ang_vel_w], dim=-1)

    # set velocity into the physics simulation
    asset.write_root_velocity_to_sim(new_vel_w, env_ids=env_ids)
