"""Small GPU smoke test: managers, all 21 channels, spectral rewards, reset."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    default="FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic",
    choices=(
        "FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic",
        "FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V0",
        "FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1",
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import gymnasium as gym
    import torch
    import humanoid_isaac_freq.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    task = args.task
    is_v0 = task.endswith("-V0")
    cfg = parse_env_cfg(task, device=args.device, num_envs=2)
    # Keep episodes alive solely to test filling a full FFT window.
    for name in vars(cfg.terminations):
        setattr(cfg.terminations, name, None)
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    analyzer = env.frequency_analyzer
    assert env.action_manager.total_action_dim == analyzer.num_joints == 21
    assert cfg.joint_frequency_analyzer.window_duration_s == 2.0
    assert analyzer.window_size == 100
    has_chain = not is_v0 and cfg.mimic_training_stage == 2
    for _ in range(analyzer.window_size + cfg.joint_frequency_analyzer.fft_update_interval):
        result = env.step(torch.zeros((2, 21), device=env.device))
        assert torch.isfinite(result[1]).all()
    assert analyzer.cached_ready.all()
    assert torch.isfinite(analyzer.cached_power).all()
    assert analyzer.cached_power.shape == (2, 21, analyzer.num_freq_bins)
    command = env.command_manager.get_command('base_velocity')
    assert torch.count_nonzero(command[:, 1]) == 0
    # Analytic term checks independent of the random physical rollout.
    stats = env.mimic_statistics
    real_update = stats.update
    stats.update = lambda env: None
    stats.energy = stats.target_energy.expand(2,-1).clone()
    stats.out_energy = stats.target_out.expand(2,-1).clone()
    stats.non_harmonic_energy = stats.target_non_harmonic_energy.expand(2, -1).clone()
    analyzer.cached_mean[:] = stats.target_mean
    stats.complex_coeff = stats.reference_fundamental_coeff.expand(2, -1).clone()
    for name in (
        'spectral_fundamental_energy_match_core',
        'spectral_fundamental_energy_match_other',
    ):
        term = env.reward_manager.get_term_cfg(name)
        torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2, device=env.device))
    if has_chain:
        term = env.reward_manager.get_term_cfg('spectral_kinematic_chain_phase_match')
        torch.testing.assert_close(term.func(env, **term.params), torch.ones(2, device=env.device))

    # Bilateral phase has a canonical half-cycle target, independent of small
    # asymmetries in the selected motion reference.
    stats.complex_coeff[:] = 1
    bilateral_name = 'spectral_bilateral_phase_match' if is_v0 else 'spectral_bilateral_phase_match_core'
    term = env.reward_manager.get_term_cfg(bilateral_name)
    for left_id, right_id in zip(term.func.left, term.func.right):
        stats.complex_coeff[:, right_id] = -stats.complex_coeff[:, left_id]
    torch.testing.assert_close(term.func(env, **term.params), torch.ones(2, device=env.device))
    for name in ('spectral_reference_dc_match_core', 'spectral_reference_dc_match_other'):
        term = env.reward_manager.get_term_cfg(name)
        torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2, device=env.device))
    if is_v0:
        term = env.reward_manager.get_term_cfg('spectral_non_harmonic_energy')
        torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2, device=env.device))
    else:
        analyzer.history.zero_()
        for name in ('spectral_non_harmonic_energy_core', 'spectral_non_harmonic_energy_other'):
            term = env.reward_manager.get_term_cfg(name)
            term.func.last_step = None
            torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2, device=env.device))

    # Cross-limb targets remain explicitly in phase and are independent of the
    # selected reference's bilateral phase differences.
    stats.complex_coeff[:] = 1
    term = env.reward_manager.get_term_cfg('spectral_cross_limb_phase_match')
    torch.testing.assert_close(term.func(env, **term.params), torch.ones(2, device=env.device))
    stats.complex_coeff[:] = 0
    for name in (
        bilateral_name,
        'spectral_cross_limb_phase_match',
    ):
        term = env.reward_manager.get_term_cfg(name)
        torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2, device=env.device))
    if has_chain:
        term = env.reward_manager.get_term_cfg('spectral_kinematic_chain_phase_match')
        torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2, device=env.device))
    stats.update = real_update
    env._reset_idx(torch.tensor([0], device=env.device))
    assert analyzer.valid_count[0] == 0 and not analyzer.cached_ready[0]
    assert analyzer.cached_ready[1]
    print(f"PASS [{task}]: 21 actions, spectral target/phase/silent checks, FFT, isolated reset", flush=True)
    env.close()
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
