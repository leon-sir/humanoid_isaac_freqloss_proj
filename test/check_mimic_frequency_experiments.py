"""CPU regression and NarrowBand replay check; run from the project root."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / 'source/humanoid_isaac_freq/humanoid_isaac_freq/tasks/manager_based/locomotion/velocity/config/mimic/mdp'


def main():
    scope = dict(torch=torch, math=math)
    nodes = [n for n in ast.parse((MDP / 'frequency_fit.py').read_text()).body
             if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frequency_fit.py', 'exec'), scope)
    freq, basis = scope['frequency_basis'](100, .02, (.4, 2.5), .025, 'cpu')
    t = torch.arange(100) * .02
    for actual in (.9375, 1., 1.1, 1.6):
        signals = torch.cos(2 * math.pi * actual * t[None, :, None]
                            + torch.arange(16)[:, None, None] * (2 * math.pi / 16))
        f, c, v = scope['estimate_frequency'](signals, freq, basis)
        assert (f - actual).abs().max() < .026
        assert c.min() > .99
    f, c, v = scope['estimate_frequency'](torch.zeros(2, 100, 4), freq, basis)
    assert not c.any() and not v.any()
    print('PASS: real production frequency estimator, known signals and silence')

    # Execute the actual band reward with lightweight environment/analyzer stubs.
    class Term:
        def __init__(self, cfg, env):
            self.stats = env.stats
            self.ids = torch.arange(4)
    node = next(n for n in ast.parse((MDP / 'freq_rewards.py').read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'ReferenceBandEnergyReward')
    scope['_Term'] = Term
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'freq_rewards.py', 'exec'), scope)
    run = '2026-09-13_20-05-11_freq_mimic_v1'
    cfg = yaml.load((ROOT / f'logs/rsl_rl/flat_21dof_freq_mimic/{run}/params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    joints = ['left_hip_pitch_joint', 'right_hip_pitch_joint',
              'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint']
    coefficients = torch.tensor([[float(v) for v in cfg['reference_motion']['joints'][j][1:]] for j in joints])
    amplitude = coefficients.square().sum(-1).sqrt() / torch.tensor([.55, .55, .5, .5])
    paths = {
        'trained': ROOT / f'scripts/output/spectrum_analysis/{run}',
        'natural': ROOT / 'scripts/output/spectrum_mocap_analysis/2026-09-10_10-49-03_453816_Neutral_walk_forward_002__A057_periodic',
        'filtered': ROOT / 'scripts/output/spectrum_mocap_analysis/2026-09-10_10-49-55_151188_Neutral_walk_forward_002__A057_periodic_filtered',
    }
    params = dict(joint_names=joints, energy_band_hz=(.75, 1.), reference_energy_ratio=.25,
                  command_name='base_velocity', energy_floor=1.e-4)
    for label, path in paths.items():
        data = np.genfromtxt(path / 'runner_joint_angles_scale.csv', delimiter=',', names=True)
        assert np.allclose(np.diff(data['time_s']), .02)
        x = np.stack([data[j + '_z_normalized'] for j in joints], -1)
        history = torch.tensor(np.stack([x[i:i+200] for i in range(0, len(x)-199, 5)]), dtype=torch.float32)
        w = torch.hann_window(200, periodic=False)
        p = torch.fft.rfft((history-history.mean(1, keepdim=True))*w[None, :, None], dim=1, norm='ortho').abs().square()
        p[:, 1:-1] *= 2
        a = NS(step_dt=.02, window_size=200, window=w, window_energy=w.square().sum(),
               freq=torch.fft.rfftfreq(200, .02), cached_power=p.transpose(1, 2),
               cached_ready=torch.ones(len(history), dtype=torch.bool))
        command = NS(is_standing_env=torch.zeros(len(history), dtype=torch.bool))
        env = NS(device='cpu', stats=NS(analyzer=a, f0=float(cfg['reference_motion']['frequency_hz']),
                 target_fundamental_amplitude=amplitude, update=lambda env: None),
                 command_manager=NS(get_term=lambda name: command))
        term = scope['ReferenceBandEnergyReward'](NS(params=params), env)
        score = term(env, **params)
        per_joint = ((a.cached_power[:, :, term.band].sum(-1)/a.window_energy)/term.target).clamp(0, 1)
        print(label, 'windows', len(history), 'per_joint_mean', per_joint.mean(0).tolist(),
              'score mean/min/max', score.mean().item(), score.min().item(), score.max().item())
        command.is_standing_env[:] = True
        assert not term(env, **params).any()
        command.is_standing_env[:] = False
        a.cached_ready[:] = False
        assert not term(env, **params).any()
    print('PASS: production band reward standing/warmup masks')

    # Test the production frequency term (not only its estimator), including cache/gates.
    scope['_Term'] = Term
    node = next(n for n in ast.parse((MDP / 'frequency_fit.py').read_text()).body
                if isinstance(n, ast.ClassDef))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'frequency_fit.py', 'exec'), scope)
    h = torch.cos(2*math.pi*1.6*t)[None, :, None].expand(3, 100, 4).clone()
    a = NS(window_size=100, step_dt=.02, history=h, write_index=torch.zeros(3, dtype=torch.long),
           _time_indices=torch.arange(100), last_fft_step=0, cached_ready=torch.tensor([True, True, False]))
    command = NS(is_standing_env=torch.tensor([False, True, False]))
    env = NS(device='cpu', stats=NS(analyzer=a, target_fundamental_amplitude=amplitude, update=lambda e: None),
             command_manager=NS(get_term=lambda n: command))
    params = dict(joint_names=joints, command_name='base_velocity', search_band_hz=(.4,2.5),
                  search_step_hz=.025, allowed_band_hz=(.8,1.2), sigma_hz=.5,
                  min_reference_energy_ratio=.05, energy_floor=1.e-4)
    term = scope['CoreFrequencyRangePenalty'](NS(params=params), env)
    out = term(env, **params)
    assert out[0] > 0 and out[1] == 0 and out[2] == 0
    assert torch.equal(out, term(env, **params))
    a.history.zero_(); a.last_fft_step += 1
    assert not term(env, **params).any()
    print('PASS: frequency reward cache, standing, readiness and low-amplitude handling')

    # Check actual experiment overrides without importing the simulator runtime.
    class Parent:
        def __post_init__(self):
            self.terminations = NS()
            self.rewards = NS(spectral_kinematic_chain_phase_match='chain', ankle_joint_pos_limits='ankle',
                spectral_fundamental_energy_match_core=NS(params={'energy_band_hz': (.5,1.5)}))
            self.joint_frequency_analyzer = NS(window_duration_s=2.)
    scope.update(configclass=lambda c:c, RewTerm=NS, DoneTerm=NS, SwapTimeTooShort=object,
                 YMBOY21DOFFrequencyMimicEnvCfg_v1=Parent, CORE_FREQUENCY_JOINTS=joints)
    path = MDP.parent / 'ymboy_21dof_envcfg_freq_mimic_experiments.py'
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    for name, window in [('FrequencyFit', 2.), ('NarrowBand', 4.)]:
        cfg = scope[f'YMBOY21DOFMimic{name}EnvCfg'](); cfg.__post_init__()
        assert cfg.rewards.spectral_kinematic_chain_phase_match is None
        assert cfg.rewards.ankle_joint_pos_limits is None
        assert cfg.joint_frequency_analyzer.window_duration_s == window
    import gymnasium as gym
    registration = MDP.parent / '__init__.py'
    exec(compile(registration.read_text(), str(registration), 'exec'), {'__name__': 'mimic_test_registration'})
    for name in ('FrequencyFit', 'NarrowBand'):
        assert gym.spec(f'FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1-{name}')
    print('PASS: experiment config overrides and Gym task registration')


if __name__ == '__main__':
    main()
