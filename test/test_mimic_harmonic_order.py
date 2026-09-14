"""CPU checks for harmonic selection and legacy V0 output compatibility."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import torch


def main():
    path = Path(__file__).resolve().parents[1] / 'source/humanoid_isaac_freq/humanoid_isaac_freq/tasks/manager_based/locomotion/velocity/config/mimic/mdp/freq_rewards.py'
    class Manager:
        def __init__(self, cfg, env):
            pass
    scope = dict(torch=torch, math=math, ManagerTermBase=Manager)
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name in ('_Term', 'NonHarmonicEnergyPenalty')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    names = ['hip', 'shoulder']
    a = NS(window_size=100, step_dt=.02, scales=torch.ones(2), joint_names=names,
           history=torch.zeros(2, 100, 2), write_index=torch.tensor([0, 17]),
           _time_indices=torch.arange(100), last_fft_step=0,
           cached_ready=torch.tensor([True, False]),
           joint_indices=lambda ns: torch.tensor([names.index(n) for n in ns]))
    stats = NS(analyzer=a, f0=.9375, update=lambda env: None,
               target_non_harmonic_energy=torch.tensor([.003, .006]),
               target_harmonic_energy=torch.tensor([.5, .7]),
               non_harmonic_energy=torch.tensor([[.4, .8], [.2, .3]]))
    reference = dict(harmonic_count=5, natural_harmonics={n: [[1., 0.]] * 5 for n in names})
    env = NS(device='cpu', mimic_statistics=stats, cfg=NS(reference_motion=reference))
    cls = scope['NonHarmonicEnergyPenalty']
    legacy = cls(NS(params=dict(joint_names=names)), env)
    # Evaluate exactly the original expression/order, not an algebraic rewrite.
    expected = ((stats.non_harmonic_energy - (stats.target_non_harmonic_energy + .01)).clamp_min(0)
                / (stats.target_harmonic_energy + .02)).mean(-1)
    expected[1] = 0
    assert torch.equal(legacy(env, names), expected)
    terms = {k: cls(NS(params=dict(joint_names=names, harmonic_count=k,
                                  use_reference_residual=False)), env) for k in (2, 5)}
    t = torch.arange(100) * .02
    for harmonic in (1, 2, 3, 5):
        signal = 2.3 + torch.cos(2 * math.pi * .9375 * harmonic * t)
        for row, offset in enumerate(a.write_index.tolist()):
            a.history[row] = torch.roll(signal[:, None].expand(-1, 2), offset, dims=0)
        a.last_fft_step += 5
        scores = {k: term(env, names, allowance=0., harmonic_count=k,
                          use_reference_residual=False) for k, term in terms.items()}
        assert scores[5][0] < 1e-8 and scores[5][1] == 0
        assert scores[2][1] == 0
        if harmonic <= 2:
            assert scores[2][0] < 1e-8
        else:
            assert scores[2][0] > .1
    torch.testing.assert_close(terms[2].allowed_reference_energy, torch.ones(2))
    for invalid in (0, 6, 2.5, True):
        try:
            cls(NS(params=dict(joint_names=names, harmonic_count=invalid)), env)
        except ValueError:
            continue
        raise AssertionError(f'Accepted invalid order {invalid}')
    print('PASS: V0 exact output, orders 2/5, ring ordering, readiness and invalid orders')
    # Evaluate the actual reward class body for both import-time stage values.
    config_path = path.parent.parent / 'ymboy_21dof_envcfg_freq_mimic_v1.py'
    node = next(n for n in ast.parse(config_path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'YMBOY21DOFFrequencyOnlyRewardsCfg_v1')
    functions = ['ReferenceBandEnergyReward', 'FundamentalEnergyMatch_v2', 'ReferenceDcMatch', 'BilateralPhaseMatch',
                 'CrossLimbPhaseMatch', 'KinematicChainPhaseMatch', 'NonHarmonicEnergyPenalty']
    for stage in (1, 0):
        scope = dict(configclass=lambda cls: cls, RewTerm=NS, firstStage=stage,
                     CORE_ENERGY_BAND_HZ=(.5, 1.5), CORE_REFERENCE_ENERGY_RATIO=.25,
                     mimic_mdp=NS(**{name: name for name in functions}),
                     CORE_FREQUENCY_JOINTS=['core'], OTHER_FREQUENCY_JOINTS=['other'],
                     CORE_BILATERAL_PHASE_PAIRS=[], CROSS_LIMB_PHASE_PAIRS=[],
                     KINEMATIC_CHAIN_PHASE_PAIRS=[])
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(config_path), 'exec'), scope)
        cfg = scope[node.name]()
        assert (cfg.spectral_kinematic_chain_phase_match is None) == bool(stage)
        assert cfg.spectral_fundamental_energy_match_other.weight == -.5
        assert cfg.spectral_reference_dc_match_other.weight == -.5
        assert cfg.spectral_non_harmonic_energy_core.params['harmonic_count'] == 2
        assert cfg.spectral_non_harmonic_energy_other.params['harmonic_count'] == 5
    print('PASS: both V1 stage branches retain other regulation and toggle chain phase')


if __name__ == '__main__':
    main()
