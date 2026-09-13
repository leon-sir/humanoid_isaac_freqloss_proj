"""CPU regression checks for DC terms without launching Isaac Sim.

Extract the production classes to avoid simulator imports; stub only the manager
and analyzer interfaces. Run with a Python environment that has torch installed.
"""
import ast
import math
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
VELOCITY = ROOT / 'source/humanoid_isaac_freq/humanoid_isaac_freq/tasks/manager_based/locomotion/velocity'


class Manager:
    def __init__(self, cfg, env):
        self.cfg = cfg


def extract(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    torch.manual_seed(7)
    names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint']
    analyzer = NS(joint_names=names, cached_mean=torch.randn(100, 3),
                  cached_ready=torch.arange(100) % 3 != 0,
                  joint_indices=lambda selected: [names.index(n) for n in selected],
                  update_once=lambda env: None)
    moving = torch.arange(100) % 2 == 0
    namespace = dict(torch=torch, math=math, re=re, Sequence=Sequence, deepcopy=deepcopy,
                     ManagerTermBase=Manager, RewardTermCfg=NS, JointFrequencyAnalyzerCfg=NS,
                     _get_shared_analyzer=lambda env, cfg: analyzer,
                     _command_is_moving=lambda *args: moving)
    term_cls = extract(VELOCITY / 'mdp/freq_rewards.py', 'JointDcPosturePenalty', namespace)
    env = NS(device='cpu')
    for limit in (0.0, 0.2, 1.5):
        params = dict(analyzer_cfg=None, command_name='base_velocity', k_omega=0.5,
                      stand_command_threshold=0.1, moving_dc_limit=limit)
        term = term_cls(NS(params=params), env)
        mean = analyzer.cached_mean
        old = torch.where(moving, torch.relu(mean.abs()-limit).square().mean(-1), mean.square().mean(-1))
        old = torch.where(analyzer.cached_ready, old, torch.zeros_like(old))
        assert torch.equal(term(env, **params), old), 'Default behavior changed'
        explicit = dict(params, target_dc=0.0)
        assert torch.equal(term_cls(NS(params=explicit), env)(env, **explicit), old)
    target = {name: float(i + 1) for i, name in enumerate(names)}
    params.update(target_dc=target, moving_dc_limit=0.0)
    term = term_cls(NS(params=params), env)
    error = analyzer.cached_mean - torch.tensor([1., 2., 3.])
    expected = torch.where(analyzer.cached_ready, error.square().mean(-1), 0.)
    assert torch.equal(term(env, **params), expected)
    mimic_cls = extract(VELOCITY / 'config/mimic/mdp/freq_rewards.py', 'ReferenceDcMatch', namespace)
    analyzer.joint_ids = [0, 1, 2]
    analyzer.scales = torch.tensor([0.5, 0.25, 1.0])
    analyzer.asset = NS(data=NS(default_joint_pos=torch.tensor([[0.1, -0.2, 0.3]])))
    env.frequency_analyzer = analyzer
    env.cfg = NS(joint_frequency_analyzer=None,
                 reference_motion={'joints': {name: [float(i), 0., 0.] for i, name in enumerate(names)}})
    params = dict(joint_names=names, moving_dc_limit=0.0)
    term = mimic_cls(NS(params=params), env)
    target = (torch.tensor([0., 1., 2.])-analyzer.asset.data.default_joint_pos[0])/analyzer.scales
    assert torch.allclose(term.target_dc, target)
    expected = torch.where(analyzer.cached_ready, (analyzer.cached_mean-target).square().mean(-1), 0.)
    assert torch.allclose(term(env, **params), expected)
    print('PASS: implicit/explicit zero targets exactly preserve old outputs; arbitrary and reference targets match.')


if __name__ == '__main__':
    main()
