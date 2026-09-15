"""CPU checks for the production short-swap state machine."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import torch

p = Path(__file__).resolve().parents[1] / 'source/humanoid_isaac_freq/humanoid_isaac_freq/tasks/manager_based/locomotion/velocity/config/mimic/mdp/terminations.py'
class Manager:
    def __init__(self, cfg, env):
        pass
scope = dict(torch=torch, ManagerTermBase=Manager)
node = next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.ClassDef))
exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), 'exec'), scope)
robot = NS(data=NS(joint_pos=torch.zeros(3, 2)),
           find_joints=lambda name: ([0 if name.startswith('left') else 1], []))
cmd = NS(is_standing_env=torch.tensor([False, False, True]))
env = NS(num_envs=3, device='cpu', step_dt=.02, scene=NS(articulations={'robot':robot}),
         command_manager=NS(get_term=lambda n:cmd, get_command=lambda n:torch.ones(3,3)))
params = dict(swap_time_threshold=20)
term = scope['SwapTimeTooShort'](NS(params=params), env)
detected = torch.zeros(3,dtype=torch.bool)
for i in range(200):
    robot.data.joint_pos[:,0] = torch.tensor([.3*math.sin(2*math.pi*1.6*i*.02),
                                              .3*math.sin(2*math.pi*.9375*i*.02),
                                              .3*math.sin(2*math.pi*1.6*i*.02)])
    detected |= term(env, **params)
assert detected.tolist() == [True,False,False], detected
term.reset()
for i in range(100):
    robot.data.joint_pos[:,0] = .01 if i%2 else -.01
    assert not term(env, **params).any()
assert not term.started.any()
print('PASS: fast swaps terminate, reference/standing do not; reset and hysteresis work')
