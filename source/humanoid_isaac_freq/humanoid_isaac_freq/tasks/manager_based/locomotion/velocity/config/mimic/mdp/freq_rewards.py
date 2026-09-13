"""Window-normalized frequency imitation, sharing the environment's analyzer.

Reference is a compact harmonic fit of the archived collector's absolute angles.
Target FFT statistics are recomputed for actual training dt/window/scales/defaults.
No runtime mocap file IO, no reference time/phase observation.
"""
import math
from copy import deepcopy

import torch
from isaaclab.managers import ManagerTermBase
from humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.mdp.freq_rewards import JointDcPosturePenalty

class MimicStatistics:
    def __init__(self, env):
        self.analyzer = a = env.frequency_analyzer
        reference = env.cfg.reference_motion
        self.f0 = reference['frequency_hz']
        half_width = env.cfg.mimic_band_half_width_hz
        self.band = (a.freq > 0) & ((a.freq-self.f0).abs() <= half_width)
        if not self.band.any():
            raise ValueError('Mimic fundamental band contains no FFT bins')
        self.out = (a.freq > 0) & ~self.band
        t = torch.arange(a.window_size, device=env.device)*a.step_dt
        theta = 2*math.pi*self.f0*t
        design = torch.stack([torch.ones_like(t), theta.cos(), theta.sin()], dim=1)
        self.projector = torch.linalg.pinv(design)
        coeff = torch.tensor([reference['joints'][n] for n in a.joint_names], device=env.device)
        if coeff.shape != (a.num_joints, 3) or not torch.isfinite(coeff).all():
            raise ValueError('Reference joints require finite [DC, cos, sin] coefficients')
        default = a.asset.data.default_joint_pos[0, a.joint_ids]
        coeff[:, 0] -= default
        coeff /= a.scales[:, None]
        # Exact-f0 harmonic amplitude in the same normalized coordinates as the
        # analyzer history.  Unlike target_energy, this has no FFT-bin or Hann
        # leakage dependence and is therefore the direct mimic target for v2.
        self.target_fundamental_amplitude = coeff[:, 1:].square().sum(-1).sqrt()
        self.reference_fundamental_coeff = torch.complex(coeff[:, 1], -coeff[:, 2])

        harmonic_count = int(reference['harmonic_count'])
        if harmonic_count != env.cfg.mimic_harmonic_count:
            raise ValueError(
                f"Reference contains {harmonic_count} harmonics, "
                f"but mimic_harmonic_count={env.cfg.mimic_harmonic_count}"
            )
        if harmonic_count * self.f0 >= 0.5 / a.step_dt:
            raise ValueError(
                f"The highest allowed harmonic ({harmonic_count} * {self.f0:g} Hz) "
                f"must remain below the analyzer Nyquist frequency ({0.5 / a.step_dt:g} Hz)"
            )
        harmonic_coeff = torch.tensor(
            [reference['natural_harmonics'][name] for name in a.joint_names],
            device=env.device,
        )
        if harmonic_coeff.shape != (a.num_joints, harmonic_count, 2):
            raise ValueError('Natural harmonic coefficients have an invalid shape')
        harmonic_coeff /= a.scales[:, None, None]
        self.target_harmonic_energy = 0.5 * harmonic_coeff.square().sum(dim=(1, 2))
        self.target_non_harmonic_energy = torch.tensor(
            [reference['natural_non_harmonic_energy_rad2'][name] for name in a.joint_names],
            device=env.device,
        ) / a.scales.square()

        harmonic_design_columns = [torch.ones_like(t)]
        for harmonic in range(1, harmonic_count + 1):
            harmonic_theta = harmonic * theta
            harmonic_design_columns.extend([harmonic_theta.cos(), harmonic_theta.sin()])
        self.harmonic_design = torch.stack(harmonic_design_columns, dim=1)
        self.harmonic_projector = torch.linalg.pinv(self.harmonic_design)
        self.harmonic_gram = self.harmonic_design.T @ self.harmonic_design
        # Average 64 reference window start phases: same short Hann window as RL.
        phase = torch.arange(64, device=env.device)*2*math.pi/64
        angles = theta[None, :, None]+phase[:, None, None]
        ref = coeff[None,None,:,0]+angles.cos()*coeff[None,None,:,1]+angles.sin()*coeff[None,None,:,2]
        means = ref.mean(1)
        spectrum = torch.fft.rfft((ref-means[:,None])*a.window[None,:,None],dim=1,norm='ortho')
        power = spectrum.abs().square().transpose(1,2)
        power[:,:,1:-1 if a.window_size%2==0 else None] *= 2
        power /= a.window_energy
        self.target_energy = power[:,:,self.band].sum(-1).mean(0)
        self.target_out = power[:,:,self.out].sum(-1).mean(0)
        self.out_tolerance = power[:,:,self.out].sum(-1).std(0,unbiased=False)*2
        self.target_mean = coeff[:,0]
        self.dc_window_tolerance = (means-self.target_mean).abs().amax(0)
        self.last_step = -1

    def update(self, env):
        a = self.analyzer
        a.update_once(env)
        if self.last_step == a.last_fft_step:
            return
        self.last_step = a.last_fft_step
        self.energy = a.cached_power[:,:,self.band].sum(-1)/a.window_energy
        self.out_energy = a.cached_power[:,:,self.out].sum(-1)/a.window_energy
        indices = (a.write_index[:,None]+a._time_indices[None]) % a.window_size
        ordered = torch.gather(a.history,1,indices[:,:,None].expand(-1,-1,a.num_joints))
        # Fit at exact f0, including intercept: no nearest-bin phase bias.
        fit = torch.einsum('kt,ntj->nkj',self.projector,ordered)
        self.complex_coeff = torch.complex(fit[:,1],-fit[:,2])
        harmonic_fit = torch.einsum('kt,ntj->nkj', self.harmonic_projector, ordered)
        total_energy = ordered.square().mean(dim=1)
        explained_energy = torch.einsum(
            'nkj,kl,nlj->nj', harmonic_fit, self.harmonic_gram, harmonic_fit
        ) / a.window_size
        self.non_harmonic_energy = (total_energy - explained_energy).clamp_min(0.0)


class _Term(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not hasattr(env,'mimic_statistics'):
            env.mimic_statistics = MimicStatistics(env)
        self.stats = env.mimic_statistics
        self.ids = self.stats.analyzer.joint_indices(cfg.params['joint_names']) if 'joint_names' in cfg.params else None

    def gate(self, value):
        return torch.where(self.stats.analyzer.cached_ready,value,torch.zeros_like(value))


class FundamentalEnergyMatch(_Term):
    """Legacy bounded score for matching energy in the FFT band around f0.

    This returns one at a match and approaches zero for a large mismatch.  It is
    intentionally retained so existing experiments remain reproducible.
    """

    def __call__(self, env, joint_names, amplitude_floor=0.1, sigma=0.5):
        self.stats.update(env)
        target = self.stats.target_energy[self.ids].clamp_min(0).sqrt()
        measured = self.stats.energy[:,self.ids].clamp_min(0).sqrt()
        loss = ((measured-target)/(target+amplitude_floor)).square().mean(-1)
        return self.gate(torch.exp(-loss/sigma**2))


class FundamentalEnergyMatch_v2(_Term):
    """Penalize L2 error from each joint's reference fundamental amplitude.

    A DC+cos+sin least-squares fit is evaluated at the exact reference gait
    frequency, so this term does not accept arbitrary energy elsewhere inside a
    broad FFT band.  The complex coefficient magnitude is the sinusoid amplitude
    in normalized joint coordinates.  Dividing by ``target + amplitude_floor``
    makes errors comparable across joints and keeps near-zero reference targets
    well conditioned.

    The returned value is a non-negative loss with optimum zero and no
    exponential saturation.  It must therefore be configured with a negative
    reward weight.
    """

    def __call__(self, env, joint_names, amplitude_floor=0.1):
        self.stats.update(env)
        target = self.stats.target_fundamental_amplitude[self.ids]
        measured = self.stats.complex_coeff[:, self.ids].abs()
        relative_error = (measured - target) / (target + amplitude_floor)
        return self.gate(relative_error.square().mean(dim=-1))


class ReferenceDcMatch(JointDcPosturePenalty):
    """Squared DC penalty using the selected reference, rather than default pose.

    Convert absolute reference radians to analyzer coordinates once at startup.
    There is no exponential score or implicit reference-window tolerance.
    """

    def __init__(self, cfg, env):
        a = env.frequency_analyzer
        reference = env.cfg.reference_motion['joints']
        defaults = a.asset.data.default_joint_pos[0, a.joint_ids].detach().cpu().tolist()
        scales = a.scales.detach().cpu().tolist()
        target = {name: (float(reference[name][0]) - default) / scale
                  for name, default, scale in zip(a.joint_names, defaults, scales)}
        internal_cfg = deepcopy(cfg)
        internal_cfg.params = {
            'analyzer_cfg': env.cfg.joint_frequency_analyzer,
            'joint_names': cfg.params['joint_names'],
            'moving_dc_limit': cfg.params.get('moving_dc_limit', 0.0),
            'target_dc': target,
        }
        super().__init__(internal_cfg, env)

    def __call__(self, env, joint_names, moving_dc_limit=0.0):
        return super().__call__(
            env, analyzer_cfg=env.cfg.joint_frequency_analyzer,
            command_name='base_velocity', k_omega=0.5,
            stand_command_threshold=0.1, moving_dc_limit=moving_dc_limit,
            joint_names=joint_names,
        )


class _Phase(_Term):
    target_sign = 1

    def __init__(self, cfg, env):
        super().__init__(cfg,env)
        pairs = cfg.params['joint_pairs']
        if not pairs:
            raise ValueError('Phase pairs cannot be empty')
        a = self.stats.analyzer
        self.left = a.joint_indices([p[0] for p in pairs])
        self.right = a.joint_indices([p[1] for p in pairs])
        self.target_cross = torch.full(
            (len(pairs),), complex(float(self.target_sign), 0.0), device=env.device
        )

    def __call__(self, env, joint_pairs, amplitude_floor=0.02):
        self.stats.update(env)
        c = self.stats.complex_coeff
        left,right = c[:,self.left],c[:,self.right]
        cross = left*right.conj()
        normalized_cross = cross / (cross.abs() + 1e-8)
        cosine = torch.real(normalized_cross * self.target_cross.conj()).clamp(-1, 1)
        score = .5*(1+cosine)
        valid = (left.abs()>amplitude_floor)&(right.abs()>amplitude_floor)
        # Silent pairs get zero reward, not a perfect phase score or exclusion.
        return self.gate(torch.where(valid,score,torch.zeros_like(score)).mean(-1))


class BilateralPhaseMatch(_Phase):
    """Reward a fixed 180-degree phase difference for bilateral joint pairs."""

    target_sign = -1


class CrossLimbPhaseMatch(_Phase):
    """Left shoulder/right hip and right shoulder/left hip: zero difference."""
    target_sign = 1


class KinematicChainPhaseMatch(_Phase):
    """Match reference-f0 phase delays from each limb anchor to distal joints."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        reference = self.stats.reference_fundamental_coeff
        reference_cross = reference[self.left] * reference[self.right].conj()
        if torch.any(reference_cross.abs() <= 1e-8):
            raise ValueError("Kinematic-chain reference phase is undefined for a zero-amplitude joint")
        self.target_cross = reference_cross / reference_cross.abs()


class OutOfBandEnergy(_Term):
    """Legacy FFT-bin penalty around only the fundamental band."""

    def __call__(self, env, joint_names, energy_floor=0.02, allowance=0.01):
        self.stats.update(env)
        budget = self.stats.target_out[self.ids]+self.stats.out_tolerance[self.ids]+allowance
        excess = (self.stats.out_energy[:,self.ids]-budget).clamp_min(0)
        loss = (excess/(self.stats.target_energy[self.ids]+energy_floor)).mean(-1)
        return self.gate(loss)


class NonHarmonicEnergyPenalty(_Term):
    """Penalize motion outside integer multiples of the reference fundamental.

    The allowed basis is DC plus k*f0 for k=1..K, where f0 and K come from the
    selected reference motion.  A joint may use any amplitude and phase in those
    harmonic components.  Only least-squares residual energy above the natural
    periodic reference residual and ``allowance`` is penalized.  This preserves
    the reference's contact-related harmonics without accepting a new arbitrary
    gait frequency.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.harmonic_count = cfg.params.get('harmonic_count')
        self.use_reference_residual = cfg.params.get('use_reference_residual', True)
        if self.harmonic_count is None:
            return  # Preserve the original V0 computation, including its normalization.
        count = self.harmonic_count
        available = int(env.cfg.reference_motion['harmonic_count'])
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= available:
            raise ValueError(f'harmonic_count must be an integer in [1, {available}]')
        a = self.stats.analyzer
        if 2 * count + 1 > a.window_size:
            raise ValueError('Not enough samples for the requested harmonic fit')
        t = torch.arange(a.window_size, device=env.device) * a.step_dt
        columns = [torch.ones_like(t)]
        for k in range(1, count + 1):
            theta = 2 * math.pi * k * self.stats.f0 * t
            columns.extend([theta.cos(), theta.sin()])
        self.design = torch.stack(columns, dim=1)
        self.projector = torch.linalg.pinv(self.design)
        coefficients = torch.tensor(
            [env.cfg.reference_motion['natural_harmonics'][a.joint_names[i]]
             for i in self.ids.tolist()], device=env.device,
        )[:, :count] / a.scales[self.ids, None, None]
        self.allowed_reference_energy = 0.5 * coefficients.square().sum(dim=(1, 2))
        self.last_step = None

    def __call__(self, env, joint_names, energy_floor=0.02, allowance=0.01,
                 harmonic_count=None, use_reference_residual=True):
        """Restrict allowed harmonics to 1..K at the reference f0.

        Omitting K preserves V0 exactly. With explicit K the denominator uses
        only the first K reference harmonics. Excluded harmonics are never added
        back to the residual budget. Set use_reference_residual=False for a
        strict, explicit allowance in normalized position-squared units (V1 core).
        """
        self.stats.update(env)
        if self.harmonic_count is not None:
            a = self.stats.analyzer
            if self.last_step != a.last_fft_step:
                indices = (a.write_index[:, None] + a._time_indices[None]) % a.window_size
                history = a.history[:, :, self.ids]
                ordered = torch.gather(history, 1, indices[:, :, None].expand_as(history))
                fit = torch.einsum('kt,ntj->nkj', self.projector, ordered)
                reconstructed = torch.einsum('tk,nkj->ntj', self.design, fit)
                self.residual_energy = (ordered - reconstructed).square().mean(dim=1)
                self.last_step = a.last_fft_step
            budget = allowance
            if self.use_reference_residual:
                budget = budget + self.stats.target_non_harmonic_energy[self.ids]
            excess = (self.residual_energy - budget).clamp_min(0)
            return self.gate((excess / (self.allowed_reference_energy + energy_floor)).mean(-1))
        budget = self.stats.target_non_harmonic_energy[self.ids] + allowance
        excess = (self.stats.non_harmonic_energy[:, self.ids] - budget).clamp_min(0)
        denominator = self.stats.target_harmonic_energy[self.ids] + energy_floor
        return self.gate((excess / denominator).mean(dim=-1))
