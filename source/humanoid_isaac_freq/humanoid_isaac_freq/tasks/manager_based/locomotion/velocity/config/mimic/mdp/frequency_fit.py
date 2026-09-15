"""Short-window sinusoidal frequency estimation; no FFT-bin peak quantization."""

import math
import torch

from .freq_rewards import _Term


def frequency_basis(window_size, dt, search_band_hz, search_step_hz, device):
    low, high = search_band_hz
    if not 0 < low < high < 0.5 / dt or search_step_hz <= 0:
        raise ValueError('Invalid frequency search band or spacing')
    count = int(math.floor((high - low) / search_step_hz + 1.e-6)) + 1
    frequencies = low + torch.arange(count, device=device) * search_step_hz
    if len(frequencies) < 2 or window_size < 4:
        raise ValueError('Frequency fit needs at least two candidates and four samples')
    theta = 2 * math.pi * frequencies[:, None] * torch.arange(window_size, device=device)[None] * dt
    design = torch.stack((torch.ones_like(theta), theta.cos(), theta.sin()), dim=-1)
    return frequencies, torch.linalg.qr(design, mode='reduced').Q


def estimate_frequency(history, frequencies, basis, eps=1.e-8):
    """history: [env,time,joint]; returns frequency, explained fraction, variance.

    Zero variance explicitly has zero confidence. Grid spacing is not estimation
    accuracy; the model assumes one dominant sinusoid plus an intercept.
    """
    centered = history - history.mean(dim=1, keepdim=True)
    total = centered.square().sum(dim=1)
    projection = torch.einsum('ftk,ntj->njfk', basis, centered)
    residual = (total[:, :, None] - projection.square().sum(-1)).clamp_min(0)
    best = residual.argmin(dim=-1)
    minimum = residual.gather(-1, best[:, :, None]).squeeze(-1)
    confidence = (1 - minimum / total.clamp_min(eps)).clamp(0, 1)
    confidence = torch.where(total > eps, confidence, torch.zeros_like(confidence))
    return frequencies[best], confidence, total / history.shape[1]


class CoreFrequencyRangePenalty(_Term):
    """Penalize per-core frequency outside a tolerated band, with soft reliability gates.

    Negative reward weight required. Standing and incomplete windows are masked.
    Amplitude gate is V/(V+Vmin), with Vmin from reference energy; periodicity
    gate is the best sinusoidal explained fraction. Existing energy encouragement
    remains necessary to discourage avoiding this penalty by not oscillating.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        a = self.stats.analyzer
        self.frequencies, self.basis = frequency_basis(
            a.window_size, a.step_dt, p['search_band_hz'], p['search_step_hz'], env.device)
        low, high = p['allowed_band_hz']
        if not p['search_band_hz'][0] < low < high < p['search_band_hz'][1]:
            raise ValueError('Allowed frequency interval must lie inside search interval')
        if p['sigma_hz'] <= 0 or p['min_reference_energy_ratio'] <= 0 or p['energy_floor'] <= 0:
            raise ValueError('Frequency scale and reliability energy thresholds must be positive')
        self.minimum_energy = (0.5 * self.stats.target_fundamental_amplitude[self.ids].square()
                               * p['min_reference_energy_ratio']).clamp_min(p['energy_floor'])
        self.command = env.command_manager.get_term(p['command_name'])
        self.last_step = None

    def __call__(self, env, joint_names, command_name, search_band_hz, search_step_hz,
                 allowed_band_hz, sigma_hz, min_reference_energy_ratio, energy_floor):
        self.stats.update(env)
        a = self.stats.analyzer
        if self.last_step != a.last_fft_step:
            indices = (a.write_index[:, None] + a._time_indices[None]) % a.window_size
            history = a.history[:, :, self.ids]
            ordered = torch.gather(history, 1, indices[:, :, None].expand_as(history))
            self.estimated_frequency_hz, self.periodicity, self.variance = estimate_frequency(
                ordered, self.frequencies, self.basis)
            self.last_step = a.last_fft_step
        low, high = allowed_band_hz
        self.frequency_error_hz = ((low - self.estimated_frequency_hz).clamp_min(0)
                                   + (self.estimated_frequency_hz - high).clamp_min(0))
        reliability = self.variance / (self.variance + self.minimum_energy) * self.periodicity
        loss = (reliability * (self.frequency_error_hz / sigma_hz).square()).mean(-1)
        return torch.where(a.cached_ready & ~self.command.is_standing_env, loss, torch.zeros_like(loss))
