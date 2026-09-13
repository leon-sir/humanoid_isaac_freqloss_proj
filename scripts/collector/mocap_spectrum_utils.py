"""Spectrum definitions shared with the policy collector."""
import csv
from pathlib import Path
import torch
SPECTRUM_EPS = 1e-8

def calculate_training_spectrum(
    normalized_positions: torch.Tensor, sample_dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply the exact training analyzer spectrum definition along time dimension 0."""
    sample_count = normalized_positions.shape[0]
    mean = normalized_positions.mean(dim=0)
    centered = normalized_positions - mean.unsqueeze(0)
    window = torch.hann_window(sample_count, periodic=False, dtype=normalized_positions.dtype)
    spectrum = torch.fft.rfft(centered * window.unsqueeze(1), dim=0, norm="ortho")
    power = spectrum.abs().square()
    if sample_count % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    frequencies = torch.fft.rfftfreq(sample_count, d=sample_dt, dtype=normalized_positions.dtype)
    window_energy = window.square().sum().clamp_min(SPECTRUM_EPS)
    return frequencies, spectrum, power, mean, float(window_energy)

def write_spectrum_csv(
    path: Path,
    frequencies: torch.Tensor,
    spectrum: torch.Tensor,
    power: torch.Tensor,
    joint_names: list[str],
    max_frequency_hz: float,
) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = frequencies <= min(max_frequency_hz, float(frequencies[-1])) + 1.0e-12
    header = ["frequency_hz"]
    for name in joint_names:
        header += [f"{name}_power_normalized", f"{name}_phase_rad"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for frequency, powers, phases in zip(frequencies[keep], power[keep], torch.angle(spectrum[keep]), strict=True):
            row = [frequency.item()]
            for joint_index in range(len(joint_names)):
                row += [powers[joint_index].item(), phases[joint_index].item()]
            writer.writerow(row)
    return path
