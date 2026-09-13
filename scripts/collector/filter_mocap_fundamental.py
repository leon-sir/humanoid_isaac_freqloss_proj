"""Keep DC + one shared gait frequency in joint trajectories; preserve root motion.

Fit raw degree-valued joints, NOT the mean-removed/Hann-windowed spectrum.
The supplied spectrum is used for diagnostics; periodic JSON supplies exact f0.
Requires NumPy only. Output is a signal-processing reference, not validated gait.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def project(q, time, frequency):
    basis = np.column_stack([np.ones(len(time)), np.cos(2*np.pi*frequency*time),
                             np.sin(2*np.pi*frequency*time)])
    coefficients = np.linalg.lstsq(basis, q, rcond=None)[0]
    return coefficients


def equalize_bilateral_energy(coefficients, names):
    """Average cycle AC energy (A^2/2), not amplitude; retain DC and phase.

    For a numerically zero side its phase is undefined: use the opposite
    phase of its partner, explicitly recorded in the report.
    """
    result = coefficients.copy()
    pairs = []
    for left, name in enumerate(names):
        if not name.startswith("left_"):
            continue
        partner = "right_" + name[5:]
        if partner not in names:
            raise ValueError(f"Missing bilateral partner: {partner}")
        right = names.index(partner)
        amplitudes = np.hypot(coefficients[1, [left, right]], coefficients[2, [left, right]])
        target = float(np.sqrt(np.mean(amplitudes**2)))
        fallbacks = []
        for index, amplitude, other in ((left, amplitudes[0], right), (right, amplitudes[1], left)):
            if amplitude > 1e-12:
                result[1:, index] *= target / amplitude
            elif target > 1e-12:
                vector = coefficients[1:, other]
                result[1:, index] = -target * vector / np.linalg.norm(vector)
                fallbacks.append(names[index])
            else:
                result[1:, index] = 0.0
        pairs.append(dict(left=name, right=partner, input_amplitudes_deg=amplitudes.tolist(),
                          target_amplitude_deg=target, target_cycle_ac_energy_deg2=target**2/2,
                          undefined_phase_fallback=fallbacks))
    return result, pairs


def equalize_chain_phase_delays(coefficients, names):
    """Give the left/right limb chains the same anchor-relative phase delays.

    Hip/shoulder pitch phases remain unchanged.  Traverse hip pitch -> knee ->
    ankle pitch and shoulder pitch -> elbow.  At every edge, circularly average
    the left and right phase delay, then rotate both distal coefficients to
    that delay.  DC and fundamental amplitudes are preserved exactly.
    """
    result = coefficients.copy()
    edge_specs = (
        ("leg", "hip_pitch_joint", "knee_joint"),
        ("leg", "knee_joint", "ankle_pitch_joint"),
        ("arm", "shoulder_pitch_joint", "elbow_joint"),
    )
    reports = []
    eps = 1e-12
    for chain_name, anchor_suffix, distal_suffix in edge_specs:
        anchor_ids = [names.index(f"{side}_{anchor_suffix}") for side in ("left", "right")]
        distal_ids = [names.index(f"{side}_{distal_suffix}") for side in ("left", "right")]
        anchor = np.array(
            [complex(result[1, index], -result[2, index]) for index in anchor_ids]
        )
        distal = np.array(
            [complex(result[1, index], -result[2, index]) for index in distal_ids]
        )
        amplitudes = np.abs(distal)
        if np.any(np.abs(anchor) <= eps) or np.any(amplitudes <= eps):
            raise ValueError(
                f"Cannot symmetrize undefined phase in {chain_name} chain: "
                f"{anchor_suffix} -> {distal_suffix}"
            )
        delays = distal / np.abs(distal) * np.conj(anchor / np.abs(anchor))
        mean_delay = delays.sum()
        if abs(mean_delay) <= eps:
            raise ValueError(
                f"Left/right {chain_name} phase delays are circularly opposite for {distal_suffix}"
            )
        mean_delay /= abs(mean_delay)
        for side_index, distal_id in enumerate(distal_ids):
            updated = amplitudes[side_index] * anchor[side_index] / abs(anchor[side_index]) * mean_delay
            result[1, distal_id] = updated.real
            result[2, distal_id] = -updated.imag
        reports.append(
            {
                "chain": chain_name,
                "anchor_joint": anchor_suffix,
                "distal_joint": distal_suffix,
                "input_phase_delays_deg": np.angle(delays, deg=True).tolist(),
                "target_phase_delay_deg": float(np.angle(mean_delay, deg=True)),
            }
        )
    return result, reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--spectrum", type=Path, required=True, help="runner_joint_power_scale.csv from this motion")
    p.add_argument("--frequency", type=float, help="Shared gait frequency [Hz]; default from input periodic JSON")
    p.add_argument("--fps", type=float, default=120.)
    p.add_argument("--output", type=Path)
    p.add_argument("--overwrite", action="store_true", help="Replace the output CSV and JSON if present.")
    p.add_argument("--symmetry", action=argparse.BooleanOptionalAction, default=False,
                   help="Equalize bilateral energy and left/right limb-chain phase delays; preserve DC.")
    a = p.parse_args()
    if not np.isfinite(a.fps) or a.fps <= 0:
        p.error("fps must be finite and positive")
    with a.csv.open(newline="") as f:
        rows = list(csv.reader(f))
    header, raw = rows[0], rows[1:]
    data = np.array(raw, dtype=float)
    if not np.isfinite(data).all() or not np.allclose(np.diff(data[:, 0]), 1):
        p.error("Require finite values and consecutive Frame indices")
    sidecar = a.csv.with_suffix(".json")
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    frequency = a.frequency
    length = None
    if frequency is None:
        if not metadata.get("stitching"):
            p.error("No periodic stitching metadata: supply --frequency explicitly")
        if not np.isclose(metadata["fps"], a.fps):
            p.error("--fps differs from periodic metadata")
        length = metadata["selected"]["end"] - metadata["selected"]["start"]
        frequency = a.fps/length
    if not np.isfinite(frequency) or not 0 < frequency < a.fps/2:
        p.error("frequency must lie strictly between zero and Nyquist")
    ids = [i for i,n in enumerate(header) if n.endswith("_dof")]
    names = [header[i][:-4] for i in ids]
    if len(names) != 21:
        p.error("Expected all 21 joint columns")
    with a.spectrum.open(newline="") as f:
        sr = list(csv.reader(f))
    sh, sd = sr[0], np.array(sr[1:], dtype=float)
    if len(sd)<2 or not np.isfinite(sd).all() or np.any(np.diff(sd[:,0])<=0):
        p.error("Invalid spectrum frequency grid")
    if sd[-1,0] < frequency:
        p.error("Spectrum does not cover requested gait frequency")
    for n in names:
        if n+"_power_normalized" not in sh:
            p.error(f"Missing spectrum column for {n}")
    time = np.arange(len(data))/a.fps
    # Use one complete original period when metadata exists: avoids weighting
    # the extra partial cycle in a 30-s clip and preserves exact cycle DC.
    fit_count = length or len(data)
    if fit_count > len(data) or fit_count < 3:
        p.error("Insufficient data for the fitting interval")
    q = data[:,ids]
    coef = project(q[:fit_count], time[:fit_count], frequency)
    input_amplitudes = np.hypot(coef[1], coef[2])
    symmetry_pairs = []
    symmetry_chain_phase_delays = []
    if a.symmetry:
        coef, symmetry_pairs = equalize_bilateral_energy(coef, names)
        coef, symmetry_chain_phase_delays = equalize_chain_phase_delays(coef, names)
    basis = np.column_stack([np.ones(len(time)), np.cos(2*np.pi*frequency*time),
                             np.sin(2*np.pi*frequency*time)])
    filtered = basis @ coef
    output = a.output or a.csv.with_name(a.csv.stem + ("_filtered_symmetric.csv" if a.symmetry else "_filtered.csv"))
    report_path = output.with_suffix(".json")
    protected = {a.csv.resolve(), sidecar.resolve(), a.spectrum.resolve()}
    if output.resolve() in protected or report_path.resolve() in protected:
        p.error("Output must not overwrite input CSV, metadata or spectrum")
    if not a.overwrite and (output.exists() or report_path.exists()):
        p.error("Output exists; choose another --output")
    report = dict(source=str(a.csv.resolve()), spectrum=str(a.spectrum.resolve()), fps=a.fps,
        frequency_hz=frequency, fit_samples=fit_count, output_samples=len(data),
        symmetry=a.symmetry, symmetry_pairs=symmetry_pairs,
        symmetry_chain_phase_delays=symmetry_chain_phase_delays,
        formula="q_j(t) = c_j + a_j*cos(2*pi*f0*t) + b_j*sin(2*pi*f0*t)",
        notes=["Only joint columns changed; root translation/rotation preserved exactly.",
               "Spectrum is diagnostic only, not inverted: its Hann window and removed mean cannot restore absolute posture.",
               "DC coefficients retained; optional symmetry equalizes bilateral AC energy and chain phase delays.",
               "Symmetry averages cycle AC energy in deg^2; equal normalized energy assumes matching bilateral scales.",
               "Root may retain harmonics; contact, joint limits and dynamic feasibility are NOT guaranteed."], joints={})
    for j,n in enumerate(names):
        residual = q[:fit_count,j]-filtered[:fit_count,j]
        variance = np.var(q[:fit_count,j])
        power = sd[:, sh.index(n+"_power_normalized")]
        peak = float(sd[1+np.argmax(power[1:]),0])
        report["joints"][n] = dict(mean_deg=float(coef[0,j]), cos_deg=float(coef[1,j]),
            input_amplitude_deg=float(input_amplitudes[j]),
            sin_deg=float(coef[2,j]), amplitude_deg=float(np.hypot(coef[1,j],coef[2,j])),
            residual_rms_deg=float(np.sqrt(np.mean(residual**2))),
            explained_variance=float(1-np.var(residual)/variance) if variance>1e-12 else None,
            input_spectrum_peak_hz=peak)
    with output.open("w" if a.overwrite else "x",newline="") as f:
        w=csv.writer(f, lineterminator="\n");w.writerow(header)
        for i,row in enumerate(raw):
            row=row.copy()
            for j,column in enumerate(ids):row[column]=format(filtered[i,j],'.12g')
            w.writerow(row)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(f"Shared f0={frequency:.9f} Hz; fit={fit_count} samples; output={len(data)} samples")
    if a.symmetry:
        print(f"Symmetry enabled for {len(symmetry_pairs)} bilateral energy pairs (waist unchanged).")
        print(f"Chain phase symmetry enabled for {len(symmetry_chain_phase_delays)} anchor/distal relations.")
        for pair in symmetry_pairs:
            if pair['undefined_phase_fallback']:
                print(f"WARNING: undefined zero-amplitude phase; using partner's opposite phase: {pair['undefined_phase_fallback']}")
    for name, values in report['joints'].items():
        print(f"{name:30s} amplitude={values['amplitude_deg']:8.3f} deg  removed RMS={values['residual_rms_deg']:8.3f} deg")
    print(f"CSV: {output}\nReport: {report_path}\nWARNING: fundamental-only reference is not a validated locomotion trajectory.")


if __name__ == '__main__':
    main()
