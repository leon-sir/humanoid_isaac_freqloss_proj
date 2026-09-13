"""Select a repeatable full-stride CSV segment without altering motion values.

Requires NumPy and SciPy, not Isaac Sim. CSV units remain cm/degrees.
Scores are heuristics, not a physical contact/stability certification.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057.csv"
SCALES = dict(hip_pitch=.55, hip_roll=.10, hip_yaw=.05, knee=.85,
              ankle_pitch=.55, ankle_roll=.25)


def select_cycle(data, header, fps, min_period, max_period, trim, min_amplitude):
    names = [n[:-4] for n in header if n.endswith("_dof")]
    if len(names) != 21 or len(set(names)) != 21:
        raise ValueError("Expected 21 unique joint columns.")
    scales = np.array([SCALES.get(n.split("_", 1)[1][:-6], .25 if n == "waist_yaw_joint" else .5) for n in names])
    q = np.deg2rad(data[:, [header.index(n + "_dof") for n in names]])
    # Smoothing is ONLY used for ranking/derivatives; output uses original rows.
    width = max(5, int(round(.075 * fps)) | 1)
    if len(q) <= width:
        raise ValueError("Clip too short.")
    z = savgol_filter(q / scales, width, 3, axis=0)
    dz = np.gradient(z, 1 / fps, axis=0)
    ids = [i for i, n in enumerate(names) if any(k in n for k in ("hip_pitch", "knee", "ankle_pitch"))]
    if len(ids) != 6:
        raise ValueError("Missing bilateral hip_pitch/knee/ankle_pitch joints.")
    xyz = data[:, [header.index("root_translate" + a) for a in "XYZ"]] * .01
    velocity = np.gradient(savgol_filter(xyz, width, 3, axis=0), 1 / fps, axis=0)
    rot = Rotation.from_euler("xyz", data[:, [header.index("root_rotate" + a) for a in "XYZ"]], degrees=True)
    lo, hi = int(np.ceil(min_period * fps)), int(np.floor(max_period * fps))
    margin = int(np.ceil(trim * fps))
    candidates = []
    for length in range(max(2, lo), hi + 1):
        period = length / fps
        for start in range(margin, len(z) - 2 * length - margin):
            end = start + length
            a, b = z[start:end, ids], z[end:end + length, ids]
            amplitude = float(np.sqrt(np.mean(np.var(a, axis=0))))
            if amplitude < min_amplitude:
                continue
            variance = (np.var(a, axis=0) + np.var(b, axis=0)) / 2
            repeat = float(np.mean(np.mean((a-b)**2, axis=0) / (variance + .01)))
            # Equal weight per normalized joint; all 21 matter at the seam.
            pose = float(np.mean((z[end]-z[start])**2))
            speed = float(np.mean((period * (dz[end]-dz[start]))**2))
            root_angle = float((rot[start].inv() * rot[end]).magnitude())
            root_height = float(xyz[end, 2] - xyz[start, 2])
            root_velocity = float(np.linalg.norm(velocity[end]-velocity[start]))
            root = (root_angle/.2)**2 + (root_height/.05)**2 + (root_velocity/.5)**2
            score = repeat + pose + .05 * speed + .1 * root
            candidates.append(dict(start=start, end=end, period_s=period, score=score,
                repeat_error=repeat, seam_pose_rms_scale=pose**.5,
                seam_velocity_rms_scale_per_s=speed**.5/period, amplitude_rms_scale=amplitude,
                root_orientation_error_rad=root_angle, root_height_error_m=root_height,
                root_velocity_error_mps=root_velocity))
    if not candidates:
        raise ValueError("No valid candidates: shorten trim/period bounds or inspect motion amplitude.")
    candidates.sort(key=lambda r: r["score"])
    best = candidates[0]
    s, e = best["start"], best["end"]
    best["root_displacement_m"] = (xyz[e]-xyz[s]).tolist()
    best["raw_joint_seam_error_deg"] = dict(zip(names, np.rad2deg(q[e]-q[s]).tolist()))
    return best, candidates, dict(zip(names, scales.tolist()))


def stitch_cycle(data, header, start, end, fps, duration):
    """C1-close a stride, then repeat it with constant world-XY displacement.

    Smooth cubic correction spans the cycle, not just one seam sample. Root
    orientation is corrected in a local rotation-vector chart (walking only).
    """
    length = end - start
    t = np.arange(length + 1) / fps
    period = length / fps
    cycle = data[start:end + 1].copy()
    cols = list(range(1, data.shape[1]))
    values = cycle[:, cols].copy()
    rotation_cols = [header.index("root_rotate" + axis) for axis in "XYZ"]
    rid = [cols.index(c) for c in rotation_cols]
    rotations = Rotation.from_euler("xyz", cycle[:, rotation_cols], degrees=True)
    local = (rotations[0].inv() * rotations).as_rotvec()
    if np.linalg.norm(local, axis=1).max() > np.pi * .8:
        raise ValueError("Large root rotation: straight-walk stitching is not suitable for this clip.")
    values[:, rid] = local
    drift = np.zeros(values.shape[1])
    for axis in "XY":
        j = cols.index(header.index("root_translate" + axis))
        drift[j] = values[-1, j] - values[0, j]
    spline = CubicSpline(t, values, axis=0)
    mismatch = values[-1] - values[0] - drift
    velocity_mismatch = spline(period, 1) - spline(0, 1)
    u = t / period
    correction = ((3*u**2 - 2*u**3)[:, None] * mismatch
                  + (u**3 - u**2)[:, None] * period * velocity_mismatch)
    closed = values - correction
    # Analytic endpoint checks for the corrected continuous curve.
    position_error = closed[-1] - closed[0] - drift
    velocity_error = spline(period, 1) - velocity_mismatch - spline(0, 1)
    count = int(np.ceil(duration * fps)) + 1  # includes the requested final time
    frame = np.arange(count)
    phases, repeats = frame % length, frame // length
    result = np.empty((count, data.shape[1]))
    result[:, 0] = frame
    result[:, cols] = closed[phases] + repeats[:, None] * drift
    result[:, rotation_cols] = (rotations[0] * Rotation.from_rotvec(closed[phases][:, rid])).as_euler("xyz", degrees=True)
    joint_ids = [cols.index(i) for i, name in enumerate(header) if name.endswith("_dof")]
    info = dict(duration_s=(count-1)/fps, output_frames=count,
        cycles=(count-1)/length, method="full-cycle cubic C1 correction; local rotation vectors for root orientation",
        max_joint_correction_deg=float(np.abs(correction[:, joint_ids]).max()),
        max_root_rotation_correction_rad=float(np.linalg.norm(correction[:, rid], axis=1).max()),
        endpoint_position_residual=float(np.abs(position_error).max()),
        endpoint_velocity_residual=float(np.abs(velocity_error).max()),
        note="Fixed-heading straight-walk loop; world XY displacement accumulates; Z/orientation repeat. Not contact-constrained or dynamically validated.")
    return result, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--fps", type=float, default=120.)
    parser.add_argument("--min_period", type=float, default=.7, help="Full stride minimum [s].")
    parser.add_argument("--max_period", type=float, default=1.6, help="Full stride maximum [s].")
    parser.add_argument("--trim", type=float, default=.5, help="Exclude clip edges [s]; not collector warmup.")
    parser.add_argument("--min_amplitude", type=float, default=.15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace the output CSV and JSON if present.")
    parser.add_argument("--duration", type=float, help="Close and repeat selected cycle for this many seconds (e.g. 30).")
    args = parser.parse_args()
    if not (0 < args.min_period <= args.max_period and args.fps > 0 and args.trim >= 0 and args.min_amplitude >= 0):
        parser.error("Invalid frequency, period, trim or amplitude.")
    source = args.csv.resolve()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive.")
    output = args.output or source.with_name(source.stem + ("_periodic.csv" if args.duration else "_stable_cycle.csv"))
    report_path = output.with_suffix(".json")
    if output.resolve() == source or report_path.resolve() == source.with_suffix(".json"):
        parser.error("Output must not overwrite the source CSV or its metadata")
    if not args.overwrite and (output.exists() or report_path.exists()):
        parser.error(f"Output exists; choose a different --output: {output}")
    with source.open(newline="") as file:
        rows = list(csv.reader(file))
    header, raw = rows[0], rows[1:]
    data = np.asarray(raw, dtype=float)
    if not np.isfinite(data).all() or not np.allclose(np.diff(data[:, 0]), 1):
        raise ValueError("Require finite data and consecutive Frame indices.")
    best, ranked, scales = select_cycle(data, header, args.fps, args.min_period, args.max_period,
                                       args.trim, args.min_amplitude)
    start, end = best["start"], best["end"]
    stitched, stitch_report = (stitch_cycle(data, header, start, end, args.fps, args.duration)
                              if args.duration else (None, None))
    # Half-open interval: exclude the matching next-cycle boundary to avoid
    # duplicating one frame per repeat. Boundary is retained in JSON metadata.
    with output.open("w" if args.overwrite else "x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for index, row in enumerate(stitched if stitched is not None else raw[start:end]):
            writer.writerow([index, *row[1:]])
    report = dict(source=str(source), fps=args.fps, selected=best,
        source_frame_start=float(data[start, 0]), source_frame_end_exclusive=float(data[end, 0]),
        start_time_s=start/args.fps, end_time_s=end/args.fps,
        output_frames=len(stitched) if stitched is not None else end-start,
        stitching=stitch_report, endpoint_row=raw[end], joint_scales_rad=scales,
        parameters=vars(args) | {"csv": str(source), "output": str(output)},
        score_formula="repeat_error + seam_pose_rms_scale^2 + 0.05*(T*seam_velocity_rms_scale_per_s)^2 + 0.1*root_error",
        root_error_formula="(orientation_rad/0.2)^2 + (height_m/0.05)^2 + (velocity_mps/0.5)^2",
        notes=["Ranking uses two consecutive cycles; export contains the first cycle only.",
               "Original cm/degree values preserved; only Frame is renumbered.",
               "No smoothing of output, time warping, mirroring, or root recentering.",
               "Looping requires root displacement accumulation and possible seam blending.",
               "Heuristic joint-state recurrence, not verified foot-contact or dynamic stability."],
        top_candidates=ranked[:10])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Source: {source.name} | {len(data)} frames @ {args.fps:g} Hz")
    print(f"Candidates: {len(ranked)} | lower score is better (no universal pass threshold)")
    print(" rank   start[s]  period[s]    score   repeat error")
    for i, candidate in enumerate(ranked[:5], 1):
        print(f" {i:4d} {candidate['start']/args.fps:10.3f} {candidate['period_s']:10.3f}"
              f" {candidate['score']:8.4f} {candidate['repeat_error']:12.4f}")
    print(f"Selected [{start/args.fps:.3f}, {end/args.fps:.3f}) s; {end-start} frames; "
          f"gait frequency={1/best['period_s']:.3f} Hz")
    print(f"Seam RMS: position={best['seam_pose_rms_scale']:.4f} scales, "
          f"velocity={best['seam_velocity_rms_scale_per_s']:.4f} scales/s")
    print(f"Root displacement [m]: {best['root_displacement_m']}")
    errors = best['raw_joint_seam_error_deg']
    print("Largest raw seam angle errors [deg]:", sorted(errors.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    print("WARNING: selected raw cycle is NOT guaranteed seamless or physically stable.")
    print(f"CSV: {output}\nReport: {report_path}")
    if stitch_report:
        # The notes above describe selection; make exported-data semantics explicit.
        report["notes"] = ["Selected raw cycle is recorded by selected/endpoint_row.",
                           "Export is cubic-corrected and repeated, not unmodified mocap.",
                           stitch_report["note"]]
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("Stitching:", json.dumps(stitch_report, indent=2))


if __name__ == "__main__":
    main()
