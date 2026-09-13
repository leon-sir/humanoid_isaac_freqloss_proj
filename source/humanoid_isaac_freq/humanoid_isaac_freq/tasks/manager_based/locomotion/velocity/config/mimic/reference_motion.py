"""Build a training reference from a filtered motion CSV and its sidecar report."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_reference_motion(csv_path: str | Path, harmonic_count: int = 5) -> dict:
    """Fit absolute joint angles (degrees in CSV, radians in returned snapshot).

    The filter report supplies the exact fundamental frequency and source FPS;
    coefficients and body-frame speed are derived from the selected CSV itself.
    The sibling unfiltered periodic CSV supplies the allowed natural harmonics.
    """
    if harmonic_count < 1:
        raise ValueError("harmonic_count must be positive")
    path = Path(csv_path).expanduser().resolve()
    report_path = path.with_suffix('.json')
    report = json.loads(report_path.read_text())
    fps, f0 = float(report['fps']), float(report['frequency_hz'])
    if not np.isfinite([fps, f0]).all() or not 0 < f0 < fps / 2:
        raise ValueError(f'Invalid FPS/fundamental frequency in {report_path}')
    if harmonic_count * f0 >= fps / 2:
        raise ValueError(
            f"The highest allowed harmonic ({harmonic_count} * {f0:g} Hz) "
            f"must remain below the reference Nyquist frequency ({fps / 2:g} Hz)"
        )
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=float, encoding='utf-8-sig')
    if data.ndim != 1 or len(data) < 3:
        raise ValueError(f'Reference CSV requires at least three frames: {path}')
    names = data.dtype.names
    joint_columns = [name for name in names if name.endswith('_joint_dof')]
    if not joint_columns:
        raise ValueError(f'No joint columns in {path}')
    frames = data['Frame']
    if not np.allclose(np.diff(frames), 1):
        raise ValueError('Reference frames must be consecutive at the reported FPS')
    t = (frames - frames[0]) / fps
    if t[-1] < 1 / f0:
        raise ValueError('Reference CSV must span at least one fundamental period')
    q = np.deg2rad(np.column_stack([data[name] for name in joint_columns]))
    position = np.column_stack([data['root_translate' + axis] for axis in 'XYZ']) * 0.01
    euler = np.column_stack([data['root_rotate' + axis] for axis in 'XYZ'])
    if not all(np.isfinite(value).all() for value in (t, q, position, euler)):
        raise ValueError(f'Non-finite reference motion values: {path}')
    design = np.column_stack([np.ones_like(t), np.cos(2*np.pi*f0*t), np.sin(2*np.pi*f0*t)])
    coefficients = np.linalg.lstsq(design, q, rcond=None)[0]

    natural_path = path.with_name(f"{path.stem.removesuffix('_filtered')}.csv")
    if not natural_path.is_file():
        natural_path = Path(report["source"]).expanduser()
        if not natural_path.is_absolute():
            natural_path = report_path.parent / natural_path
        natural_path = natural_path.resolve()
    natural_data = np.genfromtxt(
        natural_path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig"
    )
    if natural_data.ndim != 1 or len(natural_data) < 3:
        raise ValueError(f"Natural periodic CSV requires at least three frames: {natural_path}")
    natural_columns = [name for name in natural_data.dtype.names if name.endswith("_joint_dof")]
    if natural_columns != joint_columns:
        raise ValueError("Filtered and natural periodic CSV joint columns differ")
    natural_frames = natural_data["Frame"]
    if not np.allclose(np.diff(natural_frames), 1):
        raise ValueError("Natural reference frames must be consecutive")
    natural_t = (natural_frames - natural_frames[0]) / fps
    natural_q = np.deg2rad(np.column_stack([natural_data[name] for name in natural_columns]))
    harmonic_design_columns = [np.ones_like(natural_t)]
    for harmonic in range(1, harmonic_count + 1):
        harmonic_design_columns.extend(
            [
                np.cos(2 * np.pi * harmonic * f0 * natural_t),
                np.sin(2 * np.pi * harmonic * f0 * natural_t),
            ]
        )
    harmonic_design = np.column_stack(harmonic_design_columns)
    harmonic_coefficients = np.linalg.lstsq(harmonic_design, natural_q, rcond=None)[0]
    natural_residual = natural_q - harmonic_design @ harmonic_coefficients
    world_velocity = np.gradient(position, 1/fps, axis=0)
    body_velocity = Rotation.from_euler('xyz', euler, degrees=True).inv().apply(world_velocity)
    return {
        'source': str(path),
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'report_source': str(report_path),
        'natural_source': str(natural_path),
        'natural_source_sha256': hashlib.sha256(natural_path.read_bytes()).hexdigest(),
        'fps': fps,
        'frequency_hz': f0,
        'reference_forward_speed_mps': float(body_velocity[:, 0].mean()),
        'reference_lateral_speed_mps': float(body_velocity[:, 1].mean()),
        'fit_max_error_rad': float(np.max(np.abs(design @ coefficients - q))),
        'harmonic_count': harmonic_count,
        'natural_non_harmonic_energy_rad2': {
            name.removesuffix('_dof'): float(np.mean(natural_residual[:, i] ** 2))
            for i, name in enumerate(joint_columns)
        },
        'natural_harmonics': {
            name.removesuffix('_dof'): [
                [
                    float(harmonic_coefficients[2 * harmonic - 1, i]),
                    float(harmonic_coefficients[2 * harmonic, i]),
                ]
                for harmonic in range(1, harmonic_count + 1)
            ]
            for i, name in enumerate(joint_columns)
        },
        'joints': {name.removesuffix('_dof'): coefficients[:, i].tolist()
                   for i, name in enumerate(joint_columns)},
    }
