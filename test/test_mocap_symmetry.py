"""CPU regression for bilateral fundamental-energy averaging."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("filter_mocap", Path(__file__).resolve().parents[1] / "scripts/collector/filter_mocap_fundamental.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SymmetryTest(unittest.TestCase):
    def test_energy_phase_and_dc(self):
        original = np.array([[10., -3., 5.], [3., 0., 2.], [4., -2., 1.]])
        output, _ = module.equalize_bilateral_energy(original, ['left_knee_joint', 'right_knee_joint', 'waist_yaw_joint'])
        np.testing.assert_array_equal(output[0], original[0])
        np.testing.assert_array_equal(output[:, 2], original[:, 2])
        np.testing.assert_allclose(np.sum(output[1:, :2]**2, axis=0), [14.5, 14.5])
        for j in (0, 1):
            np.testing.assert_allclose(output[1:, j]/np.linalg.norm(output[1:, j]), original[1:, j]/np.linalg.norm(original[1:, j]))
        np.testing.assert_array_equal(original[1:, 0], [3., 4.])

    def test_zero_amplitude(self):
        for right in (0., 2.):
            output, pairs = module.equalize_bilateral_energy(np.array([[1., 2.], [0., right], [0., 0.]]), ['left_knee_joint', 'right_knee_joint'])
            self.assertTrue(np.isfinite(output).all())
            np.testing.assert_allclose(np.sum(output[1:]**2, axis=0), right**2/2)
            if right:
                self.assertEqual(pairs[0]['undefined_phase_fallback'], ['left_knee_joint'])

    def test_chain_phase_delay_symmetry(self):
        names = [
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "elbow")
        ]
        phases_deg = [10, -60, -110, 40, 95, 175, 120, 75, -130, -55]
        amplitudes = np.linspace(1.0, 2.0, len(names))
        coefficients = np.zeros((3, len(names)))
        coefficients[0] = np.arange(len(names))
        coefficients[1] = amplitudes * np.cos(np.deg2rad(phases_deg))
        coefficients[2] = -amplitudes * np.sin(np.deg2rad(phases_deg))

        output, report = module.equalize_chain_phase_delays(coefficients, names)
        np.testing.assert_array_equal(output[0], coefficients[0])
        np.testing.assert_allclose(np.linalg.norm(output[1:], axis=0), amplitudes)
        self.assertEqual(len(report), 3)

        complex_coeff = output[1] - 1j * output[2]
        for item in report:
            anchor = item["anchor_joint"]
            distal = item["distal_joint"]
            delays = []
            for side in ("left", "right"):
                a = complex_coeff[names.index(f"{side}_{anchor}")]
                d = complex_coeff[names.index(f"{side}_{distal}")]
                delays.append(d / abs(d) * np.conj(a / abs(a)))
            np.testing.assert_allclose(delays[0], delays[1], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
