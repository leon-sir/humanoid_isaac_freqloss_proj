"""CPU-only regression against the existing policy collector's FFT definition."""
import ast
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mocap_utils", ROOT / "scripts/collector/mocap_spectrum_utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class SpectrumTest(unittest.TestCase):
    def test_matches_policy_collector(self):
        tree = ast.parse((ROOT / "scripts/collector/collect_runner_data_analysis_scale.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "calculate_training_spectrum")
        namespace = {"torch": torch, "SPECTRUM_EPS": 1e-8}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "collector_fft", "exec"), namespace)
        for count in (749, 750):
            z = torch.randn(count, 21, dtype=torch.float64)
            actual = utils.calculate_training_spectrum(z, .02)
            expected = namespace["calculate_training_spectrum"](z, .02)
            for a, b in zip(actual[:-1], expected[:-1]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(actual[-1], expected[-1])
            window = torch.hann_window(count, periodic=False, dtype=z.dtype)
            torch.testing.assert_close(actual[2].sum(0), ((z-z.mean(0))*window[:, None]).square().sum(0))


if __name__ == "__main__":
    unittest.main()
