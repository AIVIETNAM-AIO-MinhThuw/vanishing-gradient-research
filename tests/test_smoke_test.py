import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vanishing_gradient.smoke_test import run_smoke_test  # noqa: E402


class SmokeTestCase(unittest.TestCase):
    def test_cpu_float64_instrumentation(self) -> None:
        config = json.loads((PROJECT_ROOT / "configs" / "smoke_test.json").read_text(encoding="utf-8"))
        result = run_smoke_test(config)

        self.assertTrue(result["all_passed"], result)
        self.assertEqual(result["measurements"]["activation_gradient"]["shape"], [4, 16])
        self.assertEqual(result["measurements"]["weight_gradient"]["shape"], [16, 16])
        self.assertEqual(result["measurements"]["jacobian"]["shape"], [3, 16])
        self.assertEqual(len(result["measurements"]["jacobian"]["singular_values"]), 3)
        self.assertFalse(result["scientific_evidence"])


if __name__ == "__main__":
    unittest.main()
