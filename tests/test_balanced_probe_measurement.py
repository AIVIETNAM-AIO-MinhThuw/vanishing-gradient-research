import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from measure_balanced_probes_v2 import (
    balanced_probe_indices,
    checkpoint_specs,
    label_histogram,
    load_checkpoint_model,
)
from vanishing_gradient.pilot import PilotMLP


class BalancedProbeMeasurementTests(unittest.TestCase):
    def test_balanced_selection_is_deterministic_and_disjoint_from_training(self) -> None:
        targets = torch.arange(10).repeat_interleave(20)
        training = [label * 20 + offset for label in range(10) for offset in range(10)]
        validation = [label * 20 + offset for label in range(10) for offset in range(10, 20)]
        gradient = balanced_probe_indices(validation, targets, 4, 20261014)
        repeated = balanced_probe_indices(validation, targets, 4, 20261014)
        jacobian = [gradient[label * 4 + offset] for label in range(10) for offset in range(2)]
        self.assertEqual(gradient, repeated)
        self.assertEqual(label_histogram(gradient, targets), {str(label): 4 for label in range(10)})
        self.assertEqual(label_histogram(jacobian, targets), {str(label): 2 for label in range(10)})
        self.assertTrue(set(jacobian) <= set(gradient))
        self.assertFalse(set(gradient) & set(training))

    def test_source_run_has_exactly_60_checkpoint_specs(self) -> None:
        raw = json.loads(Path("results/experiment_v2/experiment-v2-20261014T2053/raw_results.json").read_text())
        specs = checkpoint_specs(raw)
        self.assertEqual(len(specs), 60)
        self.assertEqual(len(set(specs)), 60)

    def test_checkpoint_loader_only_restores_weights_and_sets_eval(self) -> None:
        source = PilotMLP(784, 128, 3, 10, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(source.state_dict(), path)
            loaded = load_checkpoint_model(path, 3)
        self.assertFalse(loaded.training)
        for expected, observed in zip(source.parameters(), loaded.parameters()):
            self.assertTrue(torch.equal(expected, observed))


if __name__ == "__main__":
    unittest.main()
