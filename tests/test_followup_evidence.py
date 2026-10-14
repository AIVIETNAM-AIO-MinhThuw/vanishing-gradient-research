import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from followup_evidence import (  # noqa: E402
    FollowupMLP,
    parameter_displacement_rows,
    parameter_gradient_rows,
    relative_norm,
    grouped_means,
)


class FollowupEvidenceTests(unittest.TestCase):
    def test_relative_norm_and_near_zero_rule(self) -> None:
        value, excluded = relative_norm(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0]), 1e-12)
        self.assertFalse(excluded)
        self.assertAlmostEqual(value, 2.5)
        value, excluded = relative_norm(torch.ones(2), torch.zeros(2), 1e-12)
        self.assertTrue(excluded)
        self.assertIsNone(value)

    def test_gradient_and_one_step_estimate_use_same_checkpoint_gradient(self) -> None:
        torch.manual_seed(42)
        model = FollowupMLP(3, "tanh", 1.0)
        images = torch.randn(4, 1, 28, 28)
        targets = torch.tensor([0, 1, 2, 3])
        rows, loss = parameter_gradient_rows(model, images, targets, 0.01, 1e-12)
        self.assertEqual(len(rows), 4)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        for row in rows:
            self.assertAlmostEqual(row["relative_one_step_update_estimate"], 0.01 * row["relative_parameter_gradient_norm"])

    def test_displacement_is_endpoint_difference_not_gradient_estimate(self) -> None:
        model = FollowupMLP(3, "sigmoid", 1.0)
        previous = {name: value.detach().clone() for name, value in model.state_dict().items()}
        images = torch.randn(4, 1, 28, 28)
        targets = torch.tensor([0, 1, 2, 3])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = F.cross_entropy(model(images)[0], targets)
        loss.backward()
        optimizer.step()
        current = model.state_dict()
        rows = parameter_displacement_rows(previous, current, 3, 1e-12)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["net_parameter_displacement"] >= 0 for row in rows))

    def test_grouped_summary_is_an_unweighted_seed_mean(self) -> None:
        rows = [{"depth": 12, "step": 0, "seed": 42, "value": 1.0}, {"depth": 12, "step": 0, "seed": 123, "value": 3.0}]
        summary = grouped_means(rows, ("depth", "step"), ("value",))
        self.assertEqual(summary, [{"depth": 12, "step": 0, "seed_count": 2, "mean_value": 2.0}])


if __name__ == "__main__":
    unittest.main()
