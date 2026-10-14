import json
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vanishing_gradient.pilot import (  # noqa: E402
    PilotMLP,
    gradient_profile_metrics,
    jacobian_spectrum_metrics,
    measure_sample,
    normalized_directional_gain,
    stratified_indices,
    validate_run_config,
)


class PilotMetricTests(unittest.TestCase):
    def test_gradient_gap_normalization_and_slope_are_distinct(self) -> None:
        result = gradient_profile_metrics(torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64))
        self.assertAlmostEqual(result["raw_early_to_last_gap"], 2.0)
        self.assertAlmostEqual(result["normalized_gap_per_interval"], 1.0)
        self.assertAlmostEqual(result["layer_index_slope"], 1.0)

    def test_directional_gain_and_near_zero_exclusion(self) -> None:
        identity = torch.eye(2, dtype=torch.float64)
        included = normalized_directional_gain(identity, torch.tensor([3.0, 4.0], dtype=torch.float64))
        self.assertFalse(included["excluded_near_zero_denominator"])
        self.assertAlmostEqual(included["value"], 1.0)
        excluded = normalized_directional_gain(
            identity, torch.tensor([1e-8, 0.0], dtype=torch.float64), denominator_threshold=1e-6
        )
        self.assertTrue(excluded["excluded_near_zero_denominator"])
        self.assertIsNone(excluded["value"])

    def test_spectrum_separates_scale_rank_and_condition(self) -> None:
        matrix = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        result = jacobian_spectrum_metrics(matrix)
        self.assertEqual(result["numerical_rank"], 2)
        self.assertEqual(result["inherent_input_nullity"], 1)
        self.assertEqual(result["additional_numerical_rank_deficiency"], 0)
        self.assertAlmostEqual(result["condition_number_nonzero_spectrum"], 2.0)
        scaled = jacobian_spectrum_metrics(torch.eye(2, dtype=torch.float64) * 1e-8)
        self.assertEqual(scaled["numerical_rank"], 2)
        self.assertAlmostEqual(scaled["condition_number_nonzero_spectrum"], 1.0)
        self.assertAlmostEqual(scaled["sigma_max"], 1e-8)
        deficient = jacobian_spectrum_metrics(
            torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        )
        self.assertEqual(deficient["numerical_rank"], 1)
        self.assertEqual(deficient["additional_numerical_rank_deficiency"], 1)
        wide = torch.zeros(10, 128, dtype=torch.float64)
        wide[:, :10] = torch.eye(10, dtype=torch.float64)
        wide_result = jacobian_spectrum_metrics(wide)
        self.assertEqual(wide_result["inherent_input_nullity"], 118)
        self.assertEqual(wide_result["additional_numerical_rank_deficiency"], 0)

    def test_per_sample_cross_entropy_matches_batch_mean_scaling(self) -> None:
        torch.manual_seed(1)
        model = PilotMLP(4, 4, 3, 10, 1.0).to(dtype=torch.float64)
        inputs = torch.randn(2, 4, dtype=torch.float64)
        targets = torch.tensor([1, 2])
        batch_logits, batch_activations = model(inputs)
        batch_loss = F.cross_entropy(batch_logits, targets, reduction="mean")
        batch_gradient = torch.autograd.grad(batch_loss, batch_activations[0])[0][0]

        sample_logits, sample_activations = model(inputs[:1])
        sample_loss = F.cross_entropy(sample_logits, targets[:1], reduction="sum")
        sample_gradient = torch.autograd.grad(sample_loss, sample_activations[0])[0][0]
        self.assertTrue(torch.allclose(batch_gradient, sample_gradient / 2.0, rtol=1e-12, atol=1e-12))

    def test_pre_post_tanh_and_weight_gradient_identities(self) -> None:
        torch.manual_seed(2)
        model = PilotMLP(4, 4, 3, 10, 1.0).to(dtype=torch.float64)
        inputs = torch.randn(2, 4, dtype=torch.float64)
        targets = torch.tensor([3, 5])
        logits, preactivations, activations = model.forward_with_preactivations(inputs)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        pre_gradient, post_gradient, weight_gradient = torch.autograd.grad(
            loss, (preactivations[0], activations[0], model.hidden[0].weight)
        )
        self.assertTrue(
            torch.allclose(pre_gradient, post_gradient * (1.0 - activations[0].square()), rtol=1e-12, atol=1e-12)
        )
        expected_weight_gradient = pre_gradient.transpose(0, 1) @ inputs
        self.assertTrue(torch.allclose(weight_gradient, expected_weight_gradient, rtol=1e-12, atol=1e-12))

    def test_chain_rule_and_jacobian_for_ten_logits(self) -> None:
        torch.manual_seed(3)
        model = PilotMLP(4, 4, 3, 10, 1.0).to(dtype=torch.float64)
        image = torch.randn(1, 4, dtype=torch.float64)
        target = torch.tensor([4])
        logits, activations = model(image)
        loss = F.cross_entropy(logits, target, reduction="sum")
        logit_gradient, activation_gradient = torch.autograd.grad(
            loss, (logits, activations[0]), retain_graph=True
        )
        hidden = activations[0][0].detach().clone().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(lambda value: model.from_hidden(0, value), hidden)
        self.assertEqual(list(jacobian.shape), [10, 4])
        self.assertTrue(
            torch.allclose(activation_gradient[0], jacobian.transpose(0, 1) @ logit_gradient[0], rtol=1e-12, atol=1e-12)
        )
        epsilon = 1e-6
        offset = torch.zeros_like(hidden)
        offset[1] = epsilon
        finite_difference = (model.from_hidden(0, hidden + offset)[2] - model.from_hidden(0, hidden - offset)[2]) / (2 * epsilon)
        self.assertTrue(torch.isclose(jacobian[2, 1], finite_difference, rtol=1e-7, atol=1e-9))

    def test_stratified_indices_are_deterministic_and_balanced(self) -> None:
        targets = torch.arange(10).repeat_interleave(10)
        first = stratified_indices(targets, 20, 42)
        second = stratified_indices(targets, 20, 42)
        self.assertEqual(first, second)
        selected_targets = targets[first]
        self.assertTrue(torch.equal(torch.bincount(selected_targets, minlength=10), torch.full((10,), 2)))

    def test_fixed_probe_measurement_is_reproducible(self) -> None:
        torch.manual_seed(9)
        model = PilotMLP(4, 4, 3, 10, 1.0).to(dtype=torch.float64)
        image = torch.randn(4, dtype=torch.float64)
        target = torch.tensor(2)
        first = measure_sample(model, image, target, [], 1e-6)
        second = measure_sample(model, image, target, [], 1e-6)
        self.assertEqual(first, second)
        self.assertGreater(first["logit_loss_gradient_norm"], 0.0)

    def test_frozen_config_requires_approved_runtime_budget(self) -> None:
        config = json.loads((PROJECT_ROOT / "configs" / "pilot_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(config["runtime_budget_seconds"], 2700)
        config["runtime_budget_seconds"] = None
        with self.assertRaisesRegex(ValueError, "runtime_budget_seconds"):
            validate_run_config(config)
        config["runtime_budget_seconds"] = 2700
        validate_run_config(config)

    def test_diagnostic_gain_changes_only_initialization_scale(self) -> None:
        torch.manual_seed(7)
        reference = PilotMLP(4, 4, 12, 10, 1.0)
        torch.manual_seed(7)
        diagnostic = PilotMLP(4, 4, 12, 10, 0.5)
        self.assertTrue(torch.allclose(diagnostic.hidden[0].weight, reference.hidden[0].weight * 0.5))
        self.assertTrue(torch.equal(diagnostic.hidden[0].bias, reference.hidden[0].bias))


if __name__ == "__main__":
    unittest.main()

