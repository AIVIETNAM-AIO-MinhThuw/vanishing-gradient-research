import json
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vanishing_gradient.experiment_v2 import (
    aggregate_probes,
    deterministic_batch_positions,
    saturation_metrics,
    stratified_split,
    validate_config,
)


class ExperimentV2Tests(unittest.TestCase):
    def test_stratified_split_is_disjoint_balanced_and_reproducible(self) -> None:
        targets = torch.arange(10).repeat_interleave(30)
        first = stratified_split(targets, 100, 50, 42)
        second = stratified_split(targets, 100, 50, 42)
        self.assertEqual(first, second)
        train, validation = first
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(torch.bincount(targets[train], minlength=10).tolist(), [10] * 10)
        self.assertEqual(torch.bincount(targets[validation], minlength=10).tolist(), [5] * 10)

    def test_batch_order_is_fixed_and_complete(self) -> None:
        first = deterministic_batch_positions(10, 4, 7, 123)
        self.assertEqual(first, deterministic_batch_positions(10, 4, 7, 123))
        self.assertNotEqual(first, deterministic_batch_positions(10, 4, 7, 124))
        self.assertEqual(len(first), 7)
        self.assertTrue(all(len(batch) == 4 for batch in first))

    def test_saturation_metrics_match_definition(self) -> None:
        values = torch.tensor([[-3.0, -2.0, 0.0, 2.0, 3.0]])
        result = saturation_metrics(values, 2.0, [-2.0, 0.0, 2.0])
        expected = (1.0 - torch.tanh(values).square()).abs().mean().item()
        self.assertAlmostEqual(result["mean_absolute_tanh_derivative"], expected)
        self.assertAlmostEqual(result["fraction_abs_preactivation_gt_threshold"], 0.4)
        self.assertEqual(sum(result["histogram_counts"]), 5)

    def test_gap_aggregation_keeps_mean_log_ratio_distinct(self) -> None:
        def profile(first: float, last: float) -> dict:
            gap = math.log10(last / first)
            return {
                "gradient_rms": [first, last], "raw_early_to_last_gap": gap,
                "normalized_gap_per_interval": gap, "layer_index_slope": gap,
            }

        rows = []
        for first, last in ((1.0, 1.0), (9.0, 81.0)):
            rows.append({
                "pre_activation_gradient_profile": profile(first, last),
                "post_activation_gradient_profile": profile(first, last),
                "saturation_by_layer": [
                    {"_values": [0.0], "mean_absolute_tanh_derivative": 1.0,
                     "fraction_abs_preactivation_gt_threshold": 0.0, "histogram_counts": [0, 1],
                     "quantiles": [0.0] * 7},
                    {"_values": [0.0], "mean_absolute_tanh_derivative": 1.0,
                     "fraction_abs_preactivation_gt_threshold": 0.0, "histogram_counts": [0, 1],
                     "quantiles": [0.0] * 7},
                ],
            })
        summary = aggregate_probes(rows, 2, 2.0, [0.0])
        gap = summary["gap_summaries"]["post_activation"]
        self.assertAlmostEqual(gap["mean_per_probe_raw_gap"], math.log10(3.0))
        self.assertAlmostEqual(gap["gap_from_arithmetic_mean_layer_rms"], math.log10(8.2))
        self.assertNotAlmostEqual(gap["mean_per_probe_raw_gap"], gap["gap_from_arithmetic_mean_layer_rms"])

    def test_checked_in_config_matches_frozen_contract(self) -> None:
        config = json.loads(Path("configs/experiment_v2.json").read_text(encoding="utf-8"))
        validate_config(config)


if __name__ == "__main__":
    unittest.main()
