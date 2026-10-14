from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def all_numbers(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from all_numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_numbers(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    raw = json.loads((result_dir / "raw_posthoc_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((result_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((result_dir / "posthoc_audit.json").read_text(encoding="utf-8"))
    gradient_comparison = list(csv.DictReader((result_dir / "gradient_gap_comparison.csv").open(encoding="utf-8")))
    jacobian_comparison = list(csv.DictReader((result_dir / "jacobian_comparison.csv").open(encoding="utf-8")))

    profile_rows, layer_rows = [], []
    max_gap_error = 0.0
    max_normalized_error = 0.0
    gradient_index_sets = set()
    for checkpoint in raw["gradient_results"]:
        indices = tuple(row["dataset_index"] for row in checkpoint["probe_rows"])
        gradient_index_sets.add(indices)
        for probe in checkpoint["probe_rows"]:
            for location in ("pre_activation", "post_activation"):
                profile = probe[f"{location}_gradient_profile"]
                computed_gap = math.log10(profile["gradient_rms"][-1] / profile["gradient_rms"][0])
                max_gap_error = max(max_gap_error, abs(computed_gap - profile["raw_early_to_last_gap"]))
                max_normalized_error = max(
                    max_normalized_error,
                    abs(computed_gap / (checkpoint["depth"] - 1) - profile["normalized_gap_per_interval"]),
                )
                profile_rows.append({
                    "depth": checkpoint["depth"], "seed": checkpoint["seed"], "step": checkpoint["step"],
                    "dataset_index": probe["dataset_index"], "target_class": probe["target_class"],
                    "gradient_location": location, "raw_gap": profile["raw_early_to_last_gap"],
                    "normalized_gap": profile["normalized_gap_per_interval"],
                    "layer_slope": profile["layer_index_slope"],
                    "logit_loss_gradient_norm": probe["logit_loss_gradient_norm"],
                })
        for layer in checkpoint["summary"]["layer_summaries"]:
            layer_rows.append({
                "depth": checkpoint["depth"], "seed": checkpoint["seed"], "step": checkpoint["step"],
                "hidden_layer_index": layer["hidden_layer_index"],
                "mean_pre_activation_gradient_rms": layer["mean_pre_activation_gradient_rms"],
                "median_pre_activation_gradient_rms": layer["median_pre_activation_gradient_rms"],
                "mean_post_activation_gradient_rms": layer["mean_post_activation_gradient_rms"],
                "median_post_activation_gradient_rms": layer["median_post_activation_gradient_rms"],
                "mean_absolute_tanh_derivative": layer["mean_absolute_tanh_derivative"],
                "fraction_abs_preactivation_gt_2": layer["fraction_abs_preactivation_gt_threshold"],
                "preactivation_histogram_counts_json": json.dumps(layer["histogram_counts"]),
                "preactivation_quantiles_json": json.dumps(layer["quantiles"]),
            })

    jacobian_index_sets = {
        tuple(row["dataset_index"] for row in snapshot["rows"])
        for snapshot in raw["jacobian_results"]
    }
    gradient_flags = sum(row["numerical_change_flag"].lower() == "true" for row in gradient_comparison)
    jacobian_flags = sum(row["numerical_change_flag"].lower() == "true" for row in jacobian_comparison)
    checks = {
        "classification_is_post_hoc_exploratory": raw["classification"] == "post_hoc_exploratory_measurement",
        "claim_remains_unverified": raw["claim_status"] == "unverified" and raw["scientific_conclusions"] is None,
        "40_balanced_gradient_probes": manifest["gradient_label_histogram"] == {str(label): 4 for label in range(10)},
        "20_balanced_jacobian_probes": manifest["jacobian_label_histogram"] == {str(label): 2 for label in range(10)},
        "jacobian_subset_of_gradient": set(manifest["jacobian_probe_indices"]) <= set(manifest["gradient_probe_indices"]),
        "no_training_overlap": manifest["training_overlap_count"] == 0,
        "official_test_unused": manifest["official_test_used"] is False,
        "all_models_same_gradient_indices": len(gradient_index_sets) == 1 and next(iter(gradient_index_sets)) == tuple(manifest["gradient_probe_indices"]),
        "all_models_same_jacobian_indices": len(jacobian_index_sets) == 1 and next(iter(jacobian_index_sets)) == tuple(manifest["jacobian_probe_indices"]),
        "60_saved_checkpoints_loaded": audit["loaded_checkpoint_count"] == 60 and audit["unique_loaded_checkpoint_count"] == 60,
        "no_training_executed": audit["training_executed"] is False,
        "source_v2_unchanged": audit["source_artifacts_unchanged"] is True,
        "gradient_formulas_consistent": max_gap_error < 2e-6 and max_normalized_error < 2e-6,
        "no_nonfinite_values": all(math.isfinite(value) for value in all_numbers(raw)),
        "expected_measurement_counts": (
            len(raw["gradient_results"]) == 60 and len(profile_rows) == 4800
            and len(raw["jacobian_results"]) == 27
            and sum(len(snapshot["rows"]) for snapshot in raw["jacobian_results"]) == 540
        ),
    }
    verification = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "max_gap_formula_error": max_gap_error,
        "max_normalized_gap_formula_error": max_normalized_error,
        "gradient_comparison_rows": len(gradient_comparison),
        "gradient_numerical_change_flag_rows": gradient_flags,
        "jacobian_comparison_rows": len(jacobian_comparison),
        "jacobian_numerical_change_flag_rows": jacobian_flags,
        "note": "Numerical change flags use frozen descriptive thresholds and are not statistical significance tests."
    }
    write_csv_exclusive(result_dir / "gradient_probe_profiles.csv", profile_rows)
    write_csv_exclusive(result_dir / "gradient_saturation_layer_summary.csv", layer_rows)
    with (result_dir / "verification.json").open("x", encoding="utf-8") as handle:
        json.dump(verification, handle, indent=2)
        handle.write("\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
