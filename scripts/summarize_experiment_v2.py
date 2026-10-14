from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    raw = json.loads((run_dir / "raw_results.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((run_dir / "data_and_batch_manifest.json").read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads((run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))["checkpoints"]

    checks: dict[str, bool] = {}
    checks["12_primary_runs"] = len(raw["primary_results"]) == 12
    checks["depth_seed_grid_complete"] = {
        (run["hidden_layers"], run["seed"]) for run in raw["primary_results"]
    } == {(depth, seed) for depth in (3, 6, 12, 24) for seed in (42, 123, 2026)}
    checks["five_frozen_checkpoints_each"] = all(
        [checkpoint["step"] for checkpoint in run["checkpoints"]] == [0, 25, 100, 300, 600]
        for run in raw["primary_results"]
    )
    checks["fixed_16_probes_everywhere"] = all(
        len(checkpoint["probe_rows"]) == 16
        and [row["dataset_index"] for row in checkpoint["probe_rows"]] == data_manifest["fixed_probe_indices"]
        for run in raw["primary_results"] for checkpoint in run["checkpoints"]
    )
    checks["split_counts_and_disjointness"] = (
        len(data_manifest["train_indices"]) == 10000
        and len(data_manifest["validation_indices"]) == 2000
        and len(set(data_manifest["train_indices"])) == 10000
        and len(set(data_manifest["validation_indices"])) == 2000
        and not (set(data_manifest["train_indices"]) & set(data_manifest["validation_indices"]))
    )
    checks["official_test_unused"] = data_manifest["official_test_used"] is False
    checks["batch_orders_complete"] = all(
        len(batches) == 600 and all(len(batch) == 64 for batch in batches)
        for batches in data_manifest["batch_order"].values()
    )
    checks["evaluation_sample_counts"] = all(
        checkpoint["training"]["sample_count"] == 10000
        and checkpoint["validation"]["sample_count"] == 2000
        for run in raw["primary_results"] for checkpoint in run["checkpoints"]
    )
    checks["no_recorded_errors"] = raw["errors"] == []

    max_gap_error = 0.0
    max_normalized_error = 0.0
    nonfinite_values = 0
    gap_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    saturation_rows: list[dict[str, Any]] = []
    for run in raw["primary_results"]:
        base = {"arm_id": run["arm_id"], "depth": run["hidden_layers"], "seed": run["seed"]}
        for checkpoint in run["checkpoints"]:
            evaluation_rows.append({
                **base, "step": checkpoint["step"],
                "training_loss": checkpoint["training"]["loss"],
                "training_accuracy": checkpoint["training"]["accuracy"],
                "validation_loss": checkpoint["validation"]["loss"],
                "validation_accuracy": checkpoint["validation"]["accuracy"],
            })
            for location, summary in checkpoint["summary"]["gap_summaries"].items():
                gap_rows.append({**base, "step": checkpoint["step"], "gradient_location": location, **summary})
            for layer in checkpoint["summary"]["layer_summaries"]:
                saturation_rows.append({
                    **base, "step": checkpoint["step"],
                    "hidden_layer_index": layer["hidden_layer_index"],
                    "mean_pre_activation_gradient_rms": layer["mean_pre_activation_gradient_rms"],
                    "mean_post_activation_gradient_rms": layer["mean_post_activation_gradient_rms"],
                    "mean_absolute_tanh_derivative": layer["mean_absolute_tanh_derivative"],
                    "fraction_abs_preactivation_gt_2": layer["fraction_abs_preactivation_gt_threshold"],
                    "preactivation_histogram_counts_json": json.dumps(layer["histogram_counts"]),
                    "preactivation_quantiles_json": json.dumps(layer["quantiles"]),
                })
            for probe in checkpoint["probe_rows"]:
                values_to_check = [probe["loss"], probe["logit_loss_gradient_norm"], *probe["logit_loss_gradient"]]
                for location in ("pre_activation", "post_activation"):
                    profile = probe[f"{location}_gradient_profile"]
                    values = profile["gradient_rms"]
                    computed_gap = math.log10(values[-1] / values[0])
                    max_gap_error = max(max_gap_error, abs(computed_gap - profile["raw_early_to_last_gap"]))
                    max_normalized_error = max(
                        max_normalized_error,
                        abs(computed_gap / (run["hidden_layers"] - 1) - profile["normalized_gap_per_interval"]),
                    )
                    values_to_check.extend(values)
                nonfinite_values += sum(not math.isfinite(value) for value in values_to_check)
    checks["gradient_formulas_recomputed"] = max_gap_error < 2e-6 and max_normalized_error < 2e-6
    checks["all_checked_primary_values_finite"] = nonfinite_values == 0

    checkpoint_hash_failures = []
    checkpoint_nonfinite_tensors = 0
    for row in checkpoint_manifest:
        path = run_dir / "checkpoints" / row["path"]
        if not path.exists() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            checkpoint_hash_failures.append(row["path"])
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_nonfinite_tensors += sum(not bool(tensor.isfinite().all()) for tensor in state.values())
    checks["60_checkpoint_files_hash_verified"] = len(checkpoint_manifest) == 60 and not checkpoint_hash_failures
    checks["all_checkpoint_tensors_finite"] = checkpoint_nonfinite_tensors == 0

    jacobian_rows: list[dict[str, Any]] = []
    jacobian_nonfinite = 0
    exclusions = 0
    for snapshot in raw["jacobian_results"]:
        for probe in snapshot["rows"]:
            for item in probe["jacobian_diagnostics"]:
                spectrum = item["spectrum"]
                gain = item["directional_gain"]
                exclusions += int(gain["excluded_near_zero_denominator"])
                values = [
                    spectrum["sigma_max"], spectrum["sigma_min"], spectrum["frobenius_norm"],
                    spectrum["effective_rank_energy_entropy"], spectrum["condition_number_nonzero_spectrum"],
                    *spectrum["singular_values"], gain["denominator_norm"], gain["numerator_norm"],
                ]
                if gain["value"] is not None:
                    values.append(gain["value"])
                jacobian_nonfinite += sum(not math.isfinite(value) for value in values)
                jacobian_rows.append({
                    "arm_id": snapshot["arm_id"], "depth": snapshot["depth"], "seed": snapshot["seed"],
                    "step": snapshot["step"], "dataset_index": probe["dataset_index"],
                    "hidden_layer_index": item["hidden_layer_index"],
                    "sigma_max": spectrum["sigma_max"], "sigma_min": spectrum["sigma_min"],
                    "condition_number": spectrum["condition_number_nonzero_spectrum"],
                    "effective_rank": spectrum["effective_rank_energy_entropy"],
                    "frobenius_norm": spectrum["frobenius_norm"],
                    "normalized_directional_gain": gain["value"],
                    "numerical_rank": spectrum["numerical_rank"],
                    "additional_rank_deficiency": spectrum["additional_numerical_rank_deficiency"],
                    "inherent_input_nullity": spectrum["inherent_input_nullity"],
                    "excluded_directional_gain": gain["excluded_near_zero_denominator"],
                })
    checks["27_complete_jacobian_snapshots"] = (
        len(raw["jacobian_results"]) == 27
        and all(len(snapshot["rows"]) == 8 for snapshot in raw["jacobian_results"])
        and len(jacobian_rows) == 648
    )
    checks["jacobian_values_finite"] = jacobian_nonfinite == 0
    checks["no_directional_gain_exclusions"] = exclusions == 0

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in jacobian_rows:
        grouped[(row["arm_id"], row["depth"], row["seed"], row["step"], row["hidden_layer_index"])].append(row)
    jacobian_summary = []
    for key, rows in sorted(grouped.items()):
        arm_id, depth, seed, step, layer = key
        jacobian_summary.append({
            "arm_id": arm_id, "depth": depth, "seed": seed, "step": step,
            "hidden_layer_index": layer, "probe_count": len(rows),
            "mean_sigma_max": statistics.fmean(row["sigma_max"] for row in rows),
            "mean_sigma_min": statistics.fmean(row["sigma_min"] for row in rows),
            "mean_condition_number": statistics.fmean(row["condition_number"] for row in rows),
            "mean_effective_rank": statistics.fmean(row["effective_rank"] for row in rows),
            "mean_frobenius_norm": statistics.fmean(row["frobenius_norm"] for row in rows),
            "mean_normalized_directional_gain": statistics.fmean(row["normalized_directional_gain"] for row in rows),
            "numerical_rank_min": min(row["numerical_rank"] for row in rows),
            "numerical_rank_max": max(row["numerical_rank"] for row in rows),
            "additional_rank_deficiency_count": sum(row["additional_rank_deficiency"] > 0 for row in rows),
            "directional_gain_excluded_count": sum(row["excluded_directional_gain"] for row in rows),
        })

    write_csv_exclusive(run_dir / "gap_slope_summary.csv", gap_rows)
    write_csv_exclusive(run_dir / "evaluation_summary.csv", evaluation_rows)
    write_csv_exclusive(run_dir / "saturation_summary.csv", saturation_rows)
    write_csv_exclusive(run_dir / "jacobian_by_probe.csv", jacobian_rows)
    write_csv_exclusive(run_dir / "jacobian_summary.csv", jacobian_summary)
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "max_gap_formula_error": max_gap_error,
        "max_normalized_gap_formula_error": max_normalized_error,
        "nonfinite_primary_value_count": nonfinite_values,
        "checkpoint_hash_failures": checkpoint_hash_failures,
        "checkpoint_nonfinite_tensor_count": checkpoint_nonfinite_tensors,
        "jacobian_nonfinite_value_count": jacobian_nonfinite,
        "directional_gain_exclusion_count": exclusions,
    }
    with (run_dir / "post_run_audit.json").open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
