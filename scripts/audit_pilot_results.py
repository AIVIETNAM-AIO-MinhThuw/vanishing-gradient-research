from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ols_slope(values: list[float]) -> float:
    logs = [math.log10(value) for value in values]
    center = (len(logs) - 1) / 2
    denominator = sum((index - center) ** 2 for index in range(len(logs)))
    return sum((index - center) * (value - statistics.fmean(logs)) for index, value in enumerate(logs)) / denominator


def all_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, dict):
        for child in value.values():
            yield from all_numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_numbers(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only consistency audit for pilot-v1 artifacts")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_files = sorted(path for path in run_dir.iterdir() if path.is_file())
    hashes_before = {path.name: sha256(path) for path in source_files}

    raw = json.loads((run_dir / "raw_results.json").read_text(encoding="utf-8"))
    spectrum_file = json.loads((run_dir / "supplementary_spectrum.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "primary_manifest.json").read_text(encoding="utf-8"))
    subset = json.loads((run_dir / "subset_indices.json").read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    with (run_dir / "primary_metrics.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    checks: dict[str, bool] = {}
    checks["raw_config_equals_config_file"] = raw["config"] == config
    checks["raw_spectrum_equals_supplementary_file"] = raw["supplementary_spectrum"] == spectrum_file["results"]
    checks["raw_primary_equals_manifest"] = raw["primary_results"] == manifest["results"]
    checks["six_primary_runs"] = len(raw["primary_results"]) == 6
    checks["three_diagnostic_runs"] = len(raw["diagnostic_results"]) == 3
    checks["no_recorded_run_errors"] = raw["errors"] == []
    checks["subset_counts"] = len(subset["train_indices"]) == 1024 and len(subset["validation_indices"]) == 256
    checks["subset_indices_unique_within_each_split"] = (
        len(set(subset["train_indices"])) == 1024 and len(set(subset["validation_indices"])) == 256
    )
    checks["config_matches_protocol_seeds"] = config["training"]["seeds"] == protocol["primary_experiment"]["approved_seeds"]
    checks["config_matches_protocol_checkpoints"] = config["training"]["checkpoints"] == protocol["primary_experiment"]["checkpoints"]

    csv_index: dict[tuple[str, int, int, int, str, int], dict[str, str]] = {}
    for row in csv_rows:
        key = (
            row["arm_id"], int(row["seed"]), int(row["step"]), int(row["dataset_index"]),
            row["gradient_location"], int(row["hidden_layer_index"]),
        )
        if key in csv_index:
            raise AssertionError(f"Duplicate CSV key: {key}")
        csv_index[key] = row

    aggregation_rows: list[dict[str, Any]] = []
    expected_csv_keys: set[tuple[str, int, int, int, str, int]] = set()
    max_formula_error = 0.0
    max_csv_error = 0.0
    invalid_gradient_values = 0
    sample_and_layer_counts: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for arm in raw["primary_results"]:
        depth = arm["hidden_layers"]
        for checkpoint in arm["checkpoints"]:
            step = checkpoint["step"]
            sample_and_layer_counts.append({
                "arm_id": arm["arm_id"], "depth": depth, "seed": arm["seed"], "step": step,
                "probe_samples": len(checkpoint["rows"]), "pre_layers_per_probe": depth,
                "post_layers_per_probe": depth,
            })
            eval_rows.append({
                "arm_id": arm["arm_id"], "depth": depth, "seed": arm["seed"], "step": step,
                "evaluation_loss": checkpoint["evaluation"]["loss"],
                "evaluation_accuracy": checkpoint["evaluation"]["accuracy"],
                "evaluation_sample_count": checkpoint["evaluation"]["sample_count"],
            })
            for location in ("pre_activation", "post_activation"):
                profile_name = f"{location}_gradient_profile"
                layer_values: list[list[float]] = [[] for _ in range(depth)]
                gaps: list[float] = []
                normalized: list[float] = []
                slopes: list[float] = []
                for sample in checkpoint["rows"]:
                    profile = sample[profile_name]
                    values = profile["gradient_rms"]
                    if len(values) != depth:
                        raise AssertionError("Layer count does not match depth")
                    invalid_gradient_values += sum(not math.isfinite(value) or value <= 0 for value in values)
                    computed_gap = math.log10(values[-1]) - math.log10(values[0])
                    computed_normalized = computed_gap / (depth - 1)
                    computed_slope = ols_slope(values)
                    max_formula_error = max(
                        max_formula_error,
                        abs(computed_gap - profile["raw_early_to_last_gap"]),
                        abs(computed_normalized - profile["normalized_gap_per_interval"]),
                        abs(computed_slope - profile["layer_index_slope"]),
                    )
                    gaps.append(profile["raw_early_to_last_gap"])
                    normalized.append(profile["normalized_gap_per_interval"])
                    slopes.append(profile["layer_index_slope"])
                    for layer_index, gradient_rms in enumerate(values):
                        layer_values[layer_index].append(gradient_rms)
                        key = (arm["arm_id"], arm["seed"], step, sample["dataset_index"], location, layer_index)
                        expected_csv_keys.add(key)
                        csv_row = csv_index[key]
                        differences = [
                            abs(float(csv_row["gradient_rms"]) - gradient_rms),
                            abs(float(csv_row["raw_early_to_last_gap"]) - profile["raw_early_to_last_gap"]),
                            abs(float(csv_row["normalized_gap_per_interval"]) - profile["normalized_gap_per_interval"]),
                            abs(float(csv_row["layer_index_slope"]) - profile["layer_index_slope"]),
                            abs(float(csv_row["evaluation_loss"]) - checkpoint["evaluation"]["loss"]),
                            abs(float(csv_row["evaluation_accuracy"]) - checkpoint["evaluation"]["accuracy"]),
                            abs(float(csv_row["logit_loss_gradient_norm"]) - sample["logit_loss_gradient_norm"]),
                        ]
                        max_csv_error = max(max_csv_error, *differences)

                arithmetic_means = [statistics.fmean(values) for values in layer_values]
                medians = [statistics.median(values) for values in layer_values]
                aggregation_rows.append({
                    "arm_id": arm["arm_id"], "depth": depth, "seed": arm["seed"], "step": step,
                    "gradient_location": location, "probe_count": len(gaps),
                    "first_layer_arithmetic_mean_rms": arithmetic_means[0],
                    "last_layer_arithmetic_mean_rms": arithmetic_means[-1],
                    "gap_from_arithmetic_mean_rms": math.log10(arithmetic_means[-1] / arithmetic_means[0]),
                    "first_layer_median_rms": medians[0],
                    "last_layer_median_rms": medians[-1],
                    "gap_from_median_rms": math.log10(medians[-1] / medians[0]),
                    "mean_of_per_sample_raw_gap": statistics.fmean(gaps),
                    "median_of_per_sample_raw_gap": statistics.median(gaps),
                    "mean_of_per_sample_normalized_gap": statistics.fmean(normalized),
                    "mean_of_per_sample_layer_slope": statistics.fmean(slopes),
                    "layerwise_arithmetic_mean_rms_json": json.dumps(arithmetic_means, separators=(",", ":")),
                    "layerwise_median_rms_json": json.dumps(medians, separators=(",", ":")),
                })

    checks["csv_row_count_2880"] = len(csv_rows) == 2880
    checks["csv_key_set_matches_raw_primary"] = set(csv_index) == expected_csv_keys
    checks["csv_numeric_values_match_raw_primary"] = max_csv_error == 0.0
    checks["gradient_formulas_recompute_with_float_tolerance"] = max_formula_error < 2e-6
    checks["all_primary_gradient_values_finite_positive"] = invalid_gradient_values == 0
    checks["all_evaluation_counts_256"] = all(row["evaluation_sample_count"] == 256 for row in eval_rows)
    checks["all_primary_checkpoints_are_0_and_100"] = all(
        [checkpoint["step"] for checkpoint in arm["checkpoints"]] == [0, 100]
        for arm in raw["primary_results"]
    )
    checks["fixed_probe_indices_across_primary"] = len({
        tuple(row["dataset_index"] for row in checkpoint["rows"])
        for arm in raw["primary_results"] for checkpoint in arm["checkpoints"]
    }) == 1
    checks["probes_are_first_16_validation_indices"] = all(
        [row["dataset_index"] for row in checkpoint["rows"]] == subset["validation_indices"][:16]
        for arm in raw["primary_results"] for checkpoint in arm["checkpoints"]
    )

    sample_spectrum_rows: list[dict[str, Any]] = []
    spectrum_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for snapshot in spectrum_file["results"]:
        for sample in snapshot["rows"]:
            for diagnostic in sample["jacobian_diagnostics"]:
                spectrum = diagnostic["spectrum"]
                gain = diagnostic["directional_gain"]
                row = {
                    "classification": snapshot["classification"], "arm_id": snapshot["arm_id"],
                    "depth": snapshot["hidden_layers"], "xavier_gain": snapshot["xavier_gain"],
                    "seed": snapshot["seed"], "step": snapshot["step"],
                    "dataset_index": sample["dataset_index"],
                    "hidden_layer_index": diagnostic["hidden_layer_index"],
                    "sigma_max": spectrum["sigma_max"], "sigma_min": spectrum["sigma_min"],
                    "condition_number_nonzero_spectrum": spectrum["condition_number_nonzero_spectrum"],
                    "effective_rank_energy_entropy": spectrum["effective_rank_energy_entropy"],
                    "frobenius_norm": spectrum["frobenius_norm"],
                    "normalized_directional_gain": gain["value"],
                    "directional_gain_excluded": gain["excluded_near_zero_denominator"],
                    "directional_gain_denominator_norm": gain["denominator_norm"],
                    "numerical_rank": spectrum["numerical_rank"],
                    "full_row_rank": spectrum["full_row_rank"],
                    "additional_numerical_rank_deficiency": spectrum["additional_numerical_rank_deficiency"],
                    "inherent_input_nullity": spectrum["inherent_input_nullity"],
                    "rank_tolerance": spectrum["rank_tolerance"],
                    "jacobian_rows": spectrum["shape"][0], "jacobian_columns": spectrum["shape"][1],
                }
                sample_spectrum_rows.append(row)
                key = (
                    row["classification"], row["arm_id"], row["depth"], row["xavier_gain"],
                    row["seed"], row["step"], row["hidden_layer_index"],
                )
                spectrum_groups[key].append(row)

    spectrum_summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(spectrum_groups.items()):
        classification, arm_id, depth, gain, seed, step, layer = key
        spectrum_summary_rows.append({
            "classification": classification, "arm_id": arm_id, "depth": depth, "xavier_gain": gain,
            "seed": seed, "step": step, "hidden_layer_index": layer, "probe_count": len(rows),
            "mean_sigma_max": statistics.fmean(row["sigma_max"] for row in rows),
            "mean_sigma_min": statistics.fmean(row["sigma_min"] for row in rows),
            "mean_condition_number_nonzero_spectrum": statistics.fmean(row["condition_number_nonzero_spectrum"] for row in rows),
            "mean_effective_rank_energy_entropy": statistics.fmean(row["effective_rank_energy_entropy"] for row in rows),
            "mean_frobenius_norm": statistics.fmean(row["frobenius_norm"] for row in rows),
            "mean_normalized_directional_gain": statistics.fmean(row["normalized_directional_gain"] for row in rows if row["normalized_directional_gain"] is not None),
            "directional_gain_excluded_count": sum(row["directional_gain_excluded"] for row in rows),
            "numerical_rank_min": min(row["numerical_rank"] for row in rows),
            "numerical_rank_max": max(row["numerical_rank"] for row in rows),
            "additional_rank_deficiency_count": sum(row["additional_numerical_rank_deficiency"] > 0 for row in rows),
            "all_full_row_rank": all(row["full_row_rank"] for row in rows),
            "inherent_input_nullity": rows[0]["inherent_input_nullity"],
        })

    checks["spectrum_has_216_sample_layer_rows"] = len(sample_spectrum_rows) == 216
    checks["spectrum_has_54_condition_layer_rows"] = len(spectrum_summary_rows) == 54
    checks["all_spectrum_shapes_10x128"] = all(
        row["jacobian_rows"] == 10 and row["jacobian_columns"] == 128 for row in sample_spectrum_rows
    )
    checks["no_directional_gain_exclusions"] = not any(row["directional_gain_excluded"] for row in sample_spectrum_rows)
    checks["all_spectrum_rows_full_row_rank"] = all(row["full_row_rank"] for row in sample_spectrum_rows)
    checks["no_additional_rank_deficiency"] = not any(
        row["additional_numerical_rank_deficiency"] for row in sample_spectrum_rows
    )

    diagnostic_arms = raw["diagnostic_results"]
    checks["diagnostic_metadata_depth12_gain0_5"] = all(
        arm["hidden_layers"] == 12 and arm["xavier_gain"] == 0.5 and
        arm["classification"] == "supplementary_diagnostic" for arm in diagnostic_arms
    )
    checks["primary_d12_metadata_depth12_gain1"] = all(
        arm["hidden_layers"] == 12 and arm["xavier_gain"] == 1.0
        for arm in raw["primary_results"] if arm["arm_id"] == "primary_d12"
    )
    checks["diagnostic_uses_same_checkpoint_and_probe_structure"] = all(
        [checkpoint["step"] for checkpoint in arm["checkpoints"]] == [0, 100] and
        all(len(checkpoint["rows"]) == 16 for checkpoint in arm["checkpoints"])
        for arm in diagnostic_arms
    )
    checks["all_raw_numbers_finite"] = all(math.isfinite(value) for value in all_numbers(raw))

    svg = (run_dir / "primary_gradient_gap.svg").read_text(encoding="utf-8")
    expected_plot_labels = {}
    for row in aggregation_rows:
        if row["step"] == 100 and row["gradient_location"] == "post_activation":
            key = f'{row["arm_id"]}/s{row["seed"]}'
            expected_plot_labels[key] = f'{row["mean_of_per_sample_raw_gap"]:.3f}'
    checks["svg_contains_all_primary_run_labels"] = all(label in svg for label in expected_plot_labels)
    checks["svg_contains_all_rounded_mean_sample_gap_values"] = all(value in svg for value in expected_plot_labels.values())

    hashes_after = {path.name: sha256(path) for path in source_files}
    checks["source_artifacts_unchanged_during_audit"] = hashes_before == hashes_after

    audit_summary = {
        "verdict": "AGGREGATION_DIFFERENCE" if all(checks.values()) else "DATA_ISSUE",
        "run_id": raw["run_id"],
        "source_directory": str(run_dir),
        "explanation": {
            "layer_table": "Arithmetic mean of per-sample RMS at each layer across 16 fixed probes.",
            "reported_gap_and_plot": "Arithmetic mean across 16 probes of each probe's log10(G_last/G_first).",
            "identity": "mean(log10(G_last/G_first)) = log10(geometric_mean(G_last)/geometric_mean(G_first)); it generally differs from log10(arithmetic_mean(G_last)/arithmetic_mean(G_first)).",
        },
        "checks": checks,
        "max_absolute_gradient_formula_recompute_error": max_formula_error,
        "max_absolute_csv_vs_raw_error": max_csv_error,
        "counts": {
            "primary_runs": len(raw["primary_results"]), "diagnostic_runs": len(raw["diagnostic_results"]),
            "primary_csv_rows": len(csv_rows), "spectrum_sample_layer_rows": len(sample_spectrum_rows),
            "spectrum_condition_layer_rows": len(spectrum_summary_rows),
            "spectrum_directional_gain_exclusions": sum(row["directional_gain_excluded"] for row in sample_spectrum_rows),
        },
        "metric_availability": {
            "measured": [
                "singular_values", "sigma_max", "sigma_min", "condition_number_nonzero_spectrum",
                "effective_rank_energy_entropy", "frobenius_norm", "normalized_directional_gain",
                "numerical_rank", "full_row_rank", "additional_numerical_rank_deficiency",
                "inherent_input_nullity", "rank_tolerance",
            ],
            "not_measured": [
                "uncertainty_intervals", "spectral metrics for all 16 gradient probes",
                "spectral metrics for every hidden layer", "weight-gradient spectrum",
            ],
        },
        "source_sha256": hashes_before,
        "environment_git_revision": raw["environment"].get("git_revision"),
    }

    write_csv_exclusive(output_dir / "gradient_aggregation_audit.csv", aggregation_rows)
    write_csv_exclusive(output_dir / "evaluation_metrics_audit.csv", eval_rows)
    write_csv_exclusive(output_dir / "sample_layer_counts.csv", sample_and_layer_counts)
    write_csv_exclusive(output_dir / "jacobian_metrics_by_probe.csv", sample_spectrum_rows)
    write_csv_exclusive(output_dir / "jacobian_metrics_summary.csv", spectrum_summary_rows)
    write_json_exclusive(output_dir / "audit_summary.json", audit_summary)
    print(json.dumps(audit_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
