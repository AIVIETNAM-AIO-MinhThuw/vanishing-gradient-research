from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import torch
from torchvision import __version__ as torchvision_version
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vanishing_gradient.experiment_v2 import (
    PilotMLP,
    aggregate_probes,
    measure_jacobian,
    measure_probe,
    selected_layers,
    write_json_exclusive,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_probe_indices(
    validation_indices: Sequence[int], targets: torch.Tensor, per_label: int, seed: int
) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for label in range(10):
        candidates = [index for index in validation_indices if int(targets[index]) == label]
        if len(candidates) < per_label:
            raise ValueError(f"Label {label} has only {len(candidates)} validation candidates")
        order = torch.randperm(len(candidates), generator=generator)[:per_label].tolist()
        selected.extend(candidates[position] for position in order)
    return selected


def label_histogram(indices: Sequence[int], targets: torch.Tensor) -> dict[str, int]:
    counts = Counter(int(targets[index]) for index in indices)
    return {str(label): counts[label] for label in range(10)}


def checkpoint_specs(raw: dict[str, Any]) -> list[tuple[str, int, int, int]]:
    return [
        (run["arm_id"], run["hidden_layers"], run["seed"], checkpoint["step"])
        for run in raw["primary_results"] for checkpoint in run["checkpoints"]
    ]


def load_checkpoint_model(path: Path, depth: int) -> PilotMLP:
    model = PilotMLP(784, 128, depth, 10, 1.0)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def old_jacobian_summary(raw: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, float]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for snapshot in raw["jacobian_results"]:
        for probe in snapshot["rows"]:
            for item in probe["jacobian_diagnostics"]:
                groups[(snapshot["depth"], snapshot["seed"], snapshot["step"], item["hidden_layer_index"])].append(item)
    return {
        key: {
            "sigma_max": fmean(item["spectrum"]["sigma_max"] for item in items),
            "sigma_min": fmean(item["spectrum"]["sigma_min"] for item in items),
            "condition_number": fmean(item["spectrum"]["condition_number_nonzero_spectrum"] for item in items),
            "effective_rank": fmean(item["spectrum"]["effective_rank_energy_entropy"] for item in items),
            "frobenius_norm": fmean(item["spectrum"]["frobenius_norm"] for item in items),
            "normalized_directional_gain": fmean(item["directional_gain"]["value"] for item in items),
        }
        for key, items in groups.items()
    }


def relative_delta(old: float, new: float) -> float:
    return abs(new - old) / max(abs(old), abs(new), 1e-12)


def run(source_run: Path, protocol_path: Path, output_dir: Path) -> None:
    started = time.perf_counter()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_MEASUREMENT":
        raise ValueError("Post-hoc protocol is not frozen")
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "run.log"

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    source_files = sorted(path for path in source_run.rglob("*") if path.is_file())
    source_hashes_before = {str(path.relative_to(source_run)): sha256(path) for path in source_files}
    raw = json.loads((source_run / "raw_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((source_run / "data_and_batch_manifest.json").read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads((source_run / "checkpoint_manifest.json").read_text(encoding="utf-8"))["checkpoints"]
    recorded_checkpoints = {row["path"]: row for row in checkpoint_manifest}
    specs = checkpoint_specs(raw)
    if len(specs) != 60 or len(set(specs)) != 60:
        raise ValueError("Expected exactly 60 unique v2 checkpoint specifications")

    dataset = FashionMNIST("data", train=True, download=False, transform=ToTensor())
    validation_indices = manifest["validation_indices"]
    train_indices = set(manifest["train_indices"])
    seed = protocol["probe_selection"]["seed"]
    gradient_indices = balanced_probe_indices(validation_indices, dataset.targets, 4, seed)
    jacobian_indices = [gradient_indices[label * 4 + offset] for label in range(10) for offset in range(2)]
    gradient_histogram = label_histogram(gradient_indices, dataset.targets)
    jacobian_histogram = label_histogram(jacobian_indices, dataset.targets)
    if gradient_histogram != {str(label): 4 for label in range(10)}:
        raise AssertionError("Gradient probe histogram is not balanced")
    if jacobian_histogram != {str(label): 2 for label in range(10)}:
        raise AssertionError("Jacobian probe histogram is not balanced")
    if set(gradient_indices) & train_indices or not set(gradient_indices) <= set(validation_indices):
        raise AssertionError("Balanced probes violate persisted split boundaries")
    if not set(jacobian_indices) <= set(gradient_indices):
        raise AssertionError("Jacobian probes are not a subset of gradient probes")
    write_json_exclusive(output_dir / "probe_manifest.json", {
        "classification": "post_hoc_exploratory", "selection_seed": seed,
        "gradient_probe_indices": gradient_indices, "gradient_label_histogram": gradient_histogram,
        "jacobian_probe_indices": jacobian_indices, "jacobian_label_histogram": jacobian_histogram,
        "training_overlap_count": len(set(gradient_indices) & train_indices),
        "official_test_used": False,
    })
    log(f"probe_histograms gradient={gradient_histogram} jacobian={jacobian_histogram}")

    threshold = raw["config"]["measurement"]["saturation_absolute_threshold"]
    boundaries = raw["config"]["measurement"]["preactivation_histogram_boundaries"]
    old_by_key = {
        (run["hidden_layers"], run["seed"], checkpoint["step"]): checkpoint
        for run in raw["primary_results"] for checkpoint in run["checkpoints"]
    }
    old_jacobian = old_jacobian_summary(raw)
    gradient_results = []
    jacobian_results = []
    gradient_comparison = []
    jacobian_by_probe = []
    loaded_checkpoint_files: list[dict[str, Any]] = []

    for arm_id, depth, run_seed, step in specs:
        filename = f"{arm_id}_step_{step}.pt"
        checkpoint_path = source_run / "checkpoints" / filename
        recorded = recorded_checkpoints.get(filename)
        if recorded is None or not checkpoint_path.exists() or sha256(checkpoint_path) != recorded["sha256"]:
            raise ValueError(f"Missing or hash-mismatched checkpoint: {filename}")
        model = load_checkpoint_model(checkpoint_path, depth)
        loaded_checkpoint_files.append({"filename": filename, "sha256": recorded["sha256"]})
        probe_rows = []
        for dataset_index in gradient_indices:
            image, target = dataset[dataset_index]
            row = measure_probe(model, image, torch.tensor(target), threshold, boundaries)
            row["dataset_index"] = dataset_index
            probe_rows.append(row)
        summary = aggregate_probes(probe_rows, depth, threshold, boundaries)
        gradient_results.append({
            "arm_id": arm_id, "depth": depth, "seed": run_seed, "step": step,
            "checkpoint_file": filename, "probe_rows": probe_rows, "summary": summary,
        })
        old_checkpoint = old_by_key[(depth, run_seed, step)]
        for location in ("pre_activation", "post_activation"):
            old_summary = old_checkpoint["summary"]["gap_summaries"][location]
            new_summary = summary["gap_summaries"][location]
            gap_delta = new_summary["mean_per_probe_raw_gap"] - old_summary["mean_per_probe_raw_gap"]
            slope_delta = new_summary["mean_per_probe_layer_slope"] - old_summary["mean_per_probe_layer_slope"]
            gradient_comparison.append({
                "depth": depth, "seed": run_seed, "step": step, "gradient_location": location,
                "old_probe_count": 16, "new_probe_count": 40,
                "old_mean_raw_gap": old_summary["mean_per_probe_raw_gap"],
                "new_mean_raw_gap": new_summary["mean_per_probe_raw_gap"], "raw_gap_delta": gap_delta,
                "old_mean_normalized_gap": old_summary["mean_per_probe_normalized_gap"],
                "new_mean_normalized_gap": new_summary["mean_per_probe_normalized_gap"],
                "normalized_gap_delta": new_summary["mean_per_probe_normalized_gap"] - old_summary["mean_per_probe_normalized_gap"],
                "old_mean_layer_slope": old_summary["mean_per_probe_layer_slope"],
                "new_mean_layer_slope": new_summary["mean_per_probe_layer_slope"], "layer_slope_delta": slope_delta,
                "numerical_change_flag": abs(gap_delta) >= 0.05 or abs(slope_delta) >= 0.05,
            })

        if depth in (3, 12, 24) and step in (0, 100, 600):
            rows = []
            exclusions = 0
            for dataset_index in jacobian_indices:
                image, target = dataset[dataset_index]
                diagnostics = measure_jacobian(model, image, torch.tensor(target), selected_layers(depth), 1e-6)
                exclusions += sum(int(item["directional_gain"]["excluded_near_zero_denominator"]) for item in diagnostics)
                rows.append({"dataset_index": dataset_index, "target_class": int(target), "jacobian_diagnostics": diagnostics})
                for item in diagnostics:
                    spectrum = item["spectrum"]
                    gain = item["directional_gain"]
                    jacobian_by_probe.append({
                        "depth": depth, "seed": run_seed, "step": step, "dataset_index": dataset_index,
                        "target_class": int(target), "hidden_layer_index": item["hidden_layer_index"],
                        "sigma_max": spectrum["sigma_max"], "sigma_min": spectrum["sigma_min"],
                        "condition_number": spectrum["condition_number_nonzero_spectrum"],
                        "effective_rank": spectrum["effective_rank_energy_entropy"],
                        "frobenius_norm": spectrum["frobenius_norm"],
                        "normalized_directional_gain": gain["value"],
                        "numerical_rank": spectrum["numerical_rank"],
                        "additional_rank_deficiency": spectrum["additional_numerical_rank_deficiency"],
                        "directional_gain_excluded": gain["excluded_near_zero_denominator"],
                    })
            jacobian_results.append({
                "arm_id": arm_id, "depth": depth, "seed": run_seed, "step": step,
                "checkpoint_file": filename, "probe_count": len(rows),
                "directional_gain_excluded_count": exclusions, "rows": rows,
            })
        log(f"measured checkpoint={filename}")

    new_jacobian_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in jacobian_by_probe:
        new_jacobian_groups[(row["depth"], row["seed"], row["step"], row["hidden_layer_index"])].append(row)
    jacobian_comparison = []
    fields = ("sigma_max", "sigma_min", "condition_number", "effective_rank", "frobenius_norm", "normalized_directional_gain")
    for key, rows in sorted(new_jacobian_groups.items()):
        depth, run_seed, step, layer = key
        old = old_jacobian[key]
        new = {field: fmean(row[field] for row in rows) for field in fields}
        deltas = {f"{field}_symmetric_relative_delta": relative_delta(old[field], new[field]) for field in fields}
        jacobian_comparison.append({
            "depth": depth, "seed": run_seed, "step": step, "hidden_layer_index": layer,
            "old_probe_count": 8, "new_probe_count": 20,
            **{f"old_{field}": old[field] for field in fields},
            **{f"new_{field}": new[field] for field in fields}, **deltas,
            "numerical_change_flag": any(value >= 0.20 for value in deltas.values()),
            "new_numerical_rank_min": min(row["numerical_rank"] for row in rows),
            "new_numerical_rank_max": max(row["numerical_rank"] for row in rows),
            "new_additional_rank_deficiency_count": sum(row["additional_rank_deficiency"] > 0 for row in rows),
            "new_directional_gain_excluded_count": sum(row["directional_gain_excluded"] for row in rows),
        })

    source_hashes_after = {str(path.relative_to(source_run)): sha256(path) for path in source_files}
    audit = {
        "status": "PASS",
        "classification": "post_hoc_exploratory_measurement",
        "training_executed": False,
        "loaded_checkpoint_count": len(loaded_checkpoint_files),
        "unique_loaded_checkpoint_count": len({row["filename"] for row in loaded_checkpoint_files}),
        "gradient_checkpoint_measurements": len(gradient_results),
        "gradient_probe_sample_measurements": len(gradient_results) * len(gradient_indices),
        "jacobian_checkpoint_measurements": len(jacobian_results),
        "jacobian_probe_measurements": len(jacobian_results) * len(jacobian_indices),
        "jacobian_probe_layer_measurements": len(jacobian_by_probe),
        "source_artifacts_unchanged": source_hashes_before == source_hashes_after,
        "gradient_label_histogram": gradient_histogram,
        "jacobian_label_histogram": jacobian_histogram,
        "training_overlap_count": len(set(gradient_indices) & train_indices),
        "same_gradient_indices_for_all_models": True,
        "same_jacobian_indices_for_all_models": True,
        "directional_gain_excluded_count": sum(row["directional_gain_excluded"] for row in jacobian_by_probe),
        "nonfinite_count": sum(
            not math.isfinite(float(value)) for row in gradient_comparison + jacobian_comparison
            for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        "source_sha256_before": source_hashes_before,
    }
    if audit["loaded_checkpoint_count"] != 60 or audit["unique_loaded_checkpoint_count"] != 60:
        raise AssertionError("Did not load all 60 saved checkpoints exactly once")
    if not audit["source_artifacts_unchanged"] or audit["nonfinite_count"]:
        raise AssertionError("Source mutation or non-finite output detected")

    write_json_exclusive(output_dir / "raw_posthoc_results.json", {
        "classification": "post_hoc_exploratory_measurement", "claim_status": "unverified",
        "source_run_id": raw["run_id"], "protocol": protocol,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision_version, "git_revision": git_revision()},
        "probe_manifest": {"gradient_indices": gradient_indices, "jacobian_indices": jacobian_indices},
        "gradient_results": gradient_results, "jacobian_results": jacobian_results,
        "runtime_seconds": time.perf_counter() - started, "scientific_conclusions": None,
    })
    write_csv_exclusive(output_dir / "gradient_gap_comparison.csv", gradient_comparison)
    write_csv_exclusive(output_dir / "jacobian_by_probe.csv", jacobian_by_probe)
    write_csv_exclusive(output_dir / "jacobian_comparison.csv", jacobian_comparison)
    write_json_exclusive(output_dir / "posthoc_audit.json", audit)
    log(f"complete runtime_seconds={time.perf_counter() - started:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_run.resolve(), args.protocol.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
