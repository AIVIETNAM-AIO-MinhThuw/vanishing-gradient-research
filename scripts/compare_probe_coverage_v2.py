from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def relative_delta(old: float, new: float) -> float:
    return abs(new - old) / max(abs(old), abs(new), 1e-12)


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--posthoc-dir", type=Path, required=True)
    args = parser.parse_args()
    old_raw = json.loads((args.source_run / "raw_results.json").read_text(encoding="utf-8"))
    new_raw = json.loads((args.posthoc_dir / "raw_posthoc_results.json").read_text(encoding="utf-8"))
    old = {
        (run["hidden_layers"], run["seed"], checkpoint["step"]): checkpoint
        for run in old_raw["primary_results"] for checkpoint in run["checkpoints"]
    }
    layer_rows, logit_rows = [], []
    layer_metrics = (
        "mean_pre_activation_gradient_rms", "mean_post_activation_gradient_rms",
        "mean_absolute_tanh_derivative", "fraction_abs_preactivation_gt_threshold",
    )
    for checkpoint in new_raw["gradient_results"]:
        key = (checkpoint["depth"], checkpoint["seed"], checkpoint["step"])
        old_checkpoint = old[key]
        for new_layer, old_layer in zip(
            checkpoint["summary"]["layer_summaries"], old_checkpoint["summary"]["layer_summaries"], strict=True
        ):
            row: dict[str, Any] = {
                "depth": key[0], "seed": key[1], "step": key[2],
                "hidden_layer_index": new_layer["hidden_layer_index"],
            }
            for metric in layer_metrics:
                old_value, new_value = old_layer[metric], new_layer[metric]
                row[f"old_{metric}"] = old_value
                row[f"new_{metric}"] = new_value
                row[f"{metric}_absolute_delta"] = new_value - old_value
                row[f"{metric}_symmetric_relative_delta"] = relative_delta(old_value, new_value)
            layer_rows.append(row)
        old_norm = fmean(row["logit_loss_gradient_norm"] for row in old_checkpoint["probe_rows"])
        new_norm = fmean(row["logit_loss_gradient_norm"] for row in checkpoint["probe_rows"])
        logit_rows.append({
            "depth": key[0], "seed": key[1], "step": key[2],
            "old_mean_logit_loss_gradient_norm": old_norm,
            "new_mean_logit_loss_gradient_norm": new_norm,
            "absolute_delta": new_norm - old_norm,
            "symmetric_relative_delta": relative_delta(old_norm, new_norm),
        })
    write_csv_exclusive(args.posthoc_dir / "layer_metric_comparison.csv", layer_rows)
    write_csv_exclusive(args.posthoc_dir / "logit_gradient_norm_comparison.csv", logit_rows)
    print(json.dumps({"layer_rows": len(layer_rows), "logit_rows": len(logit_rows)}, indent=2))


if __name__ == "__main__":
    main()
