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
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision import __version__ as torchvision_version
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

try:
    from vanishing_gradient.pilot import (
        PilotMLP,
        gradient_profile_metrics,
        jacobian_spectrum_metrics,
        normalized_directional_gain,
    )
except ModuleNotFoundError:  # Supports direct execution from the source file.
    from pilot import PilotMLP, gradient_profile_metrics, jacobian_spectrum_metrics, normalized_directional_gain


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def stratified_split(
    targets: Tensor, train_count: int, validation_count: int, seed: int
) -> tuple[list[int], list[int]]:
    classes = torch.unique(targets, sorted=True)
    if train_count % len(classes) or validation_count % len(classes):
        raise ValueError("Requested split counts must divide evenly across classes")
    train_per_class = train_count // len(classes)
    validation_per_class = validation_count // len(classes)
    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_id in classes.tolist():
        class_indices = torch.nonzero(targets == class_id, as_tuple=False).flatten()
        required = train_per_class + validation_per_class
        if required > class_indices.numel():
            raise ValueError(f"Class {class_id} has too few samples")
        selected = class_indices[torch.randperm(class_indices.numel(), generator=generator)[:required]]
        train_indices.extend(selected[:train_per_class].tolist())
        validation_indices.extend(selected[train_per_class:].tolist())
    if set(train_indices) & set(validation_indices):
        raise AssertionError("Training and validation indices overlap")
    return train_indices, validation_indices


def deterministic_batch_positions(sample_count: int, batch_size: int, steps: int, seed: int) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    positions: list[int] = []
    required = batch_size * steps
    while len(positions) < required:
        positions.extend(torch.randperm(sample_count, generator=generator).tolist())
    return [positions[offset : offset + batch_size] for offset in range(0, required, batch_size)]


def cache_images(dataset: FashionMNIST, indices: Sequence[int]) -> tuple[Tensor, Tensor]:
    images, targets = zip(*(dataset[index] for index in indices))
    return torch.stack(images), torch.tensor(targets, dtype=torch.long)


def evaluate(model: PilotMLP, images: Tensor, targets: Tensor, batch_size: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for offset in range(0, len(images), batch_size):
            batch_images = images[offset : offset + batch_size].to(device)
            batch_targets = targets[offset : offset + batch_size].to(device)
            logits, _ = model(batch_images)
            total_loss += float(F.cross_entropy(logits, batch_targets, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == batch_targets).sum().item())
    return {"loss": total_loss / len(images), "accuracy": correct / len(images), "sample_count": len(images)}


def saturation_metrics(preactivation: Tensor, threshold: float, boundaries: Sequence[float]) -> dict[str, Any]:
    values = preactivation.detach().flatten().cpu()
    derivative = 1.0 - torch.tanh(values).square()
    bucket = torch.bucketize(values, torch.tensor(boundaries, dtype=values.dtype))
    counts = torch.bincount(bucket, minlength=len(boundaries) + 1)
    quantiles = torch.quantile(values, torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]))
    return {
        "mean_absolute_tanh_derivative": float(derivative.abs().mean().item()),
        "fraction_abs_preactivation_gt_threshold": float((values.abs() > threshold).float().mean().item()),
        "histogram_counts": counts.tolist(),
        "quantiles": quantiles.tolist(),
        "_values": values.tolist(),
    }


def measure_probe(
    model: PilotMLP, image: Tensor, target: Tensor, threshold: float, boundaries: Sequence[float]
) -> dict[str, Any]:
    logits, preactivations, activations = model.forward_with_preactivations(image.unsqueeze(0))
    loss = F.cross_entropy(logits, target.reshape(1), reduction="sum")
    logit_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0][0]
    post_gradients = torch.autograd.grad(loss, activations, retain_graph=True)
    pre_gradients = torch.autograd.grad(loss, preactivations)
    post_rms = torch.stack([gradient.square().mean().sqrt() for gradient in post_gradients])
    pre_rms = torch.stack([gradient.square().mean().sqrt() for gradient in pre_gradients])
    tensors = [logits, logit_gradient, post_rms, pre_rms, *preactivations]
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise FloatingPointError("Non-finite probe measurement")
    return {
        "loss": float(loss.item()),
        "predicted_class": int(logits.argmax(dim=1).item()),
        "target_class": int(target.item()),
        "logit_loss_gradient": logit_gradient.detach().cpu().tolist(),
        "logit_loss_gradient_norm": float(torch.linalg.vector_norm(logit_gradient).item()),
        "pre_activation_gradient_profile": gradient_profile_metrics(pre_rms),
        "post_activation_gradient_profile": gradient_profile_metrics(post_rms),
        "saturation_by_layer": [
            saturation_metrics(value, threshold, boundaries) for value in preactivations
        ],
    }


def aggregate_probes(rows: list[dict[str, Any]], depth: int, threshold: float, boundaries: Sequence[float]) -> dict[str, Any]:
    layer_summaries = []
    for layer_index in range(depth):
        pre_rms = [row["pre_activation_gradient_profile"]["gradient_rms"][layer_index] for row in rows]
        post_rms = [row["post_activation_gradient_profile"]["gradient_rms"][layer_index] for row in rows]
        values = torch.tensor(
            [value for row in rows for value in row["saturation_by_layer"][layer_index].pop("_values")],
            dtype=torch.float32,
        )
        saturation = saturation_metrics(values, threshold, boundaries)
        saturation.pop("_values")
        layer_summaries.append(
            {
                "hidden_layer_index": layer_index,
                "mean_pre_activation_gradient_rms": fmean(pre_rms),
                "median_pre_activation_gradient_rms": median(pre_rms),
                "mean_post_activation_gradient_rms": fmean(post_rms),
                "median_post_activation_gradient_rms": median(post_rms),
                **saturation,
            }
        )

    gap_summaries = {}
    for location in ("pre_activation", "post_activation"):
        profiles = [row[f"{location}_gradient_profile"] for row in rows]
        gaps = [profile["raw_early_to_last_gap"] for profile in profiles]
        normalized = [profile["normalized_gap_per_interval"] for profile in profiles]
        slopes = [profile["layer_index_slope"] for profile in profiles]
        first_mean = fmean(profile["gradient_rms"][0] for profile in profiles)
        last_mean = fmean(profile["gradient_rms"][-1] for profile in profiles)
        gap_summaries[location] = {
            "mean_per_probe_raw_gap": fmean(gaps),
            "median_per_probe_raw_gap": median(gaps),
            "mean_per_probe_normalized_gap": fmean(normalized),
            "median_per_probe_normalized_gap": median(normalized),
            "mean_per_probe_layer_slope": fmean(slopes),
            "median_per_probe_layer_slope": median(slopes),
            "gap_from_arithmetic_mean_layer_rms": math.log10(last_mean / first_mean),
            "first_layer_arithmetic_mean_rms": first_mean,
            "last_layer_arithmetic_mean_rms": last_mean,
        }
    return {"layer_summaries": layer_summaries, "gap_summaries": gap_summaries}


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "hidden_layers": [3, 6, 12, 24], "seeds": [42, 123, 2026],
        "checkpoints": [0, 25, 100, 300, 600],
    }
    if config["model"]["hidden_layers"] != expected["hidden_layers"]:
        raise ValueError("Depth schedule differs from frozen v2 protocol")
    if config["training"]["seeds"] != expected["seeds"]:
        raise ValueError("Seed schedule differs from frozen v2 protocol")
    if config["training"]["checkpoints"] != expected["checkpoints"]:
        raise ValueError("Checkpoint schedule differs from frozen v2 protocol")
    required = {
        "runtime_budget_seconds": 1200, "train_count": 10000, "validation_count": 2000,
        "width": 128, "output_dim": 10, "learning_rate": 0.01, "momentum": 0.0,
        "batch_size": 64, "max_steps": 600,
    }
    observed = {
        "runtime_budget_seconds": config["runtime_budget_seconds"],
        "train_count": config["dataset"]["train_count"],
        "validation_count": config["dataset"]["validation_count"],
        "width": config["model"]["width"], "output_dim": config["model"]["output_dim"],
        "learning_rate": config["training"]["learning_rate"],
        "momentum": config["training"]["momentum"],
        "batch_size": config["training"]["batch_size"], "max_steps": config["training"]["max_steps"],
    }
    if observed != required:
        raise ValueError(f"Frozen v2 configuration mismatch: {observed}")


def save_checkpoint_exclusive(path: Path, model: PilotMLP) -> dict[str, Any]:
    with path.open("xb") as handle:
        torch.save(model.state_dict(), handle)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_csvs(run_dir: Path, results: Sequence[dict[str, Any]]) -> None:
    evaluation_rows, layer_rows, profile_rows = [], [], []
    for run in results:
        base = {"arm_id": run["arm_id"], "depth": run["hidden_layers"], "seed": run["seed"]}
        for checkpoint in run["checkpoints"]:
            for split in ("training", "validation"):
                evaluation_rows.append({**base, "step": checkpoint["step"], "split": split, **checkpoint[split]})
            for layer in checkpoint["summary"]["layer_summaries"]:
                layer_rows.append({
                    **base, "step": checkpoint["step"],
                    **{key: value for key, value in layer.items() if key not in ("histogram_counts", "quantiles")},
                    "preactivation_histogram_counts_json": json.dumps(layer["histogram_counts"]),
                    "preactivation_quantiles_json": json.dumps(layer["quantiles"]),
                })
            for row in checkpoint["probe_rows"]:
                for location in ("pre_activation", "post_activation"):
                    profile = row[f"{location}_gradient_profile"]
                    profile_rows.append({
                        **base, "step": checkpoint["step"], "dataset_index": row["dataset_index"],
                        "gradient_location": location,
                        "raw_gap": profile["raw_early_to_last_gap"],
                        "normalized_gap": profile["normalized_gap_per_interval"],
                        "layer_slope": profile["layer_index_slope"],
                        "logit_loss_gradient_norm": row["logit_loss_gradient_norm"],
                    })
    for name, rows in (
        ("evaluation_metrics.csv", evaluation_rows),
        ("layer_summary_metrics.csv", layer_rows),
        ("probe_profile_metrics.csv", profile_rows),
    ):
        with (run_dir / name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_plots(run_dir: Path, results: Sequence[dict[str, Any]]) -> None:
    def line_plot(filename: str, title: str, value_fn: Any) -> None:
        series = []
        for run in results:
            series.append((f'D{run["hidden_layers"]}/s{run["seed"]}', [(cp["step"], value_fn(cp)) for cp in run["checkpoints"]]))
        values = [value for _, points in series for _, value in points]
        low, high = min(values), max(values)
        if high == low:
            high = low + 1.0
        colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="450" y="24" text-anchor="middle" font-size="16">{title}</text>']
        for index, (label, points) in enumerate(series):
            coords = []
            for step, value in points:
                x = 60 + step / 600 * 650
                y = 450 - (value - low) / (high - low) * 390
                coords.append(f"{x:.1f},{y:.1f}")
            color = colors[index % len(colors)]
            parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<text x="730" y="{45 + index * 18}" font-size="11" fill="{color}">{label}</text>')
        parts.append('</svg>')
        (run_dir / filename).write_text("".join(parts), encoding="utf-8")

    line_plot("validation_loss.svg", "Validation loss by checkpoint", lambda cp: cp["validation"]["loss"])
    line_plot(
        "post_activation_gap.svg", "Mean per-probe post-activation log10 gap",
        lambda cp: cp["summary"]["gap_summaries"]["post_activation"]["mean_per_probe_raw_gap"],
    )


def selected_layers(depth: int) -> list[int]:
    return sorted({0, (depth - 1) // 2, depth - 1})


def measure_jacobian(
    model: PilotMLP, image: Tensor, target: Tensor, layer_indices: Sequence[int], threshold: float
) -> list[dict[str, Any]]:
    logits, _, activations = model.forward_with_preactivations(image.unsqueeze(0))
    loss = F.cross_entropy(logits, target.reshape(1), reduction="sum")
    logit_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0][0].detach()
    rows = []
    for layer_index in layer_indices:
        hidden = activations[layer_index][0].detach().clone().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(
            lambda value: model.from_hidden(layer_index, value), hidden
        )
        rows.append({
            "hidden_layer_index": layer_index,
            "directional_gain": normalized_directional_gain(jacobian, logit_gradient, threshold),
            "spectrum": jacobian_spectrum_metrics(jacobian.detach()),
        })
    return rows


def run(config_path: Path, run_id: str) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    run_dir = Path(config["results_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir()
    log_path = run_dir / "run.log"
    started = time.perf_counter()
    budget = float(config["runtime_budget_seconds"])

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"start run_id={run_id} budget_seconds={budget}")
    device = torch.device(config["device"])
    dataset = FashionMNIST(config["data_root"], train=True, download=False, transform=ToTensor())
    train_indices, validation_indices = stratified_split(
        dataset.targets, config["dataset"]["train_count"], config["dataset"]["validation_count"],
        config["dataset"]["subset_seed"],
    )
    probe_indices = validation_indices[: config["measurement"]["fixed_probe_count"]]
    batch_orders = {
        str(seed): deterministic_batch_positions(len(train_indices), config["training"]["batch_size"], config["training"]["max_steps"], seed)
        for seed in config["training"]["seeds"]
    }
    subset_manifest = {
        "dataset": "FashionMNIST", "source_split": "official_train",
        "torchvision_version": torchvision_version, "selection": "stratified_without_replacement",
        "selection_seed": config["dataset"]["subset_seed"], "train_indices": train_indices,
        "validation_indices": validation_indices, "fixed_probe_indices": probe_indices,
        "official_test_used": False,
        "batch_order": {
            seed: [[train_indices[position] for position in batch] for batch in batches]
            for seed, batches in batch_orders.items()
        },
    }
    write_json_exclusive(run_dir / "data_and_batch_manifest.json", subset_manifest)
    train_images, train_targets = cache_images(dataset, train_indices)
    validation_images, validation_targets = cache_images(dataset, validation_indices)
    probe_positions = list(range(len(probe_indices)))
    threshold = config["measurement"]["saturation_absolute_threshold"]
    boundaries = config["measurement"]["preactivation_histogram_boundaries"]
    results: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []

    for depth in config["model"]["hidden_layers"]:
        for seed in config["training"]["seeds"]:
            if time.perf_counter() - started >= budget:
                raise TimeoutError("Runtime budget exhausted before all primary runs completed")
            torch.manual_seed(seed)
            model = PilotMLP(784, 128, depth, 10, 1.0).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
            arm_id = f"primary_d{depth}_seed_{seed}"
            checkpoints: list[dict[str, Any]] = []

            def capture(step: int) -> None:
                model.eval()
                checkpoint_name = f"{arm_id}_step_{step}.pt"
                checkpoint_record = save_checkpoint_exclusive(checkpoints_dir / checkpoint_name, model)
                checkpoint_record.update({"arm_id": arm_id, "depth": depth, "seed": seed, "step": step})
                checkpoint_manifest.append(checkpoint_record)
                probe_rows = []
                for position in probe_positions:
                    row = measure_probe(
                        model, validation_images[position].to(device), validation_targets[position].to(device),
                        threshold, boundaries,
                    )
                    row["dataset_index"] = probe_indices[position]
                    probe_rows.append(row)
                checkpoints.append({
                    "step": step,
                    "training": evaluate(model, train_images, train_targets, config["training"]["evaluation_batch_size"], device),
                    "validation": evaluate(model, validation_images, validation_targets, config["training"]["evaluation_batch_size"], device),
                    "probe_rows": probe_rows,
                    "summary": aggregate_probes(probe_rows, depth, threshold, boundaries),
                    "checkpoint_file": checkpoint_name,
                })
                log(f"checkpoint arm={arm_id} step={step}")

            capture(0)
            checkpoint_set = set(config["training"]["checkpoints"][1:])
            model.train()
            for step, positions in enumerate(batch_orders[str(seed)], start=1):
                if time.perf_counter() - started >= budget:
                    raise TimeoutError(f"Runtime budget exhausted in {arm_id} at step {step}")
                batch = torch.tensor(positions, dtype=torch.long)
                logits, _ = model(train_images[batch].to(device))
                loss = F.cross_entropy(logits, train_targets[batch].to(device), reduction="mean")
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite training loss in {arm_id} at step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step in checkpoint_set:
                    capture(step)
                    model.train()
            run_result = {"arm_id": arm_id, "hidden_layers": depth, "seed": seed, "checkpoints": checkpoints}
            results.append(run_result)
            write_json_exclusive(run_dir / f"{arm_id}.json", run_result)
            log(f"completed arm={arm_id}")

    write_json_exclusive(run_dir / "checkpoint_manifest.json", {"checkpoints": checkpoint_manifest})
    write_csvs(run_dir, results)
    write_plots(run_dir, results)
    log("all_12_primary_runs_valid_and_preserved")

    jacobian_results = []
    jacobian_config = config["jacobian"]
    for depth in jacobian_config["depths"]:
        for seed in config["training"]["seeds"]:
            for step in jacobian_config["checkpoints"]:
                if time.perf_counter() - started >= budget:
                    log("jacobian_skipped_remaining_runtime_budget")
                    break
                model = PilotMLP(784, 128, depth, 10, 1.0).to(device)
                path = checkpoints_dir / f"primary_d{depth}_seed_{seed}_step_{step}.pt"
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                model.eval()
                rows = []
                excluded = 0
                for position in range(jacobian_config["probe_count"]):
                    diagnostics = measure_jacobian(
                        model, validation_images[position].to(device), validation_targets[position].to(device),
                        selected_layers(depth), jacobian_config["directional_gain_denominator_threshold"],
                    )
                    excluded += sum(int(item["directional_gain"]["excluded_near_zero_denominator"]) for item in diagnostics)
                    rows.append({"dataset_index": probe_indices[position], "jacobian_diagnostics": diagnostics})
                jacobian_results.append({
                    "arm_id": f"primary_d{depth}_seed_{seed}", "depth": depth, "seed": seed,
                    "step": step, "probe_count": len(rows), "directional_gain_excluded_count": excluded,
                    "rows": rows,
                })
                log(f"jacobian depth={depth} seed={seed} step={step}")
            else:
                continue
            break
        else:
            continue
        break
    if jacobian_results:
        write_json_exclusive(run_dir / "supplementary_jacobian.json", {"classification": "supplementary", "results": jacobian_results})

    raw = {
        "run_id": run_id, "run_type": "bounded_main_experiment", "protocol_version": "v2",
        "claim_status": "unverified", "scientific_conclusions": None, "config": config,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision_version, "device": str(device), "git_revision": git_revision(),
        },
        "primary_results": results, "jacobian_results": jacobian_results,
        "runtime_seconds": time.perf_counter() - started, "errors": [],
    }
    write_json_exclusive(run_dir / "raw_results.json", raw)
    log(f"complete runtime_seconds={raw['runtime_seconds']:.6f}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.run_id)


if __name__ == "__main__":
    main()
