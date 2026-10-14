from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import __version__ as torchvision_version
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))
from vanishing_gradient.experiment_v2 import cache_images, evaluate  # noqa: E402
from vanishing_gradient.pilot import gradient_profile_metrics  # noqa: E402


class FollowupMLP(nn.Module):
    def __init__(self, depth: int, activation: str, xavier_gain: float) -> None:
        super().__init__()
        dimensions = [784] + [128] * depth
        self.hidden = nn.ModuleList(nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(depth))
        self.output = nn.Linear(128, 10)
        self.activation_name = activation
        for layer in [*self.hidden, self.output]:
            nn.init.xavier_uniform_(layer.weight, gain=xavier_gain)
            nn.init.zeros_(layer.bias)

    def activate(self, value: Tensor) -> Tensor:
        return torch.tanh(value) if self.activation_name == "tanh" else torch.sigmoid(value)

    def forward_with_preactivations(self, inputs: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        value = inputs.flatten(start_dim=1)
        preactivations, activations = [], []
        for layer in self.hidden:
            value = layer(value)
            preactivations.append(value)
            value = self.activate(value)
            activations.append(value)
        return self.output(value), preactivations, activations

    def forward(self, inputs: Tensor) -> tuple[Tensor, list[Tensor]]:
        logits, _, activations = self.forward_with_preactivations(inputs)
        return logits, activations


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


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


def relative_norm(numerator: Tensor, denominator: Tensor, threshold: float) -> tuple[float | None, bool]:
    denominator_norm = float(torch.linalg.vector_norm(denominator).item())
    excluded = denominator_norm < threshold
    value = None if excluded else float((torch.linalg.vector_norm(numerator) / denominator_norm).item())
    return value, excluded


def parameter_gradient_rows(
    model: FollowupMLP, images: Tensor, targets: Tensor, learning_rate: float, threshold: float
) -> tuple[list[dict[str, Any]], float]:
    model.eval()
    model.zero_grad(set_to_none=True)
    logits, _ = model(images)
    loss = F.cross_entropy(logits, targets, reduction="mean")
    loss.backward()
    rows = []
    layers = [("hidden", i, layer) for i, layer in enumerate(model.hidden)] + [("output", len(model.hidden), model.output)]
    for layer_type, layer_index, layer in layers:
        if layer.weight.grad is None:
            raise RuntimeError("Missing weight gradient")
        gradient_norm = float(torch.linalg.vector_norm(layer.weight.grad).item())
        weight_norm = float(torch.linalg.vector_norm(layer.weight).item())
        relative, excluded = relative_norm(layer.weight.grad, layer.weight, threshold)
        row = {
            "layer_type": layer_type,
            "layer_index": layer_index,
            "weight_norm": weight_norm,
            "parameter_gradient_norm": gradient_norm,
            "relative_parameter_gradient_norm": relative,
            "relative_one_step_update_estimate": None if relative is None else learning_rate * relative,
            "denominator_threshold": threshold,
            "excluded_near_zero_denominator": excluded,
        }
        if not all(math.isfinite(v) for v in (gradient_norm, weight_norm)):
            raise FloatingPointError("Non-finite parameter metric")
        rows.append(row)
    return rows, float(loss.item())


def parameter_displacement_rows(
    previous: dict[str, Tensor], current: dict[str, Tensor], depth: int, threshold: float
) -> list[dict[str, Any]]:
    rows = []
    for layer_type, layer_index, name in [
        *(("hidden", i, f"hidden.{i}.weight") for i in range(depth)),
        ("output", depth, "output.weight"),
    ]:
        relative, excluded = relative_norm(current[name] - previous[name], previous[name], threshold)
        rows.append({
            "layer_type": layer_type,
            "layer_index": layer_index,
            "net_parameter_displacement": relative,
            "denominator_threshold": threshold,
            "excluded_near_zero_denominator": excluded,
        })
    return rows


def measure_probe(model: FollowupMLP, image: Tensor, target: Tensor) -> dict[str, Any]:
    logits, preactivations, activations = model.forward_with_preactivations(image.unsqueeze(0))
    loss = F.cross_entropy(logits, target.reshape(1), reduction="sum")
    post_gradients = torch.autograd.grad(loss, activations, retain_graph=True)
    pre_gradients = torch.autograd.grad(loss, preactivations)
    post_rms = torch.stack([gradient.square().mean().sqrt() for gradient in post_gradients])
    pre_rms = torch.stack([gradient.square().mean().sqrt() for gradient in pre_gradients])
    if not all(bool(torch.isfinite(value).all()) for value in [logits, post_rms, pre_rms, *preactivations]):
        raise FloatingPointError("Non-finite probe measurement")
    return {
        "loss": float(loss.item()),
        "target_class": int(target.item()),
        "pre_activation_gradient_profile": gradient_profile_metrics(pre_rms),
        "post_activation_gradient_profile": gradient_profile_metrics(post_rms),
    }


def aggregate_probe_rows(rows: list[dict[str, Any]], depth: int) -> dict[str, Any]:
    layers = []
    for index in range(depth):
        pre = [row["pre_activation_gradient_profile"]["gradient_rms"][index] for row in rows]
        post = [row["post_activation_gradient_profile"]["gradient_rms"][index] for row in rows]
        layers.append({
            "hidden_layer_index": index,
            "mean_pre_activation_gradient_rms": fmean(pre),
            "median_pre_activation_gradient_rms": median(pre),
            "mean_post_activation_gradient_rms": fmean(post),
            "median_post_activation_gradient_rms": median(post),
        })
    gaps = {}
    for location in ("pre_activation", "post_activation"):
        profiles = [row[f"{location}_gradient_profile"] for row in rows]
        gaps[location] = {
            "mean_per_probe_raw_gap": fmean(p["raw_early_to_last_gap"] for p in profiles),
            "median_per_probe_raw_gap": median(p["raw_early_to_last_gap"] for p in profiles),
            "mean_per_probe_normalized_gap": fmean(p["normalized_gap_per_interval"] for p in profiles),
            "mean_per_probe_layer_slope": fmean(p["layer_index_slope"] for p in profiles),
        }
    return {"layer_summaries": layers, "gap_summaries": gaps}


def scatter_svg(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    points = [(float(row[x_key]), float(row[y_key])) for row in rows if row[x_key] is not None and row[y_key] is not None]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xspan, yspan = max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12)
    circles = "".join(
        f'<circle cx="{60 + 700 * (x - xmin) / xspan:.2f}" cy="{360 - 300 * (y - ymin) / yspan:.2f}" r="3" fill="#2457a7" opacity="0.7"/>'
        for x, y in points
    )
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="820" height="420"><rect width="100%" height="100%" fill="white"/><text x="20" y="25" font-size="16">{title}</text><line x1="60" y1="360" x2="760" y2="360" stroke="black"/><line x1="60" y1="60" x2="60" y2="360" stroke="black"/>{circles}<text x="340" y="405" font-size="12">{x_key}</text><text x="8" y="210" font-size="12">{y_key}</text></svg>'
    with path.open("x", encoding="utf-8") as handle:
        handle.write(svg)


def grouped_means(rows: list[dict[str, Any]], keys: Sequence[str], fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group_key, items in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summary = dict(zip(keys, group_key))
        summary["seed_count"] = len({item.get("seed") for item in items if "seed" in item})
        for field in fields:
            values = [item[field] for item in items if item.get(field) is not None]
            summary[f"mean_{field}"] = fmean(values) if values else None
        output.append(summary)
    return output


def summarize_existing(output_dir: Path) -> None:
    raw = json.loads((output_dir / "raw_results.json").read_text(encoding="utf-8"))
    parameter_fields = ("weight_norm", "parameter_gradient_norm", "relative_parameter_gradient_norm", "relative_one_step_update_estimate")
    parameter_summary = grouped_means(raw["part_a"]["parameter_rows"], ("depth", "step", "layer_type", "layer_index"), parameter_fields)
    displacement_summary = grouped_means(raw["part_a"]["displacement_rows"], ("depth", "from_step", "to_step", "layer_type", "layer_index"), ("net_parameter_displacement",))
    combined_fields = (
        "validation_loss", "validation_accuracy", "first_post_activation_gradient_rms", "last_post_activation_gradient_rms",
        "post_activation_raw_gap", "post_activation_normalized_gap", "post_activation_layer_slope",
        "first_parameter_gradient_norm", "last_parameter_gradient_norm", "first_relative_one_step_update_estimate",
        "last_relative_one_step_update_estimate", "first_net_parameter_displacement", "last_net_parameter_displacement",
    )
    combined_summary = grouped_means(raw["part_b"]["rows"], ("depth", "step"), combined_fields)
    followup_rows = []
    for result in raw["part_c"]["results"]:
        for checkpoint in result["checkpoints"]:
            gap = checkpoint["summary"]["gap_summaries"]["post_activation"]
            layers = checkpoint["summary"]["layer_summaries"]
            followup_rows.append({
                "arm_id": result["arm_id"], "activation": result["activation"], "xavier_gain": result["xavier_gain"],
                "seed": result["seed"], "step": checkpoint["step"], "validation_loss": checkpoint["validation"]["loss"],
                "validation_accuracy": checkpoint["validation"]["accuracy"], "first_post_activation_gradient_rms": layers[0]["mean_post_activation_gradient_rms"],
                "last_post_activation_gradient_rms": layers[-1]["mean_post_activation_gradient_rms"],
                "post_activation_raw_gap": gap["mean_per_probe_raw_gap"], "post_activation_normalized_gap": gap["mean_per_probe_normalized_gap"],
                "post_activation_layer_slope": gap["mean_per_probe_layer_slope"],
            })
    condition_fields = ("validation_loss", "validation_accuracy", "first_post_activation_gradient_rms", "last_post_activation_gradient_rms", "post_activation_raw_gap", "post_activation_normalized_gap", "post_activation_layer_slope")
    condition_summary = grouped_means(followup_rows, ("arm_id", "activation", "xavier_gain", "step"), condition_fields)
    baseline = {row["step"]: row for row in combined_summary if row["depth"] == 12}
    comparison = []
    for row in condition_summary:
        base = baseline[row["step"]]
        item = {"arm_id": row["arm_id"], "activation": row["activation"], "xavier_gain": row["xavier_gain"], "step": row["step"], "seed_count": row["seed_count"], "baseline": "v2_d12_tanh_gain_1_0"}
        for field in condition_fields:
            condition_value = row[f"mean_{field}"]
            baseline_value = base[f"mean_{field}"]
            item[f"baseline_mean_{field}"] = baseline_value
            item[f"condition_mean_{field}"] = condition_value
            item[f"condition_minus_baseline_{field}"] = condition_value - baseline_value
        comparison.append(item)
    write_csv_exclusive(output_dir / "parameter_gradient_summary.csv", parameter_summary)
    write_csv_exclusive(output_dir / "parameter_displacement_summary.csv", displacement_summary)
    write_csv_exclusive(output_dir / "combined_learning_gradient_summary.csv", combined_summary)
    write_csv_exclusive(output_dir / "followup_condition_summary.csv", condition_summary)
    write_csv_exclusive(output_dir / "followup_vs_baseline_d12.csv", comparison)
    scatter_svg(output_dir / "followup_accuracy_vs_gap.svg", followup_rows, "validation_accuracy", "post_activation_raw_gap", "Follow-up conditions: accuracy and post-activation gap")
    write_json_exclusive(output_dir / "summary_audit.json", {
        "status": "PASS", "source": "raw_results.json", "parameter_summary_rows": len(parameter_summary),
        "displacement_summary_rows": len(displacement_summary), "combined_summary_rows": len(combined_summary),
        "condition_summary_rows": len(condition_summary), "baseline_comparison_rows": len(comparison),
        "aggregation": "arithmetic mean across the three approved seeds; no threshold or significance test",
    })


def validate_inputs(config: dict[str, Any], protocol: dict[str, Any]) -> None:
    if protocol["status"] != "FROZEN_BEFORE_RUN" or protocol["classification"] != "follow_up_experiment_designed_after_reviewing_v2":
        raise ValueError("Follow-up protocol is not frozen or correctly classified")
    expected = {"runtime_budget_seconds": 600, "learning_rate": 0.01, "batch_size": 64, "checkpoints": [0, 25, 100, 300, 600], "seeds": [42, 123, 2026]}
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("Follow-up configuration differs from the frozen protocol")
    expected_arms = [("tanh", 0.5), ("sigmoid", 1.0)]
    if [(arm["activation"], arm["xavier_gain"]) for arm in config["followup_arms"]] != expected_arms:
        raise ValueError("Follow-up arms differ from the frozen protocol")


def run(config_path: Path, run_id: str) -> Path:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol_path = PROJECT_ROOT / config["protocol"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_inputs(config, protocol)
    output_dir = PROJECT_ROOT / config["results_root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    log_path = output_dir / "run.log"
    budget = float(config["runtime_budget_seconds"])

    def check_budget() -> None:
        if time.perf_counter() - started >= budget:
            raise TimeoutError("Follow-up runtime budget exhausted")

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

    source_v2 = PROJECT_ROOT / config["source_v2_run"]
    source_posthoc = PROJECT_ROOT / config["source_posthoc_run"]
    raw_v2 = json.loads((source_v2 / "raw_results.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((source_v2 / "data_and_batch_manifest.json").read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads((source_v2 / "checkpoint_manifest.json").read_text(encoding="utf-8"))["checkpoints"]
    posthoc_raw = json.loads((source_posthoc / "raw_posthoc_results.json").read_text(encoding="utf-8"))
    probe_manifest = json.loads((source_posthoc / "probe_manifest.json").read_text(encoding="utf-8"))
    if len(checkpoint_manifest) != 60 or len(probe_manifest["gradient_probe_indices"]) != 40:
        raise ValueError("Expected 60 V2 checkpoints and 40 balanced probes")
    recorded = {row["path"]: row for row in checkpoint_manifest}
    protected = [source_v2 / "raw_results.json", source_v2 / "data_and_batch_manifest.json", source_v2 / "checkpoint_manifest.json", source_posthoc / "raw_posthoc_results.json", source_posthoc / "probe_manifest.json"]
    protected += [source_v2 / "checkpoints" / row["path"] for row in checkpoint_manifest]
    before_hashes = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in protected}

    dataset = FashionMNIST(PROJECT_ROOT / config["data_root"], train=True, download=False, transform=ToTensor())
    probe_indices = probe_manifest["gradient_probe_indices"]
    probe_images, probe_targets = cache_images(dataset, probe_indices)
    device = torch.device(config["device"])
    probe_images, probe_targets = probe_images.to(device), probe_targets.to(device)
    threshold = float(config["denominator_threshold"])
    learning_rate = float(config["learning_rate"])

    parameter_rows, displacement_rows = [], []
    checkpoint_states: dict[tuple[int, int, int], dict[str, Tensor]] = {}
    log("part_a_start")
    for run_result in raw_v2["primary_results"]:
        depth, seed = run_result["hidden_layers"], run_result["seed"]
        previous_step, previous_state = None, None
        for checkpoint in run_result["checkpoints"]:
            check_budget()
            step = checkpoint["step"]
            filename = checkpoint["checkpoint_file"]
            path = source_v2 / "checkpoints" / filename
            if filename not in recorded or sha256(path) != recorded[filename]["sha256"]:
                raise ValueError(f"Checkpoint provenance failure: {filename}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint_states[(depth, seed, step)] = state
            model = FollowupMLP(depth, "tanh", 1.0).to(device)
            model.load_state_dict(state)
            rows, batch_loss = parameter_gradient_rows(model, probe_images, probe_targets, learning_rate, threshold)
            for row in rows:
                row.update({"depth": depth, "seed": seed, "step": step, "checkpoint_file": filename, "fixed_batch_loss": batch_loss, "fixed_batch_size": 40})
                parameter_rows.append(row)
            if previous_state is not None:
                for row in parameter_displacement_rows(previous_state, state, depth, threshold):
                    row.update({"depth": depth, "seed": seed, "from_step": previous_step, "to_step": step})
                    displacement_rows.append(row)
            previous_step, previous_state = step, state
    log("part_a_complete")

    posthoc_by_key = {(row["depth"], row["seed"], row["step"]): row for row in posthoc_raw["gradient_results"]}
    parameter_by_key = {(row["depth"], row["seed"], row["step"], row["layer_type"], row["layer_index"]): row for row in parameter_rows}
    displacement_by_key = {(row["depth"], row["seed"], row["to_step"], row["layer_type"], row["layer_index"]): row for row in displacement_rows}
    combined_rows = []
    for run_result in raw_v2["primary_results"]:
        depth, seed = run_result["hidden_layers"], run_result["seed"]
        for checkpoint in run_result["checkpoints"]:
            step = checkpoint["step"]
            balanced = posthoc_by_key[(depth, seed, step)]
            layers = balanced["summary"]["layer_summaries"]
            gap = balanced["summary"]["gap_summaries"]["post_activation"]
            first_param = parameter_by_key[(depth, seed, step, "hidden", 0)]
            last_param = parameter_by_key[(depth, seed, step, "hidden", depth - 1)]
            first_disp = displacement_by_key.get((depth, seed, step, "hidden", 0))
            last_disp = displacement_by_key.get((depth, seed, step, "hidden", depth - 1))
            combined_rows.append({
                "depth": depth, "seed": seed, "step": step,
                "validation_loss": checkpoint["validation"]["loss"], "validation_accuracy": checkpoint["validation"]["accuracy"],
                "first_post_activation_gradient_rms": layers[0]["mean_post_activation_gradient_rms"],
                "last_post_activation_gradient_rms": layers[-1]["mean_post_activation_gradient_rms"],
                "post_activation_raw_gap": gap["mean_per_probe_raw_gap"],
                "post_activation_normalized_gap": gap["mean_per_probe_normalized_gap"],
                "post_activation_layer_slope": gap["mean_per_probe_layer_slope"],
                "first_parameter_gradient_norm": first_param["parameter_gradient_norm"],
                "last_parameter_gradient_norm": last_param["parameter_gradient_norm"],
                "first_relative_parameter_gradient_norm": first_param["relative_parameter_gradient_norm"],
                "last_relative_parameter_gradient_norm": last_param["relative_parameter_gradient_norm"],
                "first_relative_one_step_update_estimate": first_param["relative_one_step_update_estimate"],
                "last_relative_one_step_update_estimate": last_param["relative_one_step_update_estimate"],
                "displacement_from_step": None if first_disp is None else first_disp["from_step"],
                "first_net_parameter_displacement": None if first_disp is None else first_disp["net_parameter_displacement"],
                "last_net_parameter_displacement": None if last_disp is None else last_disp["net_parameter_displacement"],
                "coexistence_status": "INCONCLUSIVE_NO_PREDECLARED_JOINT_THRESHOLD",
            })
    log("part_b_complete")

    train_indices, validation_indices = data_manifest["train_indices"], data_manifest["validation_indices"]
    train_images, train_targets = cache_images(dataset, train_indices)
    validation_images, validation_targets = cache_images(dataset, validation_indices)
    position_by_index = {dataset_index: position for position, dataset_index in enumerate(train_indices)}
    followup_results = []
    log("part_c_start")
    for arm in config["followup_arms"]:
        for seed in config["seeds"]:
            check_budget()
            torch.manual_seed(seed)
            model = FollowupMLP(12, arm["activation"], arm["xavier_gain"]).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
            checkpoints = []

            def capture(step: int) -> None:
                model.eval()
                name = f'{arm["arm_id"]}_seed_{seed}_step_{step}.pt'
                path = checkpoint_dir / name
                with path.open("xb") as handle:
                    torch.save(model.state_dict(), handle)
                probe_rows = []
                for dataset_index in probe_indices:
                    image, target = dataset[dataset_index]
                    row = measure_probe(model, image.to(device), torch.tensor(target, device=device))
                    row["dataset_index"] = dataset_index
                    probe_rows.append(row)
                checkpoints.append({
                    "step": step,
                    "training": evaluate(model, train_images, train_targets, config["evaluation_batch_size"], device),
                    "validation": evaluate(model, validation_images, validation_targets, config["evaluation_batch_size"], device),
                    "summary": aggregate_probe_rows(probe_rows, 12),
                    "checkpoint_file": name,
                    "checkpoint_sha256": sha256(path),
                })

            capture(0)
            checkpoint_set = set(config["checkpoints"][1:])
            for step, batch_indices in enumerate(data_manifest["batch_order"][str(seed)], start=1):
                check_budget()
                positions = torch.tensor([position_by_index[index] for index in batch_indices])
                images = train_images[positions].to(device)
                targets = train_targets[positions].to(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(images)
                loss = F.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite follow-up training loss")
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                    raise FloatingPointError("Non-finite follow-up training gradient")
                optimizer.step()
                if step in checkpoint_set:
                    capture(step)
            followup_results.append({"arm_id": arm["arm_id"], "activation": arm["activation"], "xavier_gain": arm["xavier_gain"], "seed": seed, "checkpoints": checkpoints})
            log(f'part_c_arm_complete arm={arm["arm_id"]} seed={seed}')
    log("part_c_complete")

    followup_rows = []
    for result in followup_results:
        for checkpoint in result["checkpoints"]:
            gap = checkpoint["summary"]["gap_summaries"]["post_activation"]
            layers = checkpoint["summary"]["layer_summaries"]
            followup_rows.append({
                "arm_id": result["arm_id"], "activation": result["activation"], "xavier_gain": result["xavier_gain"], "seed": result["seed"], "step": checkpoint["step"],
                "validation_loss": checkpoint["validation"]["loss"], "validation_accuracy": checkpoint["validation"]["accuracy"],
                "first_post_activation_gradient_rms": layers[0]["mean_post_activation_gradient_rms"],
                "last_post_activation_gradient_rms": layers[-1]["mean_post_activation_gradient_rms"],
                "post_activation_raw_gap": gap["mean_per_probe_raw_gap"], "post_activation_normalized_gap": gap["mean_per_probe_normalized_gap"],
                "post_activation_layer_slope": gap["mean_per_probe_layer_slope"],
            })

    after_hashes = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in protected}
    if before_hashes != after_hashes:
        raise RuntimeError("A protected V2/post-hoc source artifact changed")
    all_numeric = [value for rows in (parameter_rows, displacement_rows, combined_rows, followup_rows) for row in rows for value in row.values() if isinstance(value, float)]
    if not all(math.isfinite(value) for value in all_numeric):
        raise FloatingPointError("Non-finite exported metric")

    write_csv_exclusive(output_dir / "parameter_gradient_by_layer.csv", parameter_rows)
    write_csv_exclusive(output_dir / "parameter_displacement_by_layer.csv", displacement_rows)
    write_csv_exclusive(output_dir / "combined_learning_gradient_table.csv", combined_rows)
    write_csv_exclusive(output_dir / "followup_condition_metrics.csv", followup_rows)
    scatter_svg(output_dir / "accuracy_vs_first_gradient.svg", combined_rows, "validation_accuracy", "first_post_activation_gradient_rms", "V2 accuracy and first-layer activation gradient (no threshold)")
    scatter_svg(output_dir / "accuracy_vs_parameter_gradient.svg", combined_rows, "validation_accuracy", "first_parameter_gradient_norm", "V2 accuracy and first-layer parameter gradient (no threshold)")
    runtime = time.perf_counter() - started
    raw = {
        "run_id": run_id, "classification": protocol["classification"], "claim_status": "unverified",
        "protocol": protocol, "config": config,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision_version, "device": str(device), "git_revision": git_revision()},
        "part_a": {"status": "DONE", "parameter_rows": parameter_rows, "displacement_rows": displacement_rows},
        "part_b": {"status": "DONE", "coexistence_status": "INCONCLUSIVE_NO_PREDECLARED_JOINT_THRESHOLD", "rows": combined_rows},
        "part_c": {"status": "DONE", "results": followup_results},
        "source_hashes_before": before_hashes, "source_artifacts_unchanged": True,
        "runtime_seconds": runtime, "errors": [], "scientific_conclusions": None,
    }
    write_json_exclusive(output_dir / "raw_results.json", raw)
    audit = {
        "status": "PASS", "source_artifacts_unchanged": True, "source_checkpoint_count": 60,
        "parameter_gradient_checkpoint_count": 60, "parameter_gradient_row_count": len(parameter_rows),
        "parameter_displacement_row_count": len(displacement_rows), "combined_row_count": len(combined_rows),
        "followup_run_count": len(followup_results), "followup_checkpoint_count": len(followup_rows),
        "nonfinite_count": 0, "near_zero_parameter_denominator_exclusion_count": sum(row["excluded_near_zero_denominator"] for row in parameter_rows),
        "near_zero_displacement_denominator_exclusion_count": sum(row["excluded_near_zero_denominator"] for row in displacement_rows),
        "official_test_used": False, "runtime_seconds": runtime,
    }
    write_json_exclusive(output_dir / "audit.json", audit)
    with (output_dir / "REPRODUCE.md").open("x", encoding="utf-8") as handle:
        handle.write(f"# Reproduce {run_id}\n\nThis is a post-V2 follow-up, not preregistered V2 evidence. Fashion-MNIST is not packaged.\n\n```powershell\n.venv\\Scripts\\python.exe -m unittest discover -s tests -v\n.venv\\Scripts\\python.exe scripts\\followup_evidence.py --config configs\\followup_evidence.json --run-id followup-evidence-NEW-UNIQUE-ID\n```\n\nRequires the original V2 run, its 60 checkpoints, and the balanced-probe post-hoc artifacts at the paths in the config.\n")
    log(f"complete runtime_seconds={runtime:.6f}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--summarize-dir", type=Path)
    args = parser.parse_args()
    if args.summarize_dir:
        summarize_existing(args.summarize_dir.resolve())
        print(args.summarize_dir.resolve())
    elif args.config and args.run_id:
        print(run(args.config.resolve(), args.run_id))
    else:
        parser.error("provide --summarize-dir or both --config and --run-id")


if __name__ == "__main__":
    main()
