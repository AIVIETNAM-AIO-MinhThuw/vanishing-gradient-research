from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PilotMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int,
        hidden_layers: int,
        output_dim: int,
        xavier_gain: float,
    ) -> None:
        super().__init__()
        dimensions = [input_dim] + [width] * hidden_layers
        self.hidden = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(hidden_layers)
        )
        self.output = nn.Linear(width, output_dim)
        for layer in [*self.hidden, self.output]:
            nn.init.xavier_uniform_(layer.weight, gain=xavier_gain)
            nn.init.zeros_(layer.bias)

    def forward_with_preactivations(
        self, inputs: Tensor
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        value = inputs.flatten(start_dim=1)
        preactivations: list[Tensor] = []
        activations: list[Tensor] = []
        for layer in self.hidden:
            preactivation = layer(value)
            value = torch.tanh(preactivation)
            preactivations.append(preactivation)
            activations.append(value)
        return self.output(value), preactivations, activations

    def forward(self, inputs: Tensor) -> tuple[Tensor, list[Tensor]]:
        logits, _, activations = self.forward_with_preactivations(inputs)
        return logits, activations

    def from_hidden(self, hidden_layer_index: int, activation: Tensor) -> Tensor:
        value = activation
        for layer in self.hidden[hidden_layer_index + 1 :]:
            value = torch.tanh(layer(value))
        return self.output(value)


def gradient_profile_metrics(layerwise_gradient_rms: Tensor) -> dict[str, Any]:
    if layerwise_gradient_rms.ndim != 1 or layerwise_gradient_rms.numel() < 2:
        raise ValueError("At least two one-dimensional layerwise gradient values are required")
    if not bool(torch.isfinite(layerwise_gradient_rms).all()) or bool((layerwise_gradient_rms <= 0).any()):
        raise ValueError("Layerwise gradient RMS values must be finite and strictly positive")

    log_gradient = torch.log10(layerwise_gradient_rms)
    layer_index = torch.arange(
        layerwise_gradient_rms.numel(), device=log_gradient.device, dtype=log_gradient.dtype
    )
    centered_index = layer_index - layer_index.mean()
    slope = ((centered_index * (log_gradient - log_gradient.mean())).sum() / centered_index.square().sum())
    raw_gap = log_gradient[-1] - log_gradient[0]
    return {
        "layer_indices": layer_index.detach().cpu().tolist(),
        "gradient_rms": layerwise_gradient_rms.detach().cpu().tolist(),
        "log10_gradient_rms": log_gradient.detach().cpu().tolist(),
        "raw_early_to_last_gap": float(raw_gap.item()),
        "normalized_gap_per_interval": float((raw_gap / (layerwise_gradient_rms.numel() - 1)).item()),
        "layer_index_slope": float(slope.item()),
        "slope_interpretation": "rate_per_hidden_layer_interval_not_total_accumulated_attenuation",
    }


def normalized_directional_gain(
    jacobian: Tensor, logit_loss_gradient: Tensor, denominator_threshold: float = 1e-6
) -> dict[str, Any]:
    if jacobian.ndim != 2 or logit_loss_gradient.ndim != 1:
        raise ValueError("Expected a matrix Jacobian and a vector logit-loss gradient")
    if jacobian.shape[0] != logit_loss_gradient.numel():
        raise ValueError("Jacobian output dimension must match the logit-loss gradient")
    denominator = torch.linalg.vector_norm(logit_loss_gradient)
    numerator = torch.linalg.vector_norm(jacobian.transpose(0, 1) @ logit_loss_gradient)
    excluded = float(denominator.item()) < denominator_threshold
    return {
        "value": None if excluded else float((numerator / denominator).item()),
        "numerator_norm": float(numerator.item()),
        "denominator_norm": float(denominator.item()),
        "denominator_threshold": denominator_threshold,
        "excluded_near_zero_denominator": excluded,
        "interpretation_limit": "J^T g is the activation gradient, so this gain is not independent of activation-gradient magnitude.",
    }


def jacobian_spectrum_metrics(jacobian: Tensor) -> dict[str, Any]:
    if jacobian.ndim != 2 or not bool(torch.isfinite(jacobian).all()):
        raise ValueError("Jacobian must be a finite matrix")
    singular_values = torch.linalg.svdvals(jacobian)
    rows, columns = jacobian.shape
    sigma_max = singular_values.max() if singular_values.numel() else torch.zeros((), dtype=jacobian.dtype)
    tolerance = sigma_max * max(rows, columns) * torch.finfo(jacobian.dtype).eps
    numerical_rank = int((singular_values > tolerance).sum().item())
    energy = singular_values.square()
    if float(energy.sum().item()) == 0.0:
        effective_rank = 0.0
    else:
        probabilities = energy / energy.sum()
        nonzero_probabilities = probabilities[probabilities > 0]
        effective_rank = float(torch.exp(-(nonzero_probabilities * nonzero_probabilities.log()).sum()).item())
    if numerical_rank == 0:
        condition_number = None
    else:
        condition_number = float((singular_values[0] / singular_values[numerical_rank - 1]).item())
    full_row_rank = numerical_rank == min(rows, columns)
    return {
        "shape": [rows, columns],
        "singular_values": singular_values.detach().cpu().tolist(),
        "sigma_max": float(sigma_max.item()),
        "sigma_min": float(singular_values[-1].item()) if singular_values.numel() else 0.0,
        "frobenius_norm": float(torch.linalg.matrix_norm(jacobian, ord="fro").item()),
        "rank_tolerance": float(tolerance.item()),
        "numerical_rank": numerical_rank,
        "effective_rank_energy_entropy": effective_rank,
        "condition_number_nonzero_spectrum": condition_number,
        "full_row_rank": full_row_rank,
        "inherent_input_nullity": max(columns - min(rows, columns), 0),
        "additional_numerical_rank_deficiency": min(rows, columns) - numerical_rank,
        "interpretation_limit": "The inherent nullspace from a wide 10x128 shape is structural, not observed spectral collapse.",
    }


def stratified_indices(targets: Tensor, count: int, seed: int) -> list[int]:
    classes = torch.unique(targets, sorted=True)
    if count <= 0 or count > targets.numel():
        raise ValueError("Subset count must be positive and no larger than the dataset")
    base, remainder = divmod(count, classes.numel())
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for position, class_id in enumerate(classes.tolist()):
        class_indices = torch.nonzero(targets == class_id, as_tuple=False).flatten()
        take = base + (1 if position < remainder else 0)
        if take > class_indices.numel():
            raise ValueError(f"Class {class_id} has too few samples for the requested stratified subset")
        order = torch.randperm(class_indices.numel(), generator=generator)[:take]
        selected.extend(class_indices[order].tolist())
    return selected


def _selected_spectrum_layers(depth: int, requested: Sequence[str]) -> list[int]:
    mapping = {"first": 0, "middle": (depth - 1) // 2, "last": depth - 1}
    return sorted({mapping[name] for name in requested})


def measure_sample(
    model: PilotMLP,
    image: Tensor,
    target: Tensor,
    spectrum_layers: Sequence[int],
    denominator_threshold: float,
) -> dict[str, Any]:
    logits, preactivations, activations = model.forward_with_preactivations(image.unsqueeze(0))
    loss = F.cross_entropy(logits, target.reshape(1), reduction="sum")
    logit_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0][0]
    activation_gradients = torch.autograd.grad(loss, activations, retain_graph=True)
    preactivation_gradients = torch.autograd.grad(loss, preactivations, retain_graph=True)
    post_rms = torch.stack([gradient.square().mean().sqrt() for gradient in activation_gradients])
    pre_rms = torch.stack([gradient.square().mean().sqrt() for gradient in preactivation_gradients])
    spectra = []
    for layer_index in spectrum_layers:
        hidden = activations[layer_index][0].detach().clone().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(
            lambda value: model.from_hidden(layer_index, value), hidden
        )
        spectra.append(
            {
                "hidden_layer_index": layer_index,
                "directional_gain": normalized_directional_gain(
                    jacobian, logit_gradient.detach(), denominator_threshold
                ),
                "spectrum": jacobian_spectrum_metrics(jacobian.detach()),
            }
        )
    return {
        "loss": float(loss.item()),
        "predicted_class": int(logits.argmax(dim=1).item()),
        "target_class": int(target.item()),
        "logit_loss_gradient_norm": float(torch.linalg.vector_norm(logit_gradient).item()),
        "post_activation_gradient_profile": gradient_profile_metrics(post_rms),
        "pre_activation_gradient_profile": gradient_profile_metrics(pre_rms),
        "jacobian_diagnostics": spectra,
    }


def validate_run_config(config: dict[str, Any]) -> None:
    if config.get("runtime_budget_seconds") is None:
        raise ValueError("runtime_budget_seconds is unresolved; no pilot measurement run is authorized")
    if float(config["runtime_budget_seconds"]) <= 0:
        raise ValueError("runtime_budget_seconds must be positive")
    training = config["training"]
    dataset = config["dataset"]
    model = config["model"]
    if dataset["name"] != "FashionMNIST" or dataset["transform"] != "ToTensor":
        raise ValueError("Pilot v1 requires Fashion-MNIST with the frozen ToTensor transform")
    if (int(dataset["train_count"]), int(dataset["validation_count"])) != (1024, 256):
        raise ValueError("Pilot v1 subset sizes must remain 1024 train and 256 validation")
    if (
        int(model["input_dim"]),
        int(model["width"]),
        int(model["output_dim"]),
        model["activation"],
        model["initialization"],
    ) != (784, 128, 10, "tanh", "xavier_uniform"):
        raise ValueError("Pilot v1 model must remain 784-128-10 with Tanh and Xavier initialization")
    if training["optimizer"] != "SGD" or float(training["learning_rate"]) != 0.01:
        raise ValueError("Pilot v1 requires SGD with learning rate 0.01")
    if int(training["batch_size"]) != 64 or int(training["max_steps"]) != 100:
        raise ValueError("Pilot v1 requires batch size 64 and maximum 100 steps")
    if list(training["checkpoints"]) != [0, 100]:
        raise ValueError("Pilot v1 checkpoints must remain [0, 100]")
    if list(training["seeds"]) != [42, 123, 2026] or int(dataset["subset_seed"]) != 42:
        raise ValueError("Pilot v1 requires approved seeds [42, 123, 2026] and subset seed 42")
    arms = [(arm["hidden_layers"], float(arm["xavier_gain"])) for arm in config["primary_arms"]]
    if arms != [(3, 1.0), (12, 1.0)]:
        raise ValueError("Primary arms must remain D=3 and D=12 with Xavier gain 1.0")
    diagnostic = config["diagnostic"]
    if (int(diagnostic["hidden_layers"]), float(diagnostic["xavier_gain"])) != (12, 0.5):
        raise ValueError("The only permitted diagnostic is D=12 with Xavier gain 0.5")
    if float(config["measurement"]["directional_gain_denominator_threshold"]) != 1e-6:
        raise ValueError("Directional-gain denominator threshold must remain 1e-6")


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _git_revision(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE_NO_GIT_COMMIT"


def evaluate_model(model: PilotMLP, dataset: Any, indices: Sequence[int], device: torch.device) -> dict[str, float]:
    total_loss = 0.0
    correct = 0
    model.eval()
    with torch.no_grad():
        for dataset_index in indices:
            image, target = dataset[dataset_index]
            logits, _ = model(image.unsqueeze(0).to(device))
            total_loss += float(F.cross_entropy(logits, torch.tensor([target], device=device), reduction="sum"))
            correct += int(logits.argmax(dim=1).item() == int(target))
    return {"loss": total_loss / len(indices), "accuracy": correct / len(indices), "sample_count": len(indices)}


def write_primary_exports(run_directory: Path, primary_results: Sequence[dict[str, Any]]) -> None:
    csv_path = run_directory / "primary_metrics.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "arm_id", "hidden_layers", "seed", "step", "evaluation_loss", "evaluation_accuracy",
            "dataset_index", "gradient_location", "hidden_layer_index", "gradient_rms",
            "raw_early_to_last_gap", "normalized_gap_per_interval", "layer_index_slope",
            "logit_loss_gradient_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for arm in primary_results:
            for checkpoint in arm["checkpoints"]:
                for row in checkpoint["rows"]:
                    for location in ("pre_activation", "post_activation"):
                        profile = row[f"{location}_gradient_profile"]
                        for layer_index, gradient_rms in enumerate(profile["gradient_rms"]):
                            writer.writerow(
                                {
                                    "arm_id": arm["arm_id"],
                                    "hidden_layers": arm["hidden_layers"],
                                    "seed": arm["seed"],
                                    "step": checkpoint["step"],
                                    "evaluation_loss": checkpoint["evaluation"]["loss"],
                                    "evaluation_accuracy": checkpoint["evaluation"]["accuracy"],
                                    "dataset_index": row["dataset_index"],
                                    "gradient_location": location,
                                    "hidden_layer_index": layer_index,
                                    "gradient_rms": gradient_rms,
                                    "raw_early_to_last_gap": profile["raw_early_to_last_gap"],
                                    "normalized_gap_per_interval": profile["normalized_gap_per_interval"],
                                    "layer_index_slope": profile["layer_index_slope"],
                                    "logit_loss_gradient_norm": row["logit_loss_gradient_norm"],
                                }
                            )

    values = []
    for arm in primary_results:
        final_checkpoint = arm["checkpoints"][-1]
        mean_gap = sum(
            row["post_activation_gradient_profile"]["raw_early_to_last_gap"]
            for row in final_checkpoint["rows"]
        ) / len(final_checkpoint["rows"])
        values.append((arm["arm_id"], arm["seed"], mean_gap))
    width, height, margin = 720, 360, 50
    low = min((value for _, _, value in values), default=-1.0)
    high = max((value for _, _, value in values), default=1.0)
    if high == low:
        high = low + 1.0
    points = []
    for index, (arm_id, seed, value) in enumerate(values):
        x = margin + index * (width - 2 * margin) / max(len(values) - 1, 1)
        y = height - margin - (value - low) * (height - 2 * margin) / (high - low)
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"/>'
            f'<text x="{x:.1f}" y="{height - 18}" font-size="10" text-anchor="middle">{arm_id}/s{seed}</text>'
            f'<text x="{x:.1f}" y="{y - 9:.1f}" font-size="10" text-anchor="middle">{value:.3f}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="360" y="22" text-anchor="middle" font-size="15">Observed post-activation raw gradient gap at step 100</text>'
        + "".join(points)
        + '</svg>\n'
    )
    (run_directory / "primary_gradient_gap.svg").write_text(svg, encoding="utf-8")


def run_pilot(config: dict[str, Any], project_root: Path, run_id: str) -> Path:
    validate_run_config(config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("run_id must contain only letters, digits, dot, underscore, or hyphen")
    started = time.perf_counter()
    budget = float(config["runtime_budget_seconds"])
    results_root = project_root / config["results_root"]
    run_directory = results_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    log_path = run_directory / "run.log"

    def log(message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")

    log("pilot_start")
    try:
        from torchvision import __version__ as torchvision_version
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset

        device_name = config["device"]
        device = torch.device(
            "cuda" if device_name == "auto" and torch.cuda.is_available() else
            "cpu" if device_name == "auto" else device_name
        )
        transform = transforms.ToTensor()
        data_root = project_root / config["data_root"]
        train_dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
        validation_dataset = datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)
        dataset_config = config["dataset"]
        train_indices = stratified_indices(
            train_dataset.targets, int(dataset_config["train_count"]), int(dataset_config["subset_seed"])
        )
        validation_indices = stratified_indices(
            validation_dataset.targets,
            int(dataset_config["validation_count"]),
            int(dataset_config["subset_seed"]),
        )
        subset_manifest = {
            "dataset": "FashionMNIST",
            "torchvision_version": torchvision_version,
            "selection": "deterministic_stratified_by_class",
            "selection_seed": dataset_config["subset_seed"],
            "train_indices": train_indices,
            "validation_indices": validation_indices,
        }
        _write_json_exclusive(run_directory / "subset_indices.json", subset_manifest)
        log("subset_manifest_written_before_model_execution")

        primary_results: list[dict[str, Any]] = []
        diagnostic_results: list[dict[str, Any]] = []
        model_snapshots: list[dict[str, Any]] = []
        probe_count = int(config["measurement"]["probe_samples"])
        probe_indices = validation_indices[:probe_count]

        def execute_arm(arm: dict[str, Any], seed: int, is_diagnostic: bool) -> None:
                if time.perf_counter() - started >= budget:
                    raise TimeoutError("Approved runtime budget exhausted")
                torch.manual_seed(seed)
                model_config = config["model"]
                model = PilotMLP(
                    int(model_config["input_dim"]),
                    int(model_config["width"]),
                    int(arm["hidden_layers"]),
                    int(model_config["output_dim"]),
                    float(arm["xavier_gain"]),
                ).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=float(config["training"]["learning_rate"]))
                generator = torch.Generator().manual_seed(int(seed))
                loader = DataLoader(
                    Subset(train_dataset, train_indices),
                    batch_size=int(config["training"]["batch_size"]),
                    shuffle=True,
                    generator=generator,
                )
                checkpoints = []

                def measure_checkpoint(step: int) -> None:
                    model.eval()
                    rows = []
                    excluded_count = 0
                    for dataset_index in probe_indices:
                        image, target = validation_dataset[dataset_index]
                        row = measure_sample(
                            model,
                            image.to(device),
                            torch.as_tensor(target, device=device, dtype=torch.long),
                            [],
                            float(config["measurement"]["directional_gain_denominator_threshold"]),
                        )
                        row["dataset_index"] = dataset_index
                        rows.append(row)
                    checkpoints.append(
                        {
                            "step": step,
                            "evaluation": evaluate_model(model, validation_dataset, validation_indices, device),
                            "rows": rows,
                        }
                    )
                    model_snapshots.append(
                        {
                            "arm_id": arm["arm_id"],
                            "hidden_layers": arm["hidden_layers"],
                            "xavier_gain": arm["xavier_gain"],
                            "seed": seed,
                            "step": step,
                            "classification": "supplementary_diagnostic" if is_diagnostic else "primary_depth_contrast",
                            "state_dict": {
                                name: value.detach().cpu().clone()
                                for name, value in model.state_dict().items()
                            },
                        }
                    )

                measure_checkpoint(0)
                model.train()
                iterator = iter(loader)
                for step in range(1, int(config["training"]["max_steps"]) + 1):
                    if time.perf_counter() - started >= budget:
                        raise TimeoutError("Approved runtime budget exhausted")
                    try:
                        images, targets = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        images, targets = next(iterator)
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = model(images.to(device))
                    loss = F.cross_entropy(logits, targets.to(device), reduction="mean")
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite training loss")
                    loss.backward()
                    optimizer.step()
                measure_checkpoint(int(config["training"]["max_steps"]))
                arm_result = {
                    "arm_id": arm["arm_id"],
                    "hidden_layers": arm["hidden_layers"],
                    "xavier_gain": arm["xavier_gain"],
                    "seed": seed,
                    "classification": "supplementary_diagnostic" if is_diagnostic else "primary_depth_contrast",
                    "checkpoints": checkpoints,
                }
                destination = diagnostic_results if is_diagnostic else primary_results
                destination.append(arm_result)
                _write_json_exclusive(
                    run_directory / f"{arm['arm_id']}_seed_{seed}.json", arm_result
                )
                log(f"completed arm={arm['arm_id']} seed={seed}")

        for seed in config["training"]["seeds"]:
            for arm in config["primary_arms"]:
                execute_arm(arm, int(seed), is_diagnostic=False)
            log(f"validated_primary_pair seed={seed}")
        _write_json_exclusive(
            run_directory / "primary_manifest.json",
            {"protocol_version": "v1", "results": primary_results},
        )
        write_primary_exports(run_directory, primary_results)
        log("all_primary_arms_preserved")

        diagnostic = config["diagnostic"]
        if diagnostic["enabled"]:
            for seed in config["training"]["seeds"]:
                if time.perf_counter() - started < budget:
                    execute_arm(diagnostic, int(seed), is_diagnostic=True)
                else:
                    log("diagnostic_skipped_runtime_budget")
                    break

        spectrum_results: list[dict[str, Any]] = []
        spectrum_probe_count = int(config["measurement"]["spectrum_probe_samples"])
        for snapshot in model_snapshots:
            if time.perf_counter() - started >= budget:
                log("remaining_spectrum_skipped_runtime_budget")
                break
            model_config = config["model"]
            model = PilotMLP(
                int(model_config["input_dim"]),
                int(model_config["width"]),
                int(snapshot["hidden_layers"]),
                int(model_config["output_dim"]),
                float(snapshot["xavier_gain"]),
            ).to(device)
            model.load_state_dict(snapshot.pop("state_dict"))
            spectrum_layers = _selected_spectrum_layers(
                int(snapshot["hidden_layers"]), config["measurement"]["spectrum_layers"]
            )
            rows = []
            excluded_count = 0
            for dataset_index in probe_indices[:spectrum_probe_count]:
                if time.perf_counter() - started >= budget:
                    log("remaining_spectrum_skipped_runtime_budget")
                    break
                image, target = validation_dataset[dataset_index]
                measured = measure_sample(
                    model,
                    image.to(device),
                    torch.as_tensor(target, device=device, dtype=torch.long),
                    spectrum_layers,
                    float(config["measurement"]["directional_gain_denominator_threshold"]),
                )
                diagnostics = measured["jacobian_diagnostics"]
                excluded_count += sum(
                    int(row["directional_gain"]["excluded_near_zero_denominator"])
                    for row in diagnostics
                )
                rows.append({"dataset_index": dataset_index, "jacobian_diagnostics": diagnostics})
            spectrum_results.append(
                {
                    **snapshot,
                    "rows": rows,
                    "directional_gain_excluded_count": excluded_count,
                }
            )
        if spectrum_results:
            _write_json_exclusive(
                run_directory / "supplementary_spectrum.json",
                {"classification": "supplementary_exploratory", "results": spectrum_results},
            )

        artifact = {
            "run_id": run_id,
            "run_type": "feasibility_pilot",
            "protocol_version": "v1",
            "created_after_review": True,
            "scientific_conclusions": None,
            "config": config,
            "subset_manifest": "subset_indices.json",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision_version,
                "device": str(device),
                "git_revision": _git_revision(project_root),
            },
            "primary_results": primary_results,
            "diagnostic_results": diagnostic_results,
            "supplementary_spectrum": spectrum_results,
            "runtime_seconds": time.perf_counter() - started,
            "errors": [],
        }
        _write_json_exclusive(run_directory / "raw_results.json", artifact)
        log("pilot_complete")
        return run_directory
    except Exception as error:
        log(f"pilot_failed type={type(error).__name__} message={error}")
        _write_json_exclusive(
            run_directory / "failure.json",
            {
                "run_id": run_id,
                "protocol_version": "v1",
                "runtime_seconds": time.perf_counter() - started,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen bounded Fashion-MNIST pilot.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="New unique run id; existing directories are never overwritten")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        output = run_pilot(config, project_root, args.run_id)
    except Exception as error:
        print(f"Pilot did not complete: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

