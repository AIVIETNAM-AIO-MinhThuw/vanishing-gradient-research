from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vanishing_gradient.experiment_v2 import (
    PilotMLP,
    aggregate_probes,
    cache_images,
    evaluate,
    measure_jacobian,
    measure_probe,
    selected_layers,
    stratified_split,
    write_json_exclusive,
)


def main() -> None:
    config = json.loads(Path("configs/experiment_v2.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    dataset = FashionMNIST(config["data_root"], train=True, download=False, transform=ToTensor())
    train_indices, validation_indices = stratified_split(dataset.targets, 10000, 2000, 42)
    train_images, train_targets = cache_images(dataset, train_indices)
    validation_images, validation_targets = cache_images(dataset, validation_indices)
    setup_seconds = time.perf_counter() - started

    torch.manual_seed(42)
    model = PilotMLP(784, 128, 24, 10, 1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    training_started = time.perf_counter()
    for step in range(10):
        positions = torch.arange(step * 64, (step + 1) * 64)
        logits, _ = model(train_images[positions])
        loss = F.cross_entropy(logits, train_targets[positions])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    training_seconds = time.perf_counter() - training_started

    measurement_started = time.perf_counter()
    evaluate(model, train_images, train_targets, 256, torch.device("cpu"))
    evaluate(model, validation_images, validation_targets, 256, torch.device("cpu"))
    rows = [
        measure_probe(model, validation_images[index], validation_targets[index], 2.0,
                      config["measurement"]["preactivation_histogram_boundaries"])
        for index in range(16)
    ]
    aggregate_probes(rows, 24, 2.0, config["measurement"]["preactivation_histogram_boundaries"])
    measurement_seconds = time.perf_counter() - measurement_started

    jacobian_started = time.perf_counter()
    for index in range(8):
        measure_jacobian(model, validation_images[index], validation_targets[index], selected_layers(24), 1e-6)
    jacobian_seconds = time.perf_counter() - jacobian_started

    primary_estimate = training_seconds * 60 * 12 + measurement_seconds * 5 * 12 + setup_seconds
    jacobian_estimate = jacobian_seconds * 27
    result = {
        "benchmark": "experiment_v2_cpu_d24", "training_steps": 10,
        "setup_seconds": setup_seconds, "training_10_steps_seconds": training_seconds,
        "one_checkpoint_measurement_seconds": measurement_seconds,
        "one_8_probe_3_layer_jacobian_snapshot_seconds": jacobian_seconds,
        "estimated_primary_seconds_d24_upper_bound": primary_estimate,
        "estimated_conditional_jacobian_seconds_d24_upper_bound": jacobian_estimate,
        "estimated_total_with_20_percent_margin_seconds": 1.2 * (primary_estimate + jacobian_estimate),
        "approved_budget_seconds": 1200,
    }
    destination = Path("results/benchmark/experiment_v2_cpu_d24.json")
    write_json_exclusive(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
