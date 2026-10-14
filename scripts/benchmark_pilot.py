from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import __version__ as torchvision_version
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vanishing_gradient.pilot import (  # noqa: E402
    PilotMLP,
    measure_sample,
    stratified_indices,
)


def main() -> int:
    output = PROJECT_ROOT / "results" / "benchmark" / "cpu_10_steps.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Benchmark artifact already exists: {output}")

    dataset = datasets.FashionMNIST(
        PROJECT_ROOT / "data", train=True, download=False, transform=transforms.ToTensor()
    )
    indices = stratified_indices(dataset.targets, 1024, 42)
    timings = []
    for depth in (3, 12):
        torch.manual_seed(42)
        model = PilotMLP(784, 128, depth, 10, 1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = DataLoader(
            Subset(dataset, indices), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42)
        )
        iterator = iter(loader)
        started = time.perf_counter()
        for _ in range(10):
            images, targets = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
        training_seconds = time.perf_counter() - started

        image, target = dataset[indices[0]]
        started = time.perf_counter()
        measure_sample(model, image, torch.tensor(target), [], 1e-6)
        per_sample_gradient_seconds = time.perf_counter() - started
        started = time.perf_counter()
        measure_sample(model, image, torch.tensor(target), [0, (depth - 1) // 2, depth - 1], 1e-6)
        three_jacobian_seconds = time.perf_counter() - started
        timings.append(
            {
                "hidden_layers": depth,
                "ten_training_steps_seconds": training_seconds,
                "per_sample_gradient_seconds": per_sample_gradient_seconds,
                "three_jacobians_seconds": three_jacobian_seconds,
            }
        )

    estimated_primary_seconds = sum(
        10.0 * row["ten_training_steps_seconds"]
        + 32.0 * row["per_sample_gradient_seconds"]
        for row in timings
    ) * 3.0
    estimated_spectrum_seconds = sum(
        8.0 * row["three_jacobians_seconds"] for row in timings
    ) * 3.0
    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision_version,
            "device": "cpu",
        },
        "timings": timings,
        "estimate": {
            "primary_training_and_probe_seconds": estimated_primary_seconds,
            "supplementary_spectrum_seconds": estimated_spectrum_seconds,
            "combined_seconds_before_evaluation_overhead": estimated_primary_seconds + estimated_spectrum_seconds,
            "note": "Linear extrapolation from 10 steps and one fixed sample; actual runtime is recorded by the pilot.",
        },
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
