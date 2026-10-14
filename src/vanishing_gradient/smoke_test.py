from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class InstrumentedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int, hidden_layers: int, output_dim: int) -> None:
        super().__init__()
        widths = [input_dim] + [hidden_width] * hidden_layers
        self.hidden = nn.ModuleList(
            nn.Linear(widths[index], widths[index + 1]) for index in range(hidden_layers)
        )
        self.output = nn.Linear(hidden_width, output_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, list[Tensor]]:
        activations: list[Tensor] = []
        value = inputs
        for layer in self.hidden:
            value = torch.tanh(layer(value))
            activations.append(value)
        return self.output(value), activations

    def from_hidden(self, hidden_layer_index: int, activation: Tensor) -> Tensor:
        value = activation
        for layer in self.hidden[hidden_layer_index + 1 :]:
            value = torch.tanh(layer(value))
        return self.output(value)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def _resolve_dtype(name: str) -> torch.dtype:
    dtypes = {"float32": torch.float32, "float64": torch.float64}
    if name not in dtypes:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(dtypes)}")
    return dtypes[name]


def _relative_error(actual: Tensor, expected: Tensor) -> float:
    tiny = torch.finfo(actual.dtype).eps
    denominator = torch.maximum(torch.maximum(actual.abs(), expected.abs()), torch.tensor(tiny, device=actual.device))
    return float(((actual - expected).abs() / denominator).max().item())


def _finite_difference_checks(
    model: InstrumentedMLP,
    layer_index: int,
    hidden: Tensor,
    jacobian: Tensor,
    epsilon: float,
    count: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    pairs = [
        (flat_index // hidden.numel(), flat_index % hidden.numel())
        for flat_index in torch.randperm(jacobian.numel(), generator=torch.Generator().manual_seed(0))[:count].tolist()
    ]
    rows = []
    for output_index, hidden_index in pairs:
        offset = torch.zeros_like(hidden)
        offset[hidden_index] = epsilon
        with torch.no_grad():
            plus = model.from_hidden(layer_index, hidden + offset)[output_index]
            minus = model.from_hidden(layer_index, hidden - offset)[output_index]
        numerical = (plus - minus) / (2.0 * epsilon)
        analytic = jacobian[output_index, hidden_index]
        absolute_error = float((analytic - numerical).abs().item())
        relative_error = _relative_error(analytic.reshape(1), numerical.reshape(1))
        rows.append(
            {
                "output_index": output_index,
                "hidden_index": hidden_index,
                "autograd": float(analytic.item()),
                "finite_difference": float(numerical.item()),
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "pass": bool(torch.isclose(analytic, numerical, rtol=rtol, atol=atol).item()),
            }
        )
    return {
        "epsilon": epsilon,
        "rtol": rtol,
        "atol": atol,
        "elements": rows,
        "max_absolute_error": max(row["absolute_error"] for row in rows),
        "max_relative_error": max(row["relative_error"] for row in rows),
        "pass": all(row["pass"] for row in rows),
    }


def run_smoke_test(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    torch.manual_seed(int(config["seed"]))
    device = _resolve_device(str(config["device"]))
    dtype = _resolve_dtype(str(config["dtype"]))

    batch_size = int(config["batch_size"])
    input_dim = int(config["input_dim"])
    hidden_width = int(config["hidden_width"])
    hidden_layers = int(config["hidden_layers"])
    output_dim = int(config["output_dim"])
    layer_index = int(config["hidden_layer_index"])
    sample_index = int(config["sample_index"])
    if config["activation"] != "tanh":
        raise ValueError("This bounded smoke test supports activation='tanh' only")
    if not 0 <= layer_index < hidden_layers:
        raise ValueError("hidden_layer_index is outside the hidden-layer range")
    if not 0 <= sample_index < batch_size:
        raise ValueError("sample_index is outside the batch range")

    model = InstrumentedMLP(input_dim, hidden_width, hidden_layers, output_dim).to(device=device, dtype=dtype)
    inputs = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
    targets = torch.randn(batch_size, output_dim, device=device, dtype=dtype)
    logits, activations = model(inputs)
    selected_activation = activations[layer_index]
    selected_activation.retain_grad()
    loss = nn.functional.mse_loss(logits, targets)

    gradient_logits, gradient_activation = torch.autograd.grad(
        loss, (logits, selected_activation), retain_graph=True
    )
    # autograd.grad triggers retain_grad's hook; clear it before the independent backward check.
    selected_activation.grad = None
    model.zero_grad(set_to_none=True)
    loss.backward()
    weight_gradient = model.hidden[layer_index].weight.grad
    if selected_activation.grad is None or weight_gradient is None:
        raise RuntimeError("Backward did not populate the requested gradients")

    sample_hidden = selected_activation[sample_index].detach().clone().requires_grad_(True)

    def hidden_to_logits(value: Tensor) -> Tensor:
        return model.from_hidden(layer_index, value)

    jacobian = torch.autograd.functional.jacobian(hidden_to_logits, sample_hidden)
    chain_rule_gradient = jacobian.transpose(0, 1) @ gradient_logits[sample_index]
    direct_gradient = gradient_activation[sample_index]
    chain_rule_difference = direct_gradient - chain_rule_gradient
    chain_rule = {
        "rtol": float(config["chain_rule_rtol"]),
        "atol": float(config["chain_rule_atol"]),
        "max_absolute_error": float(chain_rule_difference.abs().max().item()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(chain_rule_difference).item()
            / max(torch.linalg.vector_norm(direct_gradient).item(), torch.finfo(dtype).eps)
        ),
        "pass": bool(
            torch.allclose(
                direct_gradient,
                chain_rule_gradient,
                rtol=float(config["chain_rule_rtol"]),
                atol=float(config["chain_rule_atol"]),
            )
        ),
    }

    finite_difference = _finite_difference_checks(
        model=model,
        layer_index=layer_index,
        hidden=sample_hidden.detach(),
        jacobian=jacobian.detach(),
        epsilon=float(config["finite_difference_epsilon"]),
        count=min(int(config["finite_difference_checks"]), jacobian.numel()),
        rtol=float(config["finite_difference_rtol"]),
        atol=float(config["finite_difference_atol"]),
    )
    singular_values = torch.linalg.svdvals(jacobian.detach())
    rank_tolerance = float(
        singular_values.max().item() * max(jacobian.shape) * torch.finfo(dtype).eps
    )
    numerical_rank = int((singular_values > rank_tolerance).sum().item())

    tensors = {
        "inputs": inputs,
        "targets": targets,
        "logits": logits,
        "loss": loss,
        "selected_activation": selected_activation,
        "activation_gradient": selected_activation.grad,
        "weight_gradient": weight_gradient,
        "jacobian": jacobian,
        "singular_values": singular_values,
    }
    finite_tensors = all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors.values())
    expected_weight_input = input_dim if layer_index == 0 else hidden_width
    shape_checks = {
        "logits": list(logits.shape) == [batch_size, output_dim],
        "selected_activation": list(selected_activation.shape) == [batch_size, hidden_width],
        "activation_gradient": list(selected_activation.grad.shape) == [batch_size, hidden_width],
        "weight_gradient": list(weight_gradient.shape) == [hidden_width, expected_weight_input],
        "jacobian": list(jacobian.shape) == [output_dim, hidden_width],
        "singular_values": list(singular_values.shape) == [min(output_dim, hidden_width)],
    }
    backward_matches_autograd = bool(
        torch.allclose(selected_activation.grad, gradient_activation, rtol=0.0, atol=0.0)
    )
    checks = {
        "forward_loss_backward": bool(torch.isfinite(loss).item()) and backward_matches_autograd,
        "tensor_shapes": all(shape_checks.values()),
        "no_nan_or_inf": finite_tensors,
        "activation_and_weight_gradients_distinguished": selected_activation.grad.data_ptr()
        != weight_gradient.data_ptr(),
        "chain_rule": chain_rule["pass"],
        "finite_difference_jacobian": finite_difference["pass"],
        "numerical_rank_in_bounds": 0 <= numerical_rank <= min(jacobian.shape),
        "float64": dtype == torch.float64,
    }

    return {
        "artifact_type": "engineering_instrumentation_smoke_test",
        "scientific_evidence": False,
        "phase": "setup_only",
        "verdict_scope": "PILOT ONLY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
        "measurements": {
            "loss": float(loss.item()),
            "activation_gradient": {
                "definition": "d(loss)/d(hidden activation)",
                "shape": list(selected_activation.grad.shape),
                "l2_norm": float(torch.linalg.vector_norm(selected_activation.grad).item()),
            },
            "weight_gradient": {
                "definition": "d(loss)/d(weight producing the selected hidden activation)",
                "shape": list(weight_gradient.shape),
                "l2_norm": float(torch.linalg.vector_norm(weight_gradient).item()),
            },
            "jacobian": {
                "definition": "d(sample logits)/d(sample hidden activation)",
                "shape": list(jacobian.shape),
                "singular_values": singular_values.detach().cpu().tolist(),
                "rank_tolerance": rank_tolerance,
                "numerical_rank": numerical_rank,
            },
            "chain_rule": chain_rule,
            "finite_difference": finite_difference,
        },
        "shape_checks": shape_checks,
        "checks": checks,
        "all_passed": all(checks.values()),
        "runtime_seconds": time.perf_counter() - started,
        "errors": [],
    }


def _failure_artifact(config: dict[str, Any], error: BaseException, runtime_seconds: float) -> dict[str, Any]:
    return {
        "artifact_type": "engineering_instrumentation_smoke_test",
        "scientific_evidence": False,
        "phase": "setup_only",
        "verdict_scope": "PILOT ONLY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "all_passed": False,
        "runtime_seconds": runtime_seconds,
        "errors": [{"type": type(error).__name__, "message": str(error)}],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded PyTorch instrumentation smoke test.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override config device")
    args = parser.parse_args()

    started = time.perf_counter()
    config: dict[str, Any] = {}
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.device:
            config["device"] = args.device
        artifact = run_smoke_test(config)
    except Exception as error:
        artifact = _failure_artifact(config, error, time.perf_counter() - started)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    return 0 if artifact["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

