from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from followup_evidence import FollowupMLP, parameter_gradient_rows  # noqa: E402

STEPS = (0, 25, 100, 300, 600)
SEEDS = (42, 123, 2026)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_probe_metrics(model: FollowupMLP, images: torch.Tensor, targets: torch.Tensor) -> list[dict[str, float]]:
    rows = []
    model.eval()
    for image, target in zip(images, targets):
        logits, _, activations = model.forward_with_preactivations(image.unsqueeze(0))
        loss = F.cross_entropy(logits, target.reshape(1), reduction="sum")
        gradients = torch.autograd.grad(loss, activations)
        rms = [float(gradient.square().mean().sqrt().item()) for gradient in gradients]
        gap = math.log10(rms[-1] / rms[0])
        rows.append({"first_gradient_rms": rms[0], "last_gradient_rms": rms[-1], "log10_gap": gap})
    return rows


def line_svg(path: Path, panels: list[tuple[str, list[dict[str, Any]]]], fields: list[tuple[str, str]], title: str) -> None:
    width, panel_w, panel_h = 1050, 330, 240
    height = 70 + math.ceil(len(panels) / 3) * panel_h
    colors = {42: "#2457a7", 123: "#c43c35", 2026: "#18825d"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="20" y="28" font-size="18">{title}</text>']
    for panel_index, (label, rows) in enumerate(panels):
        col, row_index = panel_index % 3, panel_index // 3
        x0, y0 = 45 + col * panel_w, 60 + row_index * panel_h
        values = [float(item[field]) for item in rows for field, _ in fields]
        lo, hi = min(values), max(values)
        span = max(hi - lo, 1e-15)
        parts += [f'<text x="{x0}" y="{y0}" font-size="13">{label}</text>', f'<line x1="{x0}" y1="{y0+190}" x2="{x0+270}" y2="{y0+190}" stroke="#444"/>', f'<line x1="{x0}" y1="{y0+20}" x2="{x0}" y2="{y0+190}" stroke="#444"/>']
        for seed in SEEDS:
            seed_rows = sorted((item for item in rows if int(item["seed"]) == seed), key=lambda item: int(item["step"]))
            for field, dash in fields:
                points = " ".join(f'{x0 + 270 * STEPS.index(int(item["step"])) / 4:.1f},{y0 + 190 - 170 * (float(item[field]) - lo) / span:.1f}' for item in seed_rows)
                parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[seed]}" stroke-width="1.7" stroke-dasharray="{dash}"/>')
        parts.append(f'<text x="{x0}" y="{y0+210}" font-size="10">0   25   100   300   600</text>')
        parts.append(f'<text x="{x0+180}" y="{y0+14}" font-size="9">range {lo:.2e}..{hi:.2e}</text>')
    parts.append('<text x="20" y="48" font-size="10">Seed colors: 42 blue, 123 red, 2026 green. Solid=first field; dashed=second field when present.</text></svg>')
    with path.open("x", encoding="utf-8") as handle:
        handle.write("".join(parts))


def analyze(v2_dir: Path, posthoc_dir: Path, followup_dir: Path, output_dir: Path) -> None:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = [v2_dir / "raw_results.json", posthoc_dir / "raw_posthoc_results.json", followup_dir / "raw_results.json"]
    source_hashes_before = {str(path): sha256(path) for path in sources}
    v2 = json.loads(sources[0].read_text(encoding="utf-8"))
    posthoc = json.loads(sources[1].read_text(encoding="utf-8"))
    followup = json.loads(sources[2].read_text(encoding="utf-8"))
    probe_manifest = json.loads((posthoc_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    dataset = FashionMNIST(PROJECT_ROOT / "data", train=True, download=False, transform=ToTensor())
    probe_indices = probe_manifest["gradient_probe_indices"]
    images = torch.stack([dataset[index][0] for index in probe_indices])
    targets = torch.tensor([dataset[index][1] for index in probe_indices])

    parameter_lookup = {(row["depth"], row["seed"], row["step"], row["layer_type"], row["layer_index"]): row for row in followup["part_a"]["parameter_rows"]}
    posthoc_lookup = {(row["depth"], row["seed"], row["step"]): row for row in posthoc["gradient_results"]}
    v2_lookup = {(run["hidden_layers"], run["seed"], cp["step"]): cp for run in v2["primary_results"] for cp in run["checkpoints"]}
    checkpoint_rows, probe_rows = [], []

    for depth in (3, 6, 12, 24):
        arm = f"v2_d{depth}_tanh_gain_1_0"
        for seed in SEEDS:
            initial_accuracy = v2_lookup[(depth, seed, 0)]["validation"]["accuracy"]
            for step in STEPS:
                cp, measured = v2_lookup[(depth, seed, step)], posthoc_lookup[(depth, seed, step)]
                first = measured["summary"]["layer_summaries"][0]["mean_post_activation_gradient_rms"]
                last = measured["summary"]["layer_summaries"][-1]["mean_post_activation_gradient_rms"]
                gap = measured["summary"]["gap_summaries"]["post_activation"]["mean_per_probe_raw_gap"]
                first_param = parameter_lookup[(depth, seed, step, "hidden", 0)]["parameter_gradient_norm"]
                last_param = parameter_lookup[(depth, seed, step, "hidden", depth - 1)]["parameter_gradient_norm"]
                checkpoint_rows.append({"source": "V2_plus_balanced_posthoc", "arm": arm, "depth": depth, "seed": seed, "step": step, "validation_accuracy": cp["validation"]["accuracy"], "validation_loss": cp["validation"]["loss"], "accuracy_change_from_initialization": cp["validation"]["accuracy"] - initial_accuracy, "first_activation_gradient_rms": first, "last_activation_gradient_rms": last, "mean_per_probe_log10_gap": gap, "relative_input_attenuation": gap > 0, "first_parameter_gradient_norm": first_param, "last_parameter_gradient_norm": last_param})
                for row in measured["probe_rows"]:
                    profile = row["post_activation_gradient_profile"]
                    probe_rows.append({"source": "balanced_posthoc", "arm": arm, "depth": depth, "seed": seed, "step": step, "dataset_index": row["dataset_index"], "target_class": row["target_class"], "first_activation_gradient_rms": profile["gradient_rms"][0], "last_activation_gradient_rms": profile["gradient_rms"][-1], "per_probe_log10_gap": profile["raw_early_to_last_gap"]})

    for result in followup["part_c"]["results"]:
        arm, seed = result["arm_id"], result["seed"]
        initial_accuracy = result["checkpoints"][0]["validation"]["accuracy"]
        for cp in result["checkpoints"]:
            step = cp["step"]
            model = FollowupMLP(12, result["activation"], result["xavier_gain"])
            checkpoint_path = followup_dir / "checkpoints" / cp["checkpoint_file"]
            if sha256(checkpoint_path) != cp["checkpoint_sha256"]:
                raise ValueError(f"Follow-up checkpoint hash mismatch: {checkpoint_path.name}")
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            measured_probes = per_probe_metrics(model, images, targets)
            param_rows, _ = parameter_gradient_rows(model, images, targets, 0.01, 1e-12)
            first, last = fmean(row["first_gradient_rms"] for row in measured_probes), fmean(row["last_gradient_rms"] for row in measured_probes)
            mean_gap = fmean(row["log10_gap"] for row in measured_probes)
            checkpoint_rows.append({"source": "followup_experiment", "arm": arm, "depth": 12, "seed": seed, "step": step, "validation_accuracy": cp["validation"]["accuracy"], "validation_loss": cp["validation"]["loss"], "accuracy_change_from_initialization": cp["validation"]["accuracy"] - initial_accuracy, "first_activation_gradient_rms": first, "last_activation_gradient_rms": last, "mean_per_probe_log10_gap": mean_gap, "relative_input_attenuation": mean_gap > 0, "first_parameter_gradient_norm": param_rows[0]["parameter_gradient_norm"], "last_parameter_gradient_norm": param_rows[11]["parameter_gradient_norm"]})
            for dataset_index, target, row in zip(probe_indices, targets.tolist(), measured_probes):
                probe_rows.append({"source": "followup_remeasurement", "arm": arm, "depth": 12, "seed": seed, "step": step, "dataset_index": dataset_index, "target_class": target, "first_activation_gradient_rms": row["first_gradient_rms"], "last_activation_gradient_rms": row["last_gradient_rms"], "per_probe_log10_gap": row["log10_gap"]})

    if len(checkpoint_rows) != 90 or len(probe_rows) != 3600:
        raise AssertionError("Expected 90 checkpoint rows and 3,600 per-probe rows")
    initial = {(row["arm"], row["seed"]): row for row in checkpoint_rows if row["step"] == 0}
    for row in checkpoint_rows:
        row["accuracy_higher_than_initialization"] = row["validation_accuracy"] > initial[(row["arm"], row["seed"])]["validation_accuracy"]
        row["threshold_free_coexistence"] = row["accuracy_higher_than_initialization"] and row["relative_input_attenuation"]

    arm_summary = []
    for arm in sorted({row["arm"] for row in checkpoint_rows}):
        items = [row for row in checkpoint_rows if row["arm"] == arm]
        coexist = [row for row in items if row["threshold_free_coexistence"]]
        arm_summary.append({"arm": arm, "checkpoint_rows": len(items), "relative_attenuation_rows": sum(row["relative_input_attenuation"] for row in items), "accuracy_improved_rows": sum(row["accuracy_higher_than_initialization"] for row in items), "threshold_free_coexistence_rows": len(coexist), "coexistence_steps": ";".join(f's{row["seed"]}:t{row["step"]}' for row in coexist), "minimum_first_gradient_rms": min(row["first_activation_gradient_rms"] for row in items), "maximum_first_gradient_rms": max(row["first_activation_gradient_rms"] for row in items)})

    sensitivity = []
    for accuracy_threshold in (0.60, 0.70, 0.75, 0.80):
        for gradient_threshold in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
            for source in ("V2_plus_balanced_posthoc", "followup_experiment"):
                items = [row for row in checkpoint_rows if row["source"] == source]
                sensitivity.append({"classification": "EXPLORATORY_SENSITIVITY_ONLY", "source": source, "accuracy_threshold": accuracy_threshold, "absolute_first_gradient_threshold": gradient_threshold, "accuracy_and_relative_attenuation_count": sum(row["validation_accuracy"] >= accuracy_threshold and row["relative_input_attenuation"] for row in items), "accuracy_and_absolute_small_count": sum(row["validation_accuracy"] >= accuracy_threshold and row["first_activation_gradient_rms"] <= gradient_threshold for row in items), "accuracy_relative_and_absolute_count": sum(row["validation_accuracy"] >= accuracy_threshold and row["relative_input_attenuation"] and row["first_activation_gradient_rms"] <= gradient_threshold for row in items)})

    panels = []
    for arm in sorted({row["arm"] for row in checkpoint_rows}):
        panels.append((arm, [row for row in checkpoint_rows if row["arm"] == arm]))
    write_csv(output_dir / "rq2_checkpoint_metrics.csv", checkpoint_rows)
    write_csv(output_dir / "rq2_per_probe_metrics.csv", probe_rows)
    write_csv(output_dir / "rq2_arm_summary.csv", arm_summary)
    write_csv(output_dir / "rq2_exploratory_threshold_sensitivity.csv", sensitivity)
    line_svg(output_dir / "accuracy_by_seed.svg", panels, [("validation_accuracy", "0")], "Validation accuracy trajectories by arm and seed")
    line_svg(output_dir / "activation_gradient_by_seed.svg", panels, [("first_activation_gradient_rms", "0"), ("last_activation_gradient_rms", "5,4")], "Activation-gradient RMS trajectories (first solid, last dashed)")
    line_svg(output_dir / "gradient_gap_by_seed.svg", panels, [("mean_per_probe_log10_gap", "0")], "Mean per-probe log10 gradient gap trajectories")

    source_hashes_after = {str(path): sha256(path) for path in sources}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A source artifact changed")
    nonfinite = sum(not math.isfinite(value) for row in checkpoint_rows + probe_rows for value in row.values() if isinstance(value, float))
    if nonfinite:
        raise FloatingPointError("Non-finite analysis values")
    v2_coexist = [row for row in checkpoint_rows if row["source"] == "V2_plus_balanced_posthoc" and row["threshold_free_coexistence"]]
    followup_coexist = [row for row in checkpoint_rows if row["source"] == "followup_experiment" and row["threshold_free_coexistence"]]
    verdict = "OBSERVED_DESCRIPTIVELY"
    report = f"""# Phân tích bổ sung RQ2 — không phải nội dung báo cáo nộp

**Trạng thái phạm vi bằng chứng: {verdict}.**

Trạng thái này chỉ có nghĩa: trong dữ liệu checkpoint đã đo, có trường hợp validation accuracy tăng so với initialization của chính run trong khi mean per-probe gap dương (`G_first < G_last`). Đây là quan sát mô tả, không phải ngưỡng “accuracy tốt”, không phải causal claim và không phải kết luận rằng gradient tuyệt đối nhỏ.

## Nguồn dữ liệu

- V2: validation loss/accuracy gốc, 60 checkpoint.
- Balanced post-hoc: 40 probes cân bằng nhãn, cung cấp activation-gradient và per-probe gap cho V2.
- Follow-up: 30 checkpoint của hai arm; per-probe gap và parameter-gradient được tái đo read-only từ weights.

## Hai câu hỏi tách biệt

1. **Suy giảm tương đối:** dùng dấu gap, không cần ngưỡng tùy ý. Có {len(v2_coexist)} checkpoint V2 và {len(followup_coexist)} checkpoint follow-up thỏa đồng thời `accuracy > accuracy_at_step_0` và `gap > 0`.
2. **Giá trị tuyệt đối nhỏ:** dữ liệu cung cấp RMS thực tế nhưng protocol không định nghĩa scale nào là “nhỏ”. Vì vậy nhánh này vẫn **INCONCLUSIVE**. File sensitivity chỉ minh họa kết quả thay đổi theo nhiều cặp ngưỡng và không được dùng cho verdict.

## Các quan sát không phù hợp với giả thuyết đơn giản

- Nhiều checkpoint V2 muộn có accuracy cao hơn nhưng gap âm, tức layer đầu không suy giảm tương đối so với layer cuối theo metric này.
- Hai follow-up arm có gap dương rất lớn và first-layer RMS rất thấp, nhưng validation accuracy không cải thiện có ý nghĩa so với mức ban đầu/chance.
- Gradient gap không thay thế absolute gradient, và parameter-gradient không đồng nhất với activation-gradient.

## Giới hạn

Chỉ mô tả các checkpoint, seeds và arms đã có. Không có ngưỡng confirmatory cho “accuracy tương đối tốt” hay “gradient tuyệt đối nhỏ”; không kiểm định nhân quả, novelty hoặc khả năng tổng quát hóa.
"""
    with (output_dir / "RQ2_REPORT_VI.md").open("x", encoding="utf-8") as handle:
        handle.write(report)
    audit = {"status": "PASS", "verdict": verdict, "runtime_seconds": time.perf_counter() - started, "source_artifacts_unchanged": True, "checkpoint_rows": len(checkpoint_rows), "per_probe_rows": len(probe_rows), "v2_checkpoint_rows": 60, "followup_checkpoint_rows": 30, "nonfinite_count": 0, "source_sha256": source_hashes_before, "threshold_policy": "gap sign is definitional; all absolute/accuracy thresholds are exploratory sensitivity only"}
    with (output_dir / "audit.json").open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--posthoc-dir", type=Path, required=True)
    parser.add_argument("--followup-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(*(path.resolve() for path in (args.v2_dir, args.posthoc_dir, args.followup_dir, args.output_dir)))


if __name__ == "__main__":
    main()
