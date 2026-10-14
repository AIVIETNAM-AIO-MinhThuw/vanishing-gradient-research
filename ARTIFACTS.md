# Artifact index

Chỉ mục này phân biệt artifact được track trong Git, artifact phát hành riêng và dữ liệu chỉ giữ local. Nó không thay đổi hay thay thế canonical results.

## Canonical runs

| Stage | Canonical directory | Classification |
|---|---|---|
| Pilot V1 | `results/pilot/pilot-v1-20261014T1920/` | Feasibility pilot |
| Pilot audit | `results/audits/pilot-v1-data-audit-20261014T193404/` | Read-only audit |
| Main V2 | `results/experiment_v2/experiment-v2-20261014T2053/` | Main experiment |
| Balanced probes | `results/experiment_v2/posthoc-balanced-probes-20261014T2125/` | Post-hoc exploratory measurement |
| Follow-up | `results/followup_evidence/followup-evidence-20261014T2312/` | Follow-up evidence experiment |
| RQ2 | `results/rq2_analysis/rq2-analysis-20261014T2340/` | Exploratory/descriptive analysis |

## A — Git repository

Track source, configs, protocols, tests, disclosure, small tables/figures and provenance summaries:

- `src/`, `scripts/`, `configs/`, `experiments/`, `tests/`.
- `.agents/skills/research-agent/`, bao gồm `LICENSE` và `SOURCE.md`.
- `README.md`, `AI_ASSISTANCE.md`, `requirements.txt` và file này.
- Canonical `*.csv`, `*.svg`, `audit*.json`, `verification.json`, `REPRODUCE.md`, `RUN_SUMMARY.md`, `README_VI.md`, `TEST_RESULTS.txt`, `GIT_REVISION.txt` và checksum text files không chứa raw payload lớn.
- Probe/checkpoint provenance metadata nhỏ như `checkpoint_manifest.json`, `probe_manifest.json`, `subset_indices.json` và `ORIGINAL_PROBE_ISSUE.json`.

Các summary trong Git giúp xem loss/accuracy, layerwise gradients, Jacobian conditioning, saturation, probe provenance và sự phân tách V2/post-hoc mà không cần tải model weights.

## B — GitHub Release hoặc artifact storage

Các file sau được giữ ngoài Git nhưng cần cho kiểm toán sâu hoặc tái chạy các bước phụ thuộc checkpoint:

- V1 `raw_results.json` và `primary_manifest.json`.
- V2 `raw_results.json`, `data_and_batch_manifest.json`, `supplementary_jacobian.json` và archive chứa 60 checkpoints.
- Balanced post-hoc `raw_posthoc_results.json`.
- Follow-up `raw_results.json` và archive chứa 30 checkpoints.
- Handoff ZIP canonical khi dùng làm provenance snapshot.

Release phải công bố SHA-256, stage/run ID, kích thước và đường dẫn phục hồi tương đối. Không đổi tên artifact thành một giai đoạn khác và không trình bày Follow-up/RQ2 như preregistered V2.

Snapshot provenance đã xác minh:

- `followup-evidence-20261014T2312-handoff.zip`: `25f0caa1aa15ee66f36229c6f7ccc22863b05fc98c9edc7f98833d40e22f4465`.
- `rq2-analysis-20261014T2340-handoff.zip`: `8d2157ad52221a63f851785962f53fee0aa2fb2aca7f8e81c94a8028daf2708a`.

Follow-up/RQ2 source local khớp snapshot có thể đối chiếu. Các script này chưa tồn tại trong commit được artifact ghi tại thời điểm chạy; commit tương lai chỉ bảo toàn phiên bản hậu nghiệm.

## C — Local only

- `data/` và toàn bộ Fashion-MNIST tải về.
- `.venv/`, Python caches, pytest caches, IDE metadata và runtime logs không cần phát hành.
- Checkpoint directories đã có bản Release archive.
- Duplicate handoff directories và các bản copy bàn giao trùng canonical artifacts.
- Per-run duplicate JSON khi cùng nội dung đã có trong canonical raw aggregation.

## Dependency map

- Balanced post-hoc cần V2 raw metadata, manifests và 60 V2 checkpoints.
- Follow-up A/B cần V2 và balanced post-hoc artifacts; Follow-up C tạo 30 checkpoints riêng.
- RQ2 cần V2 raw results, balanced post-hoc raw results, Follow-up raw results, 30 Follow-up checkpoints và Fashion-MNIST local để ánh xạ lại probes.
- Một số integration tests cần các external artifacts trên; clean clone chỉ có thể chạy chúng sau khi Release assets được phục hồi đúng canonical paths.

Không đưa dataset, `.venv`, checkpoint `.pt` hoặc ZIP vào Git history.
