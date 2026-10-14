# Experiment protocols

Thư mục này lưu protocol theo đúng giai đoạn nghiên cứu. Protocol là bằng chứng về thiết kế và phân loại; không tự chứng minh kết quả thực nghiệm.

| Giai đoạn | Protocol | Phân loại | Canonical run |
|---|---|---|---|
| Pilot V1 | `pilot_protocol_v1.json` | Feasibility pilot; primary D=3/D=12, diagnostic supplementary | `pilot-v1-20261014T1920` |
| Main V2 | `experiment_protocol_v2.json` | Main experiment, 12 runs/60 checkpoints | `experiment-v2-20261014T2053` |
| Balanced probes | `posthoc_balanced_probe_protocol_v2.json` | Post-hoc exploratory measurement; không training lại | `posthoc-balanced-probes-20261014T2125` |
| Follow-up evidence | `followup_evidence_protocol.json` | Follow-up thiết kế sau khi xem V2; không phải preregistered V2 | `followup-evidence-20261014T2312` |
| RQ2 analysis | Không có protocol độc lập | Exploratory/descriptive analysis trên artifact đã có | `rq2-analysis-20261014T2340` |

## Ranh giới diễn giải

- V2 gốc không bị thay thế bởi balanced-probe remeasurement.
- Follow-up dùng baseline V2 để đối chiếu nhưng là experimental stage riêng.
- RQ2 dùng dữ liệu V2, post-hoc và Follow-up; verdict `OBSERVED_DESCRIPTIVELY` không phải confirmatory finding và không chứng minh nhân quả.
- Run IDs là identifiers, không nên được dùng như timestamp duy nhất nếu không đối chiếu log/audit metadata.

## Entry points

| Giai đoạn | Entry point | Config/input chính |
|---|---|---|
| Pilot V1 | `src/vanishing_gradient/pilot.py` | `configs/pilot_v1.json` |
| Main V2 | `src/vanishing_gradient/experiment_v2.py` | `configs/experiment_v2.json` |
| Balanced probes | `scripts/measure_balanced_probes_v2.py` | V2 run + balanced-probe protocol |
| Follow-up | `scripts/followup_evidence.py` | `configs/followup_evidence.json` |
| RQ2 | `scripts/analyze_rq2.py` | Ba canonical source directories |

Không sửa protocol hoặc canonical artifact để làm cho provenance có vẻ sớm hơn thực tế. Danh sách artifact công bố và dependency phục hồi được ghi tại [`../ARTIFACTS.md`](../ARTIFACTS.md).
