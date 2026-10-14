# Vanishing Gradient Research — AIO2026

Thực nghiệm PyTorch khảo sát gradient propagation trong plain MLP trên Fashion-MNIST. Repository công bố code, protocol, test và các bảng/biểu đồ nhỏ có thể kiểm tra trực tiếp. Raw artifacts và model checkpoints được tách khỏi Git; xem [ARTIFACTS.md](ARTIFACTS.md).

Repository không đưa ra kết luận khoa học thay người nghiên cứu. Các kết quả dưới đây là mô tả provenance và phạm vi đo đã hoàn thành.

## Năm giai đoạn

1. **Pilot V1:** feasibility pilot D=3 so với D=12, ba seed, tối đa 100 bước; diagnostic gain=0.5 và Jacobian là supplementary.
2. **Main Experiment V2:** primary experiment với D=3, 6, 12, 24; width 128; Tanh; Xavier gain=1.0; ba seed; 600 bước; tổng 12 runs và 60 checkpoints.
3. **Post-hoc Balanced Probes:** phép đo exploratory trên 60 V2 checkpoints đã lưu, dùng 40 gradient probes cân bằng nhãn và 20 Jacobian probes; không training lại và không thay thế V2 gốc.
4. **Follow-up Evidence:** follow-up được thiết kế sau khi xem V2. Phần A/B tái sử dụng 60 V2 checkpoints; phần C chạy 6 runs/30 checkpoints cho hai đối chứng D=12. Đây không phải preregistered V2 evidence.
5. **RQ2 Exploratory Analysis:** phân tích mô tả 90 checkpoint-level rows và 3.600 per-probe rows. Verdict artifact là `OBSERVED_DESCRIPTIVELY`; đây là kết quả exploratory/descriptive, không phải confirmatory hay causal claim.

## Cấu trúc

```text
.agents/skills/research-agent/  skill hỗ trợ quy trình, kèm LICENSE và SOURCE.md
configs/                       cấu hình thực thi
experiments/                   protocol và phân loại từng giai đoạn
scripts/                       benchmark, measurement, audit và analysis entry points
src/vanishing_gradient/        implementation smoke test, Pilot V1 và V2
tests/                         unittest; một số test cần external artifacts
results/                       canonical summaries, audit metadata và local raw artifacts
ARTIFACTS.md                   bản đồ Git / Release / local-only
AI_ASSISTANCE.md               disclosure phần việc có AI hỗ trợ
```

## Môi trường

Yêu cầu Python 3.10+; môi trường đã ghi trong các artifact mới nhất dùng Python 3.11.9.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Dependencies được pin trong `requirements.txt`: PyTorch và torchvision. Fashion-MNIST được tải vào `data/` và không được đưa vào Git.

## Entry points

Các lệnh dưới đây tạo run ID/output mới; không dùng chúng để ghi đè canonical artifacts.

```powershell
# Instrumentation smoke test
.venv\Scripts\python src\vanishing_gradient\smoke_test.py `
  --config configs\smoke_test.json `
  --output results\smoke_test.json

# Pilot V1
.venv\Scripts\python src\vanishing_gradient\pilot.py `
  --config configs\pilot_v1.json `
  --run-id pilot-v1-NEW-UNIQUE-ID

# Main Experiment V2
.venv\Scripts\python src\vanishing_gradient\experiment_v2.py `
  --config configs\experiment_v2.json `
  --run-id experiment-v2-NEW-UNIQUE-ID

# Balanced-probe measurement; yêu cầu 60 V2 checkpoints
.venv\Scripts\python scripts\measure_balanced_probes_v2.py `
  --source-run results\experiment_v2\experiment-v2-20261014T2053 `
  --protocol experiments\posthoc_balanced_probe_protocol_v2.json `
  --output-dir results\experiment_v2\posthoc-balanced-probes-NEW-UNIQUE-ID

# Follow-up; yêu cầu V2 và balanced post-hoc artifacts tại paths trong config
.venv\Scripts\python scripts\followup_evidence.py `
  --config configs\followup_evidence.json `
  --run-id followup-evidence-NEW-UNIQUE-ID

# RQ2 analysis; không training nhưng cần ba source artifact và follow-up weights
.venv\Scripts\python scripts\analyze_rq2.py `
  --v2-dir results\experiment_v2\experiment-v2-20261014T2053 `
  --posthoc-dir results\experiment_v2\posthoc-balanced-probes-20261014T2125 `
  --followup-dir results\followup_evidence\followup-evidence-20261014T2312 `
  --output-dir results\rq2_analysis\rq2-analysis-NEW-UNIQUE-ID
```

Chi tiết input bắt buộc và lệnh tái lập nằm trong `REPRODUCE.md` của từng canonical run. Các bước cần checkpoints/raw data chỉ chạy được sau khi phục hồi đúng Release assets theo [ARTIFACTS.md](ARTIFACTS.md).

## Tests

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Unit tests thuần có thể chạy từ clean clone sau khi cài dependencies. Một số integration tests, đặc biệt balanced-probe measurement, phụ thuộc V2 artifacts/checkpoints không lưu trong Git. Test records đi kèm từng run là bằng chứng lịch sử; chúng không thay thế việc chạy lại trên clean clone.

## Provenance và diễn giải

- V2 là main experiment. Post-hoc balanced probes, Follow-up và RQ2 phải luôn giữ nhãn riêng; chúng không được trình bày như đã đăng ký trước cùng V2.
- Source/config/protocol local của Follow-up và source RQ2 khớp byte-for-byte với snapshot tương ứng trong handoff ZIP.
- Follow-up/RQ2 scripts chưa nằm trong commit được ghi tại thời điểm chạy. Commit Git tiếp theo chỉ bảo toàn hậu nghiệm các snapshot đã xác minh, không tạo ra preregistration hoặc provenance ngược thời gian.
- Condition number, gradient gap và accuracy là các phép đo khác nhau; repository không tự chuyển chúng thành causal claim hay research-gap claim.

Xem [AI_ASSISTANCE.md](AI_ASSISTANCE.md) để biết phạm vi AI hỗ trợ. Skill quy trình được giữ nguyên tại `.agents/skills/research-agent/` cùng license và nguồn gốc upstream.
