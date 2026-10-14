# vanishing-gradient-research

Khung tối thiểu cho nghiên cứu cơ chế vanishing gradient trong deep neural networks bằng PyTorch.

## Phạm vi hiện tại

Repository này **chỉ** chứa thiết lập ban đầu và một instrumentation smoke test. Smoke test kiểm tra đường đi tính toán và phép đo; kết quả của nó không phải bằng chứng khoa học về vanishing gradient. Verdict hiện tại là **PILOT ONLY**. Không triển khai Phase 1 hoặc full experiment nếu chưa có phê duyệt của người nghiên cứu.

## Cấu trúc

```text
configs/       cấu hình JSON có thể dùng trên CPU hoặc CUDA
experiments/   chỉ dẫn phạm vi; chưa có full experiment
results/       artifact JSON sinh từ lần chạy thật
src/           model và instrumentation smoke test
tests/         kiểm tra tự động bằng unittest
```

## Cài đặt local

Yêu cầu Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Chạy lại

```powershell
.venv\Scripts\python src\vanishing_gradient\smoke_test.py --config configs\smoke_test.json --output results\smoke_test.json
.venv\Scripts\python -m unittest discover -s tests -v
```

Lệnh smoke test trả exit code khác 0 nếu có phép kiểm tra thất bại. Artifact `results/smoke_test.json` lưu cấu hình, môi trường, tensor shape, gradient theo hidden activation, gradient theo weight, Jacobian, singular values, numerical rank, sai số chain rule, sai số finite differences, runtime và lỗi (nếu có).

## Chuyển sang Kaggle GPU T4

Logic model và instrumentation không phụ thuộc thiết bị. Trên Kaggle, upload repository hoặc notebook dataset, dùng PyTorch có sẵn, rồi chạy:

```bash
python src/vanishing_gradient/smoke_test.py \
  --config configs/smoke_test.json \
  --device cuda \
  --output results/smoke_test-kaggle.json
```

Không cần đổi code chính. `--device auto` cũng chọn CUDA khi có và CPU khi không có. Trước một pilot được phê duyệt, cần lưu phiên bản PyTorch/CUDA, loại GPU, seed và artifact đầu ra của chính lần chạy đó.

## Tự kiểm tra phép đo

- `activation_gradient`: \(\partial L / \partial h\), gradient của loss theo activation tại hidden layer được chọn; shape là `[batch_size, hidden_width]`.
- `weight_gradient`: \(\partial L / \partial W\), gradient theo tham số của linear layer sinh ra activation đó; shape là `[hidden_width, previous_width]`. Hai đại lượng không thay thế cho nhau.
- `jacobian`: \(J = \partial z / \partial h\) từ activation của một sample đến logits của sample đó; shape là `[output_dim, hidden_width]`.
- Chain rule được kiểm tra bằng `direct dL/dh` so với `J.T @ dL/dz`.
- Một số phần tử của Jacobian được đối chiếu bằng central finite differences trong float64.
- Singular values và numerical rank chỉ là chẩn đoán instrumentation của mạng ngẫu nhiên nhỏ, không được diễn giải thành kết luận nghiên cứu.

## Nguồn tham khảo quy trình

Quy trình được tham khảo có chọn lọc từ [Academic-Research-Agent-Skill](https://github.com/ngtiendong/Academic-Research-Agent-Skill): smoke test phải có seed, môi trường, runtime, artifact có cấu trúc và phải tách khỏi bằng chứng khoa học. Repository tham khảo không được clone hoặc chép toàn bộ vào project này.

Xem [AI_ASSISTANCE.md](AI_ASSISTANCE.md) để biết phần việc có AI hỗ trợ.

