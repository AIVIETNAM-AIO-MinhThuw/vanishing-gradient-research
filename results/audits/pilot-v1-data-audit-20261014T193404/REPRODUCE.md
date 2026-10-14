# Hướng dẫn tái lập và kiểm tra

Các lệnh được chạy từ thư mục gốc project trên PowerShell/Windows.

## Tạo môi trường

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Phiên bản của lượt pilot: Python 3.11.9, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, NumPy 2.4.6, Pillow 12.3.0. Fashion-MNIST không nằm trong ZIP.

## Chạy unit test (không training)

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Chạy lại kiểm toán read-only trên artifact hiện có

Chọn một output directory mới vì công cụ không ghi đè thư mục kiểm toán đã tồn tại.

```powershell
.venv\Scripts\python.exe scripts\audit_pilot_results.py `
  --run-dir results\pilot\pilot-v1-20261014T1920 `
  --config configs\pilot_v1.json `
  --protocol experiments\pilot_protocol_v1.json `
  --output-dir results\audits\pilot-v1-data-audit-NEW
```

## Tái chạy pilot

Không cần và không được thực hiện trong nhiệm vụ kiểm toán này. Nếu người nghiên cứu phê duyệt một lượt tái lập riêng, dùng run ID mới để bảo vệ artifact cũ:

```powershell
.venv\Scripts\python.exe src\vanishing_gradient\pilot.py `
  --config configs\pilot_v1.json `
  --run-id pilot-v1-NEW-RUN-ID
```

