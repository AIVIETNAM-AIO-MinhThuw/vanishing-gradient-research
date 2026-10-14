# Tái lập experiment v2

Thực hiện từ thư mục gốc repository. Fashion-MNIST không nằm trong gói ZIP.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\benchmark_experiment_v2.py
.venv\Scripts\python.exe src\vanishing_gradient\experiment_v2.py `
  --config configs\experiment_v2.json `
  --run-id experiment-v2-NEW-UNIQUE-ID
```

Lượt chạy được bàn giao dùng commit `4356d5deb3d692ee3da471e11b1b7c7c3f4e8998` và run ID
`experiment-v2-20261014T2053`. Code từ chối ghi đè run directory hoặc artifact đã tồn tại.

`data_and_batch_manifest.json` chứa toàn bộ index train/validation, fixed probes và đúng thứ tự
dataset index của 600 batch cho mỗi seed. Official Fashion-MNIST test split không được sử dụng.

Không tái chạy với cùng run ID. Một lượt tái lập phải dùng run ID mới và được xem là một measurement run riêng.

