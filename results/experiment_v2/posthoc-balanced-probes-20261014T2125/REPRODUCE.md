# Tái lập phép đo bổ sung

Không chạy training. Cần dataset Fashion-MNIST local và thư mục source run V2 cùng 60 checkpoint weights.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\measure_balanced_probes_v2.py `
  --source-run results\experiment_v2\experiment-v2-20261014T2053 `
  --protocol experiments\posthoc_balanced_probe_protocol_v2.json `
  --output-dir results\experiment_v2\posthoc-balanced-probes-NEW-ID
```

Sau đó kiểm toán output directory mới:

```powershell
.venv\Scripts\python.exe scripts\audit_balanced_probe_results_v2.py `
  --result-dir results\experiment_v2\posthoc-balanced-probes-NEW-ID
```

Protocol/code commit trước phép đo: `7ddb3c2e226433e2b8c391da84e01e00d5fbb636`.
Audit code commit: `53b0ddedc583204cbed218b610e38fad0b392a9a`.

