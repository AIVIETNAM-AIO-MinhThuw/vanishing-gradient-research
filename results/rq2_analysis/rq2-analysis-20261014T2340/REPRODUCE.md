# Tái lập phân tích RQ2

Phân tích này không training và không tạo optimizer. Nó đọc ba artifact nguồn cùng 30 checkpoint follow-up để tái đo per-probe gradient và parameter-gradient còn thiếu.

```powershell
.venv\Scripts\python.exe -m unittest tests.test_rq2_analysis -v
.venv\Scripts\python.exe scripts\analyze_rq2.py `
  --v2-dir results\experiment_v2\experiment-v2-20261014T2053 `
  --posthoc-dir results\experiment_v2\posthoc-balanced-probes-20261014T2125 `
  --followup-dir results\followup_evidence\followup-evidence-20261014T2312 `
  --output-dir results\rq2_analysis\rq2-analysis-NEW-UNIQUE-ID
```

Fashion-MNIST local là dependency để ánh xạ lại đúng 40 balanced probes; official test split không được dùng.
