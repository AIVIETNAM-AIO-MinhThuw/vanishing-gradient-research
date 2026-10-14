# Reproduce followup-evidence-20261014T2312

This is a post-V2 follow-up, not preregistered V2 evidence. Fashion-MNIST is not packaged.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\followup_evidence.py --config configs\followup_evidence.json --run-id followup-evidence-NEW-UNIQUE-ID
.venv\Scripts\python.exe scripts\followup_evidence.py --summarize-dir results\followup_evidence\followup-evidence-NEW-UNIQUE-ID
```

Requires the original V2 run, its 60 checkpoints, and the balanced-probe post-hoc artifacts at the paths in the config.
