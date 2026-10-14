# Technical run summary — experiment v2

- Run ID: `experiment-v2-20261014T2053`
- Classification: bounded AIO2026 main experiment; no scientific conclusion attached.
- Frozen protocol: `experiment_protocol_v2.json`
- Git revision: `4356d5deb3d692ee3da471e11b1b7c7c3f4e8998`
- Device: CPU
- Primary grid: 4 depths × 3 seeds = 12 completed runs.
- Checkpoints: 0, 25, 100, 300, 600; 60 checkpoint weight files.
- Data: 10,000 training and 2,000 validation images, disjoint and stratified from official train.
- Official test: unused.
- Fixed gradient/saturation probes: 16.
- Conditional Jacobian: 27 snapshots, 8 probes and 3 representative layers each.
- Internal runtime: 34.1801379001699 seconds; approved ceiling: 1,200 seconds.
- Recorded run errors: 0.
- Post-run audit: PASS; no checked NaN/Inf, checkpoint hash failure, or directional-gain exclusion.

This file records execution facts only. It does not interpret the measurements or assert a research conclusion.
