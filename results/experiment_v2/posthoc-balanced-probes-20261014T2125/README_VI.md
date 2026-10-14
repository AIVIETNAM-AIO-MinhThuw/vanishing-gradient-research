# Phép đo bổ sung hậu nghiệm: balanced probes

Phân loại: `post_hoc_exploratory_measurement`. Đây không phải protocol V2 đã đăng ký trước, không phải V3 training experiment và không chứa kết luận khoa học.

## Vấn đề nguồn

`stratified_split()` của V2 nối validation indices theo từng class. V2 lấy `validation_indices[:16]` làm gradient probes và tám phần tử đầu làm Jacobian probes. Histogram quan sát trực tiếp là `{0:16}` và `{0:8}`; 960/960 gradient probe rows trong raw V2 có `target_class=0`.

Điều này ảnh hưởng các summary dựa trên fixed probes: gradient pre/post activation, gap, normalized gap, slope, Tanh saturation, logit-loss gradient và Jacobian. Training, weights, batch order, toàn bộ training/validation loss và accuracy trên 10.000/2.000 ảnh không bị ảnh hưởng.

## Phép đo bổ sung

- Seed chọn probe: `20261014`.
- Gradient: 40 probes, đúng bốn ảnh cho mỗi class 0–9.
- Jacobian: 20 probes, đúng hai ảnh cho mỗi class và là tập con của 40 probes.
- 60 checkpoint weights được đọc đúng một lần; không training.
- Gradient: 60 checkpoint × 40 probes = 2.400 sample-checkpoint measurements.
- Jacobian: 27 checkpoint × 20 probes = 540 sample-checkpoint measurements, 1.620 probe-layer rows.
- Runtime: khoảng 25,06 giây.

Verification: `PASS`; không NaN/Inf, không directional-gain exclusion, không probe giao training subset, official test không được dùng, artifact V2 nguồn giữ nguyên hash.

Các cờ thay đổi số học dùng ngưỡng mô tả đóng băng trước đo (`0,05` cho gap/slope; `20%` symmetric relative delta cho Jacobian). Chúng không phải kiểm định thống kê.

