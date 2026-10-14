# Gói bàn giao kiểm toán pilot-v1

Trạng thái: `AGGREGATION_DIFFERENCE`.

Gói này được tạo hậu nghiệm sau khi pilot đã hoàn tất. Không có dữ liệu đo gốc nào được sửa, xóa hoặc chạy lại. Hash SHA-256 của các artifact nguồn được lưu trong `audit_summary.json` và `SOURCE_HASHES_SHA256.txt`.

## Giải thích Gradient Gap

- Bảng RMS theo layer dùng **trung bình số học** của RMS trên 16 fixed probes, riêng cho từng layer.
- Gap đã báo cáo và biểu đồ dùng **trung bình số học của gap tính riêng trên từng probe**: `mean_i[log10(G_last,i / G_first,i)]`.
- Vì log là phi tuyến, đại lượng trên tương đương log của tỷ số hai trung bình nhân, và nhìn chung không bằng `log10(arithmetic_mean(G_last) / arithmetic_mean(G_first))`.
- Ví dụ D=12, seed 42, step 100, post-activation: gap từ hai RMS trung bình số học là `0.13921095`; trung bình gap theo probe là `0.16585181`.

Không phát hiện lỗi đơn vị, nhầm checkpoint, nhầm loại gradient hoặc lệch tập probe. Tuy vậy, bảng/báo cáo tiếp theo phải ghi rõ estimator để tránh hiểu hai cách tổng hợp là một.

## Các bảng kiểm toán

- `gradient_aggregation_audit.csv`: so sánh mean/median RMS và mean/median gap cho mọi primary run, checkpoint và vị trí pre/post activation.
- `evaluation_metrics_audit.csv`: loss/accuracy và số mẫu đánh giá.
- `diagnostic_metrics_audit.csv`: loss/accuracy và kiểm tra lại gap của diagnostic gain=0.5.
- `sample_layer_counts.csv`: số probe và số layer.
- `jacobian_metrics_by_probe.csv`: 216 dòng mức probe-layer.
- `jacobian_metrics_summary.csv`: 54 dòng trung bình trên 4 spectrum probes cho mỗi arm/seed/checkpoint/layer được đo.
- `audit_summary.json`: toàn bộ check PASS/FAIL, metric có/không có và hash nguồn.

## Giới hạn

Jacobian chỉ được đo trên 4 probes và tại layer đầu/giữa/cuối. Bảng summary dùng trung bình số học, không phải uncertainty interval. Nullity 118 của ma trận 10x128 là do hình dạng; cờ thiếu rank bổ sung được tách riêng. Diagnostic gain=0.5 là supplementary control, không phải bằng chứng về vanishing gradient tự nhiên.

