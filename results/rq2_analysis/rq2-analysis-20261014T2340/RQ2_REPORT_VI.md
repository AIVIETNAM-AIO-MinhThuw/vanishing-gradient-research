# Phân tích bổ sung RQ2 — không phải nội dung báo cáo nộp

**Trạng thái phạm vi bằng chứng: OBSERVED_DESCRIPTIVELY.**

Trạng thái này chỉ có nghĩa: trong dữ liệu checkpoint đã đo, có trường hợp validation accuracy tăng so với initialization của chính run trong khi mean per-probe gap dương (`G_first < G_last`). Đây là quan sát mô tả, không phải ngưỡng “accuracy tốt”, không phải causal claim và không phải kết luận rằng gradient tuyệt đối nhỏ.

## Nguồn dữ liệu

- V2: validation loss/accuracy gốc, 60 checkpoint.
- Balanced post-hoc: 40 probes cân bằng nhãn, cung cấp activation-gradient và per-probe gap cho V2.
- Follow-up: 30 checkpoint của hai arm; per-probe gap và parameter-gradient được tái đo read-only từ weights.

## Hai câu hỏi tách biệt

1. **Suy giảm tương đối:** dùng dấu gap, không cần ngưỡng tùy ý. Có 29 checkpoint V2 và 0 checkpoint follow-up thỏa đồng thời `accuracy > accuracy_at_step_0` và `gap > 0`.
2. **Giá trị tuyệt đối nhỏ:** dữ liệu cung cấp RMS thực tế nhưng protocol không định nghĩa scale nào là “nhỏ”. Vì vậy nhánh này vẫn **INCONCLUSIVE**. File sensitivity chỉ minh họa kết quả thay đổi theo nhiều cặp ngưỡng và không được dùng cho verdict.

## Các quan sát không phù hợp với giả thuyết đơn giản

- Nhiều checkpoint V2 muộn có accuracy cao hơn nhưng gap âm, tức layer đầu không suy giảm tương đối so với layer cuối theo metric này.
- Hai follow-up arm có gap dương rất lớn và first-layer RMS rất thấp, nhưng validation accuracy không cải thiện có ý nghĩa so với mức ban đầu/chance.
- Gradient gap không thay thế absolute gradient, và parameter-gradient không đồng nhất với activation-gradient.

## Giới hạn

Chỉ mô tả các checkpoint, seeds và arms đã có. Không có ngưỡng confirmatory cho “accuracy tương đối tốt” hay “gradient tuyệt đối nhỏ”; không kiểm định nhân quả, novelty hoặc khả năng tổng quát hóa.
