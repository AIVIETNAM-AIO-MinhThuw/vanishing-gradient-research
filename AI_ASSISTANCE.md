# Ghi nhận hỗ trợ AI

AI đã hỗ trợ trong phạm vi sau:

- Khởi tạo cấu trúc repository và hướng dẫn chạy local/Kaggle.
- Viết MLP tối thiểu và instrumentation smoke test bằng PyTorch.
- Viết các kiểm tra shape, finite value, backward, chain rule, finite differences, singular values và numerical rank.
- Viết test tự động và tài liệu giải thích để người nghiên cứu tự kiểm tra.
- Tham khảo có chọn lọc nguyên tắc experiment protocol, reality gate và local-to-server từ repository `ngtiendong/Academic-Research-Agent-Skill`; không clone repository đó.

AI không viết báo cáo nghiên cứu, không tạo kết quả để giả làm thực nghiệm, không lựa chọn hypothesis/metric khoa học thay người nghiên cứu và không triển khai Phase 1/full experiment.

Ngày 14/10/2026, sau verdict `MODIFY` từ một phản biện độc lập, AI đã hỗ trợ ghi protocol pilot v1 trước khi có run mới; triển khai các metric gradient gap, normalized gap, layer-index slope, normalized directional gain và Jacobian spectrum; thêm cơ chế chống ghi đè, subset manifest và unit tests. Các metric bổ sung được ghi rõ là supplementary/exploratory. AI không chạy pilot, không thay đổi kết quả smoke test và không tạo kết luận khoa học.

Sau khi người nghiên cứu phê duyệt ngân sách 2700 giây và seeds 42, 123, 2026, AI đã cài dependency, kiểm tra Fashion-MNIST, chạy unit tests, benchmark CPU và thực thi feasibility pilot `pilot-v1-20261014T1920`. AI chỉ lưu và tóm tắt các quan sát đo được; diagnostic gain=0.5 và Jacobian spectrum vẫn được phân loại supplementary/exploratory, không phải primary contrast hay kết luận khoa học.

Mọi kết quả trong `results/` phải được xem là artifact kỹ thuật của đúng lệnh chạy đã ghi, không phải kết luận khoa học. Người nghiên cứu chịu trách nhiệm xác nhận thiết kế pilot, metric, ngưỡng chấp nhận và diễn giải.

Ngày 14/10/2026, AI hỗ trợ triển khai đường chạy v2 đã đóng băng, split và batch manifest tái lập, instrumentation gradient và Tanh saturation, lưu checkpoint, Jacobian có điều kiện, unit tests, benchmark, export artifact và thực thi có giới hạn. AI không chọn kết luận, không điều chỉnh cấu hình sau khi xem kết quả và không viết báo cáo nghiên cứu.

Sau kiểm toán độc lập phát hiện gradient/Jacobian probes v2 đều thuộc class 0, AI hỗ trợ một phép đo bổ sung hậu nghiệm trên probe cân bằng nhãn bằng cách chỉ đọc 60 checkpoint weights đã lưu. Phép đo này không chạy training, không sửa v2, không được mô tả như preregistered và không thay người nghiên cứu diễn giải kết quả.

Ngày 14/10/2026, AI hỗ trợ follow-up evidence task được người nghiên cứu phê duyệt sau khi đã xem V2: đo parameter-gradient và net checkpoint displacement từ 60 checkpoint có sẵn; ghép accuracy/loss với balanced-probe gradients; và chạy hai đối chứng D=12 đã đóng băng trước lượt chạy (Tanh Xavier gain=0.5, Sigmoid Xavier gain=1.0). Artifact được gắn nhãn follow-up, không thay thế V2, không được coi là preregistered V2 và không chứa kết luận khoa học thay người nghiên cứu.

AI tiếp tục hỗ trợ phân tích mô tả RQ2 trên 90 checkpoint đã có, không training: ghép validation metrics, activation-gradient, per-probe gap và parameter-gradient; tái đo read-only các metric còn thiếu cho checkpoint follow-up; tạo biểu đồ theo từng seed và sensitivity analysis được gắn nhãn exploratory. AI không chọn ngưỡng confirmatory, không tuyên bố nhân quả và không viết nội dung báo cáo nộp.

