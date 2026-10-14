# Ghi nhận hỗ trợ AI

AI đã hỗ trợ trong phạm vi sau:

- Khởi tạo cấu trúc repository và hướng dẫn chạy local/Kaggle.
- Viết MLP tối thiểu và instrumentation smoke test bằng PyTorch.
- Viết các kiểm tra shape, finite value, backward, chain rule, finite differences, singular values và numerical rank.
- Viết test tự động và tài liệu giải thích để người nghiên cứu tự kiểm tra.
- Tham khảo có chọn lọc nguyên tắc experiment protocol, reality gate và local-to-server từ repository `ngtiendong/Academic-Research-Agent-Skill`; không clone repository đó.

AI không viết báo cáo nghiên cứu, không tạo kết quả để giả làm thực nghiệm, không lựa chọn hypothesis/metric khoa học thay người nghiên cứu và không triển khai Phase 1/full experiment.

Ngày 14/10/2026, sau verdict `MODIFY` từ một phản biện độc lập, AI đã hỗ trợ ghi protocol pilot v1 trước khi có run mới; triển khai các metric gradient gap, normalized gap, layer-index slope, normalized directional gain và Jacobian spectrum; thêm cơ chế chống ghi đè, subset manifest và unit tests. Các metric bổ sung được ghi rõ là supplementary/exploratory. AI không chạy pilot, không thay đổi kết quả smoke test và không tạo kết luận khoa học.

Mọi kết quả trong `results/` phải được xem là artifact kỹ thuật của đúng lệnh chạy đã ghi, không phải kết luận khoa học. Người nghiên cứu chịu trách nhiệm xác nhận thiết kế pilot, metric, ngưỡng chấp nhận và diễn giải.

