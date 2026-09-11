# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU (CUDA)

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cu128 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Đặt `KHOA = "K4"` ở ô lưu bài; không thay đổi mô hình, ngưỡng dự đoán hoặc logic tạo bằng chứng.

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `468`, `cab`, `1`, `0.510915`, `ImageNet-1K`.
- Record này mô tả toàn ảnh như thế nào? Model xem toàn bộ ảnh `traffic` là một ảnh thuộc lớp `cab`; đây là prediction có điểm cao nhất trong top-5, không phải nhãn cho riêng một chiếc xe trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Bộ taxonomy dùng để huấn luyện checkpoint định nghĩa class list; với `yolo11n-cls.pt` trong notebook này là ImageNet-1K.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? `class_id` là mã ổn định để máy xử lý, `class_name` giúp con người đọc, còn `taxonomy_name` xác định không gian lớp và ý nghĩa của ID. Cùng một ID có thể mang nghĩa khác trong taxonomy hoặc phiên bản khác.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Cần quy định rõ bài toán là single-label hay multi-label; nếu single-label thì phải nêu tiêu chí chọn chủ thể chính (diện tích, vị trí, mục đích ảnh), cách xử lý đồng hạng và khi nào đánh dấu mơ hồ/escalate.
- Vì sao model score không phải ground truth? Score chỉ biểu diễn mức tin cậy của mô hình đối với prediction theo checkpoint hiện tại; mô hình có thể dự đoán sai hoặc bỏ sót. Ground truth phải do con người gán và kiểm tra theo guideline.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): `person`, `0.912625`, `[385.33, 69.24, 498.92, 348.92]`, `113.58`, `279.68` pixel.
- Diễn giải vị trí box bằng lời: Box bao người ở phía bên phải ảnh, từ góc trên-trái `(385.33, 69.24)` đến góc dưới-phải `(498.92, 348.92)`; rộng khoảng 113.58 pixel và cao khoảng 279.68 pixel. Gốc tọa độ ảnh nằm ở góc trên-trái.
- So sánh số prediction ở hai threshold: Với sample `kitchen`, ngưỡng `0.20` có 17 prediction, ngưỡng `0.35` có 11 prediction và ngưỡng `0.60` có 6 prediction. So sánh hai đầu, giảm từ `0.60` xuống `0.20` làm tăng thêm 11 prediction.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Ngưỡng thấp thường tăng recall/độ bao phủ và giảm nguy cơ bỏ sót, nhưng tạo thêm false positive nên reviewer phải xem nhiều box hơn. Ngưỡng cao giảm khối lượng review nhưng dễ bỏ sót vật thể thật.
- Đề xuất một quy tắc box chặt: Vẽ hình chữ nhật trục song song nhỏ nhất bao hết phần nhìn thấy của vật thể, bám sát các điểm ngoài cùng, không lấy nền hoặc bóng; không cắt vào vật thể và clip box tại biên ảnh.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline phải quy định gán phần nhìn thấy hay suy diễn toàn bộ vật thể, mức che khuất/cắt mép tối thiểu vẫn được gán, cách đặt box/thuộc tính occluded-truncated và ngưỡng mơ hồ phải chuyển reviewer/mentor quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): `kitchen-001`, `person`, `0.899318`, 348 điểm; tám điểm đầu là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0], [439.0, 73.0], [438.0, 74.0]]`.
- Polygon bổ sung chi tiết gì so với box? Polygon mô tả đường biên và vùng pixel có hình dạng cụ thể của vật thể, nên loại được phần nền nằm trong box và thể hiện các chỗ lồi, lõm tốt hơn hình chữ nhật.
- `instance_id` dùng để làm gì và không phải loại ID nào? Nó định danh duy nhất từng cá thể trong output, giúp liên kết polygon/box với đúng instance và phân biệt hai vật thể cùng lớp. Nó không phải `class_id`, không biểu diễn loại vật thể và không phải danh tính thật của người/vật thể.
- Đề xuất một quy tắc biên mask: Mask chỉ bao các pixel nhìn thấy thuộc vật thể, bám sát biên ngoài, loại nền, bóng và các lỗ không thuộc vật thể; không gộp hai instance đang tiếp xúc và không tự suy diễn phần bị che khuất nếu guideline không yêu cầu amodal mask.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định độ mờ chấp nhận được, cách chọn biên khi chuyển tiếp không rõ, cách tách các instance tiếp xúc, có gán phần bị che hay không và ngưỡng bất định để annotator đánh dấu mơ hồ rồi chuyển reviewer/mentor.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một `class_id`/`class_name` cho toàn ảnh (hoặc danh sách lớp nếu guideline quy định multi-label), gắn với taxonomy | Ảnh có nhiều chủ thể; model chọn `cab` với score 0.510915 nhưng còn nhiều lớp gần nghĩa trong top-5 | Xem toàn ảnh, chọn nhãn theo taxonomy và quy tắc chủ thể chính; đánh dấu/escalate trường hợp mơ hồ | Kiểm tra đúng taxonomy, tính nhất quán single/multi-label, nhãn sai hoặc thiếu và không dùng score làm ground truth |
| Phát hiện vật thể | Một lớp và box `xyxy` theo pixel cho mỗi object | Ngưỡng thấp sinh thêm false positive; ngưỡng cao có thể bỏ sót; box ở vật thể che khuất/cắt mép dễ không nhất quán | Tìm mọi object thuộc phạm vi, gán đúng lớp, vẽ box chặt và thêm cờ che khuất/cắt mép nếu có | Kiểm tra object bị bỏ sót/thừa, lớp, box có chặt và hợp lệ, quy tắc che khuất/cắt mép |
| Instance segmentation | Một lớp, `instance_id` và polygon/mask riêng cho mỗi instance | Biên mờ, vật thể tiếp xúc/che khuất; polygon nhiều điểm làm tăng lỗi biên | Tách từng instance, vẽ mask bám phần nhìn thấy và đánh dấu vùng không chắc chắn | Kiểm tra đủ instance, không gộp/tách sai, mask không ăn nền hoặc thủng vật thể, polygon hợp lệ và nhất quán |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ dùng ảnh và output đúng phạm vi bài tập, lưu/chia sẻ trong nơi được phê duyệt; không đưa họ tên, MSSV, email, số điện thoại, thông tin đăng nhập hoặc dữ liệu nhạy cảm vào báo cáo/repository.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: mentor/giảng viên hoặc người phụ trách dữ liệu trước khi tiếp tục xử lý.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
