# Báo cáo Lab Day 1 — MSSV 2A202602588

## 1. Thiết lập

- Môi trường của lần chạy notebook này: Windows, `.venv`, PyTorch 2.14.1+cpu; không dùng GPU.
- Dữ liệu Forest CoverType: train 464 809 mẫu và eval 116 203 mẫu theo metadata cố định. Tách val phân tầng 20% từ train với seed 42, được 371 847 mẫu train và 92 962 mẫu val. Chỉ thống kê train được dùng để chuẩn hoá; eval chỉ được chấm sau khi chọn cấu hình.
- Model bắt buộc M-base: 54→256→128→7, 47 879 tham số; cross-entropy, SGD+momentum 0,9, learning rate 0,1 được chọn bằng sàng lọc val, batch 512, 20 epoch, He, không dropout, FP32.
- Accuracy đoán lớp đa số trên val là 0,4876.
- Đã thử đủ bảy chủ đề: loss, optimizer, hyper-parameter, dropout, clipping, mixed precision và initialization.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / `(B, 7)` |
| Loss bước 0 (`base-s1`, so với ln 7 = 1,945910) | 2,269062; cao hơn 0,323152. Logits He khởi tạo có std 0,576533 nên phân phối lớp ban đầu không đều |
| Quá khớp 20 mẫu (phép kiểm tra chẩn đoán Part 1 trong notebook) | loss 2,480333 → 0,00001279; accuracy 0% → 100% sau 200 bước |
| Gradient mọi tham số | Từng tensor có gradient khác 0 |
| Baseline | 2 seed: `base-s1`, `base-s42` |
| Val accuracy | 0,906317 ± 0,001476 |
| Val macro-F1 | 0,851261 ± 0,000894 |

Ngưỡng tham chiếu `2σ=0,001787` được tính từ hai seed baseline trên val. Đây chỉ là ước lượng thô vì số seed nhỏ; các chênh lệch thấp hơn ngưỡng chưa đủ để khẳng định cải thiện.

## 3. Kết quả theo chủ đề

Mọi metric dưới đây lấy tại epoch có val loss thấp nhất. Các lần chạy dùng cùng phép tách val, seed 1 và 20 epoch, trừ yếu tố nêu rõ; kết luận lựa chọn cấu hình chỉ dựa trên val.

### 3.1 Hàm mất mát — CE và MSE

Dự đoán trước: CE có thể tối ưu phân loại hiệu quả hơn; loss MSE trên logits và one-hot có thang đo khác nên so bằng metric phân loại, không so trực tiếp trị số loss. `loss-mse` đạt val macro-F1 0,684368, thấp hơn baseline CE `base-s1` 0,166261, lớn hơn nhiều mốc 2σ. MSE lấy trung bình trên tất cả phần tử N×7. Quan sát phù hợp với dự đoán; ảnh: [compare_loss](figures/compare_loss.png).

### 3.2 Bộ tối ưu hoá

| Bộ tối ưu / lần chạy tốt nhất trong các LR đã thử | LR | Val macro-F1 | Epoch tốt nhất |
|---|---:|---:|---:|
| SGD + momentum (`base-s1`) | 0,1 | 0,850629 | 20 |
| Adam (`opt-adam-lr0003`) | 0,003 | 0,875523 | 20 |

Adam `lr=0,001` đạt 0,845331; SGD+momentum `lr=0,03` đạt 0,821168. So với baseline, Adam `lr=0,003` cao hơn 0,024894, vượt mốc 2σ. Kết quả ủng hộ Adam trong các cấu hình và LR đã đo, không chứng minh ưu thế phổ quát. Chỉ có hai LR cho mỗi họ optimizer; không chạy cặp cùng LR để kết luận về trường hợp không tinh chỉnh LR. Ảnh: [compare_optimizer](figures/compare_optimizer.png).

### 3.3 Hyper-parameter: batch size

`batch-2048` đạt val macro-F1 0,780408 so với 0,850629 của baseline. Batch lớn tạo khoảng 182 thay vì 727 cập nhật mỗi epoch; thời gian giảm từ 2,162 xuống 0,752 giây/epoch. Với cùng 20 epoch, số lần cập nhật ít hơn khoảng bốn lần, vì vậy thời gian/epoch thấp hơn không đồng nghĩa hội tụ tốt hơn. Ảnh: [compare_hparam](figures/compare_hparam.png).

### 3.4 Dropout

Baseline `base-s1` không thể hiện khoảng cách train–val loss lớn ở epoch chọn: gap khoảng 0,022238. `drop-03` làm gap còn 0,005853 nhưng macro-F1 giảm xuống 0,764495 (−0,086134 so với `base-s1`). Dropout thu hẹp gap nhưng có vẻ regularize quá mạnh trong cấu hình này; chưa có bằng chứng để dùng nó. Ảnh: [compare_dropout](figures/compare_dropout.png).

### 3.5 Gradient clipping

Trong lần chạy CPU, stress-test `lr=1` đạt macro-F1 0,785971 nếu không clip và 0,817119 với `c=0,55`. Trung bình grad norm theo epoch đều thấp hơn `c`, nhưng số đo này không xác nhận clipping có hay không kích hoạt ở từng batch; vì vậy không quy toàn bộ chênh lệch metric cho clipping. Kết quả `clip-lr1-c025` (0,816527; clipping trung bình 50,11% batch) đến từ một lần chạy CUDA trước đó, không phải cùng lượt CPU và không gộp thành so sánh đồng điều kiện. Ảnh chồng hiện tại: [compare_clipping](figures/compare_clipping.png).

### 3.6 Mixed precision

`amp-bf16` trên CPU đạt macro-F1 0,839892, thấp hơn `base-s1` 0,010737. Thời gian là 41,199 so với 2,162 giây/epoch; cả hai ghi peak GPU memory là 0 vì chạy CPU. BF16 chậm hơn rõ rệt ở môi trường này. Ảnh: [compare_amp](figures/compare_amp.png).

### 3.7 Khởi tạo

Std kích hoạt sau mỗi hidden ReLU trên 4096 dòng val: He `(0,38631; 0,38786)`, normal std 0,01 `(0,02062; 0,00215)`, zeros `(0; 0)`. `init-normal` đạt macro-F1 0,825609 (−0,025020, vượt 2σ); `init-zeros` đạt 0,093650 và accuracy 0,487597, gần đoán lớp đa số. Khởi tạo zero giữ các neuron đối xứng và ReLU(0) làm tín hiệu ẩn không hoạt động, nên mô hình không học được như He. He giữ phương sai phù hợp hơn khi truyền qua ReLU; Xavier thường cân bằng phương sai cho activation đối xứng như tanh. Ảnh: [compare_init](figures/compare_init.png).

## 4. Đánh giá cuối trên eval

Cấu hình cuối đã được chốt theo val trước khi chạy eval: `opt-adam-lr0003`, M-base, CE, Adam `lr=0,003`, batch 512, seed 1, 20 epoch, checkpoint epoch 20 theo val loss thấp nhất. `base-s1` là baseline được đánh giá đối chiếu. Mỗi cấu hình được huấn luyện tái lập từ seed/cấu hình đã lưu để khôi phục `best_state`; eval không tham gia lựa chọn.

| Cấu hình | Seed nộp | Val macro-F1 | Eval macro-F1 | Eval accuracy |
|---|---:|---:|---:|---:|
| Baseline (`base-s1`) | 1 | 0,850629 | 0,855398 | 0,905493 |
| Cuối (`opt-adam-lr0003`) | 1 | 0,875523 | 0,876952 | 0,915510 |

Eval macro-F1 tăng 0,021554 và accuracy tăng 0,010017 so với baseline. Mức tăng macro-F1 lớn hơn mốc 2σ val 0,001787, nhưng đây không phải ước lượng độ nhiễu eval: mỗi cấu hình chỉ có một seed eval. Val và eval gần nhau (chênh macro-F1 baseline +0,004769, cấu hình cuối +0,001429).

### 4.1 Phân tích lỗi theo lớp — `eval_result.json`

| Lớp | Support | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0 | 42 368 | 0,9247 | 0,8954 | 0,9098 |
| 1 | 56 661 | 0,9176 | 0,9389 | 0,9281 |
| 2 | 7 151 | 0,9157 | 0,9115 | 0,9136 |
| 3 | 549 | 0,8078 | 0,8342 | 0,8208 |
| 4 | 1 899 | 0,8093 | 0,7799 | 0,7943 |
| 5 | 3 473 | 0,8439 | 0,8408 | 0,8423 |
| 6 | 4 102 | 0,9158 | 0,9439 | 0,9297 |

Ma trận nhầm lẫn (hàng là nhãn thật, cột là dự đoán):

| Thật \ Dự đoán | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 37 935 | 4 061 | 7 | 0 | 43 | 11 | 311 |
| 1 | 2 874 | 53 201 | 133 | 0 | 271 | 137 | 45 |
| 2 | 0 | 173 | 6 518 | 83 | 19 | 358 | 0 |
| 3 | 0 | 0 | 62 | 458 | 0 | 29 | 0 |
| 4 | 30 | 353 | 30 | 0 | 1 481 | 5 | 0 |
| 5 | 4 | 140 | 368 | 26 | 15 | 2 920 | 0 |
| 6 | 179 | 50 | 0 | 0 | 1 | 0 | 3 872 |

Lớp 4 có F1 thấp nhất; trong 1 899 mẫu thật lớp 4, có 353 mẫu bị dự đoán thành lớp 1 và 30 mẫu thành lớp 0. Lớp 3 có ít support nhất (549), nhưng F1 của nó cao hơn lớp 4. Đây là các quan sát từ ma trận, không xác định nguyên nhân; sự tương đồng đặc trưng hoặc mất cân bằng là giả thuyết cần thí nghiệm riêng để kiểm tra.

## 5. Câu hỏi dẫn dắt

1. Trong LR đã thử, Adam ở 0,003 cho macro-F1 cao nhất; chưa đủ dữ liệu để kết luận khi cố định cùng một LR cho mọi optimizer. So công bằng hơn là chỉnh LR từng họ rồi so cấu hình tốt nhất.
2. Dropout 0,3 thu hẹp gap nhưng giảm macro-F1; với kết quả này chưa có căn cứ dùng dropout. Nên cân nhắc khi có overfit rõ và đánh giá trên val.
3. Clipping giới hạn norm gradient để tránh cập nhật quá lớn. Stress-test CPU cho thấy macro-F1 0,785971 không clip và 0,817119 với `c=0,55`, nhưng không lưu tỷ lệ batch thực sự bị cắt trong kết quả lần chạy này; không quy chắc chắn chênh lệch cho clipping.
4. BF16 không nhanh hơn FP32 trên kernel CPU này: khoảng 41,199 so với 2,162 giây/epoch. Kết luận chỉ áp dụng cho môi trường đã đo.
5. Khởi tạo zero làm các neuron cùng lớp có trọng số giống nhau, không phá đối xứng; các lớp ẩn ReLU không nhận tín hiệu học hữu ích. He điều chỉnh phương sai cho ReLU; Xavier hữu ích hơn với activation đối xứng. Kết quả zeros và normal ở `init-zeros`/`init-normal`.
6. Nếu loss không giảm sau 2 000 bước: (i) kiểm tra shape, dtype, nhãn và chuẩn hoá đầu vào/target; (ii) kiểm tra logits, loss nhận logits thô và kiến trúc/khởi tạo; (iii) kiểm tra vòng cập nhật: loss hữu hạn, gradient khác 0, `zero_grad`/optimizer đúng và tham số thực sự được cập nhật. Các kiểm tra này lần lượt phân biệt lỗi dữ liệu, mô hình và pipeline.

## 6. Hạn chế và điều bất ngờ

Ước lượng 2σ chỉ có hai seed baseline, còn eval mỗi cấu hình chỉ một seed; do đó các chênh lệch eval không mô tả biến thiên giữa seed. Các kết quả Part 3 của lượt này chạy trên CPU; `clip-lr1-c025` trong bảng là một kết quả CUDA từ lượt trước, không nên so trực tiếp như một cặp cùng thiết bị. Mọi thí nghiệm huấn luyện 20 epoch, nhưng batch size 2048 có ít cập nhật hơn mỗi epoch. LR của optimizer được thử ở số lượng hữu hạn; các so sánh chỉ áp dụng trong phạm vi đã đo. Lớp 4 có F1 thấp nhất dù lớp 3 hiếm nhất, một dấu hiệu cho thấy support thấp không phải yếu tố duy nhất quyết định lỗi; cần phân tích đặc trưng hoặc thử xử lý mất cân bằng nếu có thêm thời gian.

## 7. Phụ lục

Các kết quả có thể truy nguyên trong `experiments.xlsx`, `results/<exp_id>.json` và `figures/<exp_id>.png`; ảnh so sánh dùng tên `figures/compare_<group>.png`. Dự đoán nộp là `predictions_eval.csv`; điểm và ma trận cuối lấy từ `eval_result.json`. Notebook, các module và kết quả từng lần chạy nằm trong thư mục `code/`, `results/` và `figures/`.
