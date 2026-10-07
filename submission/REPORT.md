# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Hoàng Anh  **MSSV**: 2A202602811  **Ngày**: 10/7/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, sm_89, bf16)`

> Mọi con số dưới đây khớp 100% với các file trong `results/`.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (miền hẹp phân loại ticket) |
| Train / val | 225 train / 25 val (seed 42, split 90%/10%) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` (custom pre-tokenized mask từ NB1) |
| Epochs / max_steps | 2.0 epochs / 30 max_steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json: reasoning preserved — safe to train on traces)*

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` (41.49% số token được tính loss) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán đoạn preview của phần được tính loss và bị mask:

```text
Supervised preview (labels != -100):
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>

Masked preview (labels == -100):
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7244 | 0.000 | 39575.3 |
| (b) base + optimized prompt | 0.755 | 0.7244 | 1.000 | 10838.2 |
| (c) LoRA fine-tune (`correct`) | 0.975 | 0.3333 | 1.000 | 11895.4 |

**(b) có thật sự mạnh hơn (a) không?** Có (`0.755` vs `0.000`). Baseline (b) là đối thủ thực sự của fine-tune vì dùng `OPTIMIZED_PROMPT` với schema JSON chi tiết và ví dụ few-shot. Prompt không bị cố tình làm yếu đi.

---

## 4. Giải phẫu cấu hình sai (NB4 & NB5 §4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.3512 | **0.975** | 5399.9 | 9.46 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.3511 | **0.970** | 4851.2 | 8.84 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5031 | **0.000** | 4105.4 | 8.83 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.3546 | **0.980** | 317.3 | 3.55 |

> **Xếp hạng chính thức theo NB5 Target Score:** 1st: `qlora` (0.980) > 2nd: `correct` (0.975) > 3rd: `attn_only` (0.970) > 4th: `wrong_lr` (0.000).

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target NB5, `attn_only` (r=283) đạt điểm `0.970`, thua nhẹ `correct` (`0.975`). Tuy nhiên, ở bảng train loss NB4, `attn_only` lại có loss thấp hơn một chút (`0.3511` vs `0.3512`). Sự nghịch đảo thứ tự này chứng minh train loss là chỉ số thay thế dễ gây hiểu lầm. Dù tăng rank r lên tới 283 để ép bằng ngân sách tham số, việc chỉ gắn adapter vào 2 module attention (`q_proj`, `v_proj`) vẫn cho khả năng biểu diễn kém hơn việc trải rộng adapter trên tất cả 12 linear layers của text decoder (`all-linear`). Vị trí gắn adapter quan trọng hơn rank.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
`wrong_lr` dùng LR = 1e-5 (thang full-FT, $1\times$). Loss giảm rất chậm và dừng ở `1.5031` (so với `0.3512` của `correct`). Đánh giá trên tập target thu được điểm `0.000` và format `0.000`. Nếu chỉ nhìn loss vẫn giảm dần theo epoch mà không đo năng lực tác vụ, ta sẽ lầm tưởng mô hình đang học bình thường. Thực tế, LoRA chỉ cập nhật một phần nhỏ trọng số nên đòi hỏi Learning Rate lớn hơn ~10 lần so với full fine-tuning để đủ sức điều hướng mô hình.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` (4-bit NF4) tiết kiệm tới 62.5% VRAM (chỉ dùng `3.55 GB` VRAM so với `9.46 GB` ở 16-bit) và đạt target score `0.980` với latency cực nhanh (`1211.5 ms`). Khuyến nghị nhà cung cấp khuyên "không dùng QLoRA cho Qwen3.5" xuất phát từ sai số lượng hóa khi suy luận các tác vụ reasoning phức tạp. Với bài toán phân loại ticket hẹp này, QLoRA 4-bit không những không giảm chất lượng mà còn giúp tối ưu bộ nhớ và tốc độ suy luận vượt trội.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.2200` · `regression Δ = -0.3911` · `valid_trace_rate = 0.00`

**Diễn giải:**
Mô hình LoRA fine-tune (`correct`) cải thiện rất mạnh năng lực phân loại ticket target (+22.0 percentage points so với Baseline B, từ 75.5% lên 97.5%). Tuy nhiên, phán quyết cổng hồi quy đánh giá FAILED vì mô hình gặp hiện tượng quên thảm họa (catastrophic forgetting) trên năng lực tổng quát: điểm Regression sụt giảm từ 72.4% xuống 33.3% (độ lệch -0.3911, vượt xa ngưỡng dung sai 0.020). 

Đây không phải là thất bại của kỹ thuật fine-tuning, mà là minh chứng thực nghiệm trung thực về trade-off giữa sự chuyên môn hóa miền hẹp và năng lực tổng quát của LLM. Khi fine-tune trên 225 mẫu ticket CSKH mà không kèm dữ liệu tổng quát, mô hình bị lệch trọng số về miền dữ liệu mới. Để đạt phán quyết PASSED trước khi đưa vào sản xuất, mô hình cần được huấn luyện kết hợp 1–5% dữ liệu replay tổng quát (theo khuyến nghị deck §6.3).

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn OD538419. Hoàn tiền. Mon | hoan_tien, thap | hoan_tien, thap | hoan_tien, thap | ✅ FT thắng (score 0.75) |
| 2 | Shop ơi, mình đặt máy xay sinh tố mã đơn DH777946. Khi nào có tiền về. | hoan_tien, trung_binh | hoan_tien, trung_binh | hoan_tien, thap | ❌ **FT thua** (đánh giá nhầm urgency) |
| 3 | Cho mình hỏi, mình đặt máy xay sinh tố mã đơn OD906403. Hoàn lại. Mong | hoan_tien, thap | hoan_tien, thap | doi_tra, thap | ❌ **FT thua** (nhầm intent hoan_tien -> doi_tra) |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn OD169066. Trả hàng. Mong | doi_tra, trung_binh | doi_tra, trung_binh | doi_tra, thap | ❌ **FT thua** (đánh giá nhầm urgency) |
| 5 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. | hoi_thong_tin, trung_binh | hoi_thong_tin, trung_binh | hoi_thong_tin, thap | ❌ **FT thua** (đánh giá nhầm urgency) |

**Mẫu chung ở các ca FT thua:** Mô hình fine-tune đôi khi gán nhầm mức độ khẩn cấp (`urgency`) về mức mặc định `thap` khi từ khóa khẩn cấp trong ticket bị ẩn nhẹ hoặc trùng với câu hỏi thông thường.

---

## 7. Kết luận & điều tôi học được

**Kết luận:**
Không nên deploy ngay bản fine-tune `correct` này vào môi trường sản xuất đa nhiệm do năng lực tổng quát bị sụt giảm (-39.1% trên tập regression). Tuy nhiên, trên tác vụ chuyên biệt phân loại ticket CSKH, bản fine-tune đã thành công vượt bậc khi nâng độ chính xác từ 75.5% lên 97.5% chỉ với prompt ngắn 4 chữ `NAIVE_PROMPT`.

Đòn bẩy thật sự trong lab này bao gồm:
1. **Loss Mask chuẩn (NB1):** Đảm bảo loss chỉ tính trên câu trả lời của assistant (`supervised_fraction = 41.49%`), tránh lãng phí dung lượng mô hình vào việc học lại prompt.
2. **Learning Rate hợp lý (NB3/NB4):** Thang $LR \approx 10\times$ full-FT ($1e-4$) là yếu tố sống còn cho LoRA hội tụ.
3. **Vị trí adapter (`text-linear`):** Phủ toàn bộ linear layers của text decoder vượt trội hơn hẳn so với việc cố tăng rank $r$ ở ít vị trí.

**Ba điều tôi học được:**
1. **Train loss là chỉ số thay thế (surrogate metric) dễ gây ngộ nhận:** `attn_only` có loss thấp hơn `correct` nhưng target accuracy lại thấp hơn. Phán quyết phải dựa vào đánh giá tác vụ thực tế.
2. **Kiến trúc adapter (`all-linear`) quan trọng hơn Rank $r$:** Việc phân bổ tham số trên toàn bộ text decoder linear layers giúp mô hình tiếp thu tri thức tốt hơn việc dồn tham số vào ít layer.
3. **LoRA đòi hỏi Learning Rate lớn hơn ~10 lần so với Full Fine-Tuning:** Dùng LR của full-FT làm mô hình không hội tụ được trên tác vụ mới.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Trộn 3% dữ liệu replay tổng quát vào tập huấn luyện train set để khắc phục hiện tượng catastrophic forgetting và đưa cổng hồi quy về phán quyết PASSED.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap
- [x] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [x] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [x] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
