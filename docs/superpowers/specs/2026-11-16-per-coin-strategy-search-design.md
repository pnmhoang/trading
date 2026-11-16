# Thiết kế: Tìm chiến lược riêng cho từng coin (BTC, ETH) bằng thư viện chiến lược + walk-forward

Ngày: 2026-11-16 · Trạng thái: chờ duyệt

## 1. Ý định và lý do
Người dùng muốn một chiến lược futures OKX (BTC-USDT-SWAP, ETH-USDT-SWAP) **ít lệnh nhưng ổn định**, mỗi lệnh rủi ro đúng **1% tài khoản**, R:R 1:2–1:3, chỉ cần win rate ≥ 40%. **Mỗi coin có chiến lược riêng.** Họ muốn lấy điểm mạnh của các repo đã đọc (passivbot, okx-trade, Quant-AI-Research) và backtest rất nhiều lần để chọn phương án tối ưu.

Lý do thiết kế xoay quanh một rủi ro: càng thử nhiều thì càng dễ chọn trúng kết quả đẹp do may rủi. Vì vậy phần quan trọng nhất là **cách chọn** (kiểm định, hiệu chỉnh theo số lần thử, test đóng băng), không phải số lần chạy.

Kết quả đã có: Donchian/EMA mẫu trên BTC 1H và 15m lỗ 75–87%, walk-forward không có edge. Nghĩa là "không tìm được gì" là một kết quả có thể xảy ra và được chấp nhận.

## 2. Mục tiêu
- Với mỗi coin, ra **một đặc tả chiến lược đóng băng** (họ chiến lược + tham số + khung thời gian) kèm báo cáo, hoặc kết luận trung thực "không có chiến lược nào đạt".
- Mỗi lệnh: SL theo ATR, TP cố định 2R hoặc 3R, khối lượng tính để mất 1% vốn nếu chạm SL.
- Mọi lần thử đều được ghi lại (sổ thí nghiệm) để hiệu chỉnh theo số lần thử.
- Engine tìm kiếm nhanh (numba), đối chiếu được với engine Python hiện có.

## 3. Không phải mục tiêu
- Bot giao dịch thật, kết nối API có key, quản lý lệnh live.
- Học máy (phương án C), open interest, long/short ratio, funding carry, dữ liệu spot.
- Ép đủ 2–3 lệnh mỗi ngày.
- Tối ưu tổng lợi nhuận; mục tiêu là kỳ vọng dương sau chi phí và ổn định.

## 4. Điều làm thiết kế thất bại (anti-goals)
- Chỉnh chiến lược sau khi đã nhìn kết quả test đóng băng.
- Chọn "tổ hợp đẹp nhất" mà không qua hiệu chỉnh theo số lần thử.
- Backtest lạc quan hơn thực tế (khớp ở giá không có thật, bỏ qua phí/funding).
- Chiến lược chỉ có lãi nhờ vài lệnh may mắn (ít lệnh, hoặc lãi tập trung ở một giai đoạn).

## 5. Ràng buộc
- Dữ liệu: BTC đã có (nến 1H, 15m, mark price, funding từ 2022-01-16). **Cần tải ETH-USDT-SWAP** cùng bộ dữ liệu. Toàn bộ là perpetual swap USDT-margined của OKX.
- Engine hiện có (`okxlab/backtest.py`) chưa có TP, chưa tính khối lượng theo rủi ro; phải mở rộng.
- Chi phí mặc định: taker 0,05%, slippage 2 bps + 0,05 × biên độ nến trước, funding thật, thanh lý theo mark price.

## 6. Các phương án đã cân nhắc
- **A. Thư viện chiến lược nhỏ + walk-forward từng coin — ĐƯỢC CHỌN.** Ít tham số nên ít overfit, giải thích được.
- B. Tìm kiếm rộng/di truyền (passivbot): tìm được nhiều hơn nhưng số lần thử rất lớn, dễ tự lừa mình. Loại.
- C. Học máy dự đoán chạm TP hay SL trước: khớp cấu trúc R:R nhưng dễ rò rỉ, phức tạp. Để lại nếu A thất bại.

## 7. Thiết kế
### 7.1 Họ chiến lược (mỗi họ ≤ 3 tham số, tín hiệu chốt ở đóng cửa, vào ở mở cửa nến kế tiếp)
1. **Trend breakout**: phá Donchian N nến theo hướng của bộ lọc EMA khung 4H.
2. **Trend pullback**: trong xu hướng (EMA nhanh > chậm), vào khi giá hồi về EMA và RSI hồi phục.
3. **Range mean-reversion**: ADX thấp, vào ngược tại biên Bollinger/range, TP/SL cố định.
4. **Volatility squeeze breakout**: băng Bollinger co hẹp dưới phân vị thấp rồi phá ra.

Khung vào lệnh là tham số: 15m hoặc 1H; bộ lọc xu hướng dùng khung cao hơn.

### 7.2 Quy tắc lệnh
- SL = k × ATR (k là tham số); TP = 2R hoặc 3R (tham số rời rạc).
- Khối lượng: `notional = 1% × equity / sl_distance_pct`. Bỏ qua lệnh nếu đòn bẩy cần thiết > 3x, hoặc nếu `chi phí khứ hồi / khoảng cách SL > 0,15R`.
- Một vị thế mỗi coin tại một thời điểm. Có thể thêm time-stop (tham số).
- Trong cùng một nến chạm cả SL và TP: **coi là chạm SL trước** (thận trọng).
- TP chỉ khớp khi giá đi qua mức TP (không phải chỉ chạm); SL là lệnh market có slippage, gap thì khớp ở giá mở.
- Tài khoản dùng chung cho BTC và ETH: tối đa 2 vị thế cùng lúc (rủi ro mở tối đa 2%, BTC–ETH tương quan cao nên thực tế gần một bet). Dừng giao dịch trong ngày nếu lỗ ngày ≥ 3% vốn.

### 7.3 Quy trình chọn (cho từng coin, độc lập)
1. **Chia dữ liệu theo thời gian**: walk-forward trên 2022-01-16 → 2025-06-30; **test đóng băng 2025-07-01 → nay**, mỗi coin chỉ được chạm một lần.
2. **Lưới nhỏ** mỗi họ (khoảng ≤ 200 tổ hợp), chạy walk-forward (train 12 tháng / test 3 tháng, trượt).
3. **Bộ lọc bền vững** (tất cả phải đạt): ≥ 80 lệnh trong giai đoạn walk-forward; win rate ≥ 40%; kỳ vọng sau chi phí ≥ +0,15R/lệnh; profit factor ≥ 1,3; lãi trong ≥ 60% cửa sổ test; max drawdown ≤ 20%; không lãi chỉ nhờ một giai đoạn (dương trong ít nhất 2 trong 3 pha: 2022 giảm, 2023 đi ngang/hồi phục, 2024 tăng); độ nhạy ±20% tham số không làm kỳ vọng âm.
4. **Hiệu chỉnh theo số lần thử**: tính deflated Sharpe và kiểm bootstrap (reality check) với tổng số tổ hợp đã thử; không đạt thì loại.
5. **Test đóng băng**: chạy đúng một lần cho ứng viên tốt nhất mỗi coin. Đạt bộ lọc kỳ vọng ≥ +0,1R và win rate ≥ 38% mới được gọi là "ứng viên".
6. Chẩn đoán phụ (không bắt buộc đạt): chạy chiến lược của coin này trên coin kia để xem độ tổng quát.
7. Ứng viên đi tiếp sang demo trading của OKX ít nhất 4–8 tuần trước khi có tiền thật (ngoài phạm vi thiết kế này).

### 7.4 Thành phần
- `okxlab/search/engine_nb.py`: engine numba (chỉ cho tìm kiếm). Engine Python hiện có là chuẩn đối chiếu; test bắt buộc: kết quả hai engine khớp trên các tổ hợp ngẫu nhiên.
- `okxlab/search/families.py`: bốn họ chiến lược và lưới tham số.
- `okxlab/search/ledger.py`: ghi mỗi lần thử vào JSONL kèm hash dữ liệu và hash code (học từ Quant-AI-Research).
- `okxlab/search/select.py`: bộ lọc bền vững, deflated Sharpe, reality check.
- `okxlab/search/report.py`: báo cáo mỗi coin + file đặc tả đóng băng (JSON).
- Mở rộng `backtest.py`: TP, khối lượng theo rủi ro, bỏ lệnh khi chi phí/R quá cao, giới hạn lỗ ngày.

## 8. Việc để người xây quyết định
Khoảng giá trị cụ thể của tham số trong lưới, bố cục bộ nhớ của engine numba, định dạng báo cáo, tên file.

## 9. Các quyết định tôi tự đặt (cần người dùng kiểm tra)
- Ngày chia test đóng băng 2025-07-01 và walk-forward 12 tháng / 3 tháng.
- Ngưỡng bộ lọc (≥ 80 lệnh, +0,15R, PF 1,3, DD 20%, ≥ 60% cửa sổ lãi).
- Giả định SL-trước khi cùng nến chạm cả hai; đòn bẩy tối đa 3x; trần chi phí 0,15R.
- Tài khoản dùng chung BTC+ETH, tối đa 2 vị thế, dừng ngày khi lỗ 3%.
- **Bỏ yêu cầu "chiến lược phải chạy được trên cả BTC lẫn ETH"** vì bạn muốn mỗi coin một chiến lược riêng; thay bằng tiêu chí "dương trong ≥ 2 trong 3 pha thị trường".
- Phí taker 0,05% (cần kiểm tra tier thật). TP tính theo phí maker 0,02% hay taker: để người xây chọn thận trọng (taker) trừ khi bạn xác nhận.
- Không commit/push tài liệu này lên git cho đến khi bạn đồng ý.
