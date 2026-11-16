# okx-lab

Khung kiểm chứng chiến lược futures OKX (USDT-margined perpetual). Không dùng API key.

```
python -m okxlab download --inst BTC-USDT-SWAP --bar 1H --start 2022-01-01
python -m okxlab validate --inst BTC-USDT-SWAP --bar 1H
python -m okxlab backtest --strategy donchian --params entry_n=55,exit_n=20 --sensitivity entry_n
python -m okxlab walkforward --strategy donchian --train-days 365 --test-days 90
python -m pytest -q
```

Dữ liệu BTC-USDT-SWAP hiện có: nến 1H và 15m, mark price 1H và 15m, funding (từ 2022-01-16).
Tải mark price: `python -m okxlab download --bar 15m --start 2022-01-01 --mark`; dùng khi backtest: `--use-mark`
(thanh lý theo mark price, stop theo last price).

## Đảm bảo
- Dữ liệu: bỏ nến chưa đóng, kiểm tra trùng/khoảng trống/OHLC sai/funding thiếu, `data/manifest.json` lưu sha256.
- Funding dài hạn lấy từ file tháng của OKX (API lịch sử chỉ giữ ~3 tháng).
- Khớp lệnh ở *mở cửa nến kế tiếp*; slippage chỉ dùng thông tin nến trước; test chống look-ahead trong `tests/`.
- Chi phí: phí taker, slippage, funding (đúng chiều), thanh lý theo giá xấu nhất trong nến, stop (gap khớp ở giá mở).

## Giới hạn đã biết (chưa mô hình hóa)
- Lot size/min size, tier margin theo kích thước vị thế (mmr cố định 0.5%), hàng đợi lệnh limit, độ sâu sổ lệnh.
- Funding giữa nến dùng giá mở; nến 1H nên nếu dùng bar ≥1D funding bị xấp xỉ.
- `realized` funding trước ~3 tháng gần nhất lấy bằng `rate` (file tháng không có realizedRate).
- Phí taker mặc định 0.05%: sửa `--taker-fee` theo tier thật của tài khoản.
- Tối ưu tham số trên train rồi chọn "tốt nhất" luôn có thiên lệch; xem `trials_total` trong walk-forward.
