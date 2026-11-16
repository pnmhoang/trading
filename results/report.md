# Kết quả tìm chiến lược theo từng coin

## BTC

Số tổ hợp đã thử: 720 · Reality-check p = 0.888

| Họ chiến lược | Số tổ hợp | OOS walk-forward | Kết quả |
|---|---|---|---|
| trend_breakout | 144 | 147 lệnh, win 35%, kỳ vọng -0.04R, PF 0.95, maxDD -13.3% | Loại: win_rate 0.347 < 0.40; expectancy_r -0.035 < 0.15; profit_factor 0.95 < 1.3; window_win_pct 0.44 < 0.60; sensitivity fails (±20% params make expectancy negative); deflated Sharpe prob 0.00 < 0.9; reality check p 0.888 > 0.1 |
| trend_pullback | 192 | 73 lệnh, win 32%, kỳ vọng -0.09R, PF 0.87, maxDD -9.6% | Loại: n_trades 73 < 80; win_rate 0.315 < 0.40; expectancy_r -0.092 < 0.15; profit_factor 0.87 < 1.3; window_win_pct 0.43 < 0.60; phases 1 < 2; sensitivity fails (±20% params make expectancy negative); deflated Sharpe prob 0.00 < 0.9; reality check p 0.888 > 0.1 |
| range_reversion | 192 | 67 lệnh, win 31%, kỳ vọng -0.23R, PF 0.66, maxDD -16.1% | Loại: n_trades 67 < 80; win_rate 0.313 < 0.40; expectancy_r -0.231 < 0.15; profit_factor 0.66 < 1.3; window_win_pct 0.22 < 0.60; sensitivity fails (±20% params make expectancy negative); deflated Sharpe prob 0.00 < 0.9; reality check p 0.888 > 0.1 |
| squeeze_breakout | 192 | 70 lệnh, win 37%, kỳ vọng +0.09R, PF 1.14, maxDD -10.7% | Loại: n_trades 70 < 80; win_rate 0.371 < 0.40; expectancy_r 0.087 < 0.15; profit_factor 1.14 < 1.3; sensitivity fails (±20% params make expectancy negative); deflated Sharpe prob 0.00 < 0.9; reality check p 0.888 > 0.1 |

Không có họ chiến lược nào qua bộ lọc trên dữ liệu dev; test đóng băng **không** được chạy.

## ETH

Số tổ hợp đã thử: 720 · Reality-check p = 0.990

| Họ chiến lược | Số tổ hợp | OOS walk-forward | Kết quả |
|---|---|---|---|
| trend_breakout | 144 | 156 lệnh, win 31%, kỳ vọng -0.25R, PF 0.64, maxDD -38.3% | Loại: win_rate 0.308 < 0.40; expectancy_r -0.249 < 0.15; profit_factor 0.64 < 1.3; max_drawdown -0.383 < -0.20; window_win_pct 0.38 < 0.60; phases 1 < 2; sensitivity fails (±20% params make expectancy negative); deflated Sharpe prob 0.00 < 0.9; reality check p 0.990 > 0.1 |
| trend_pullback | 192 | 105 lệnh, win 26%, kỳ vọng -0.33R, PF 0.59, maxDD -31.6% | Loại: win_rate 0.257 < 0.40; expectancy_r -0.327 < 0.15; profit_factor 0.59 < 1.3; max_drawdown -0.316 < -0.20; window_win_pct 0.38 < 0.60; phases 1 < 2; sensitivity fails (±20% params make expectancy negative); deflated Sharpe prob 0.00 < 0.9; reality check p 0.990 > 0.1 |
| range_reversion | 192 | 54 lệnh, win 41%, kỳ vọng +0.20R, PF 1.34, maxDD -8.6% | Loại: n_trades 54 < 80; sensitivity fails (±20% params make expectancy negative); deflated Sharpe prob 0.01 < 0.9; reality check p 0.990 > 0.1 |
| squeeze_breakout | 192 | 69 lệnh, win 41%, kỳ vọng +0.09R, PF 1.15, maxDD -9.3% | Loại: n_trades 69 < 80; expectancy_r 0.085 < 0.15; profit_factor 1.15 < 1.3; window_win_pct 0.50 < 0.60; sensitivity fails (±20% params make expectancy negative); deflated Sharpe prob 0.00 < 0.9; reality check p 0.990 > 0.1 |

Không có họ chiến lược nào qua bộ lọc trên dữ liệu dev; test đóng băng **không** được chạy.

## Kết luận

**Không có chiến lược nào đạt** tiêu chuẩn sau test đóng băng. Đây là kết quả hợp lệ: không nên giao dịch tiền thật với các cấu hình đã thử.
