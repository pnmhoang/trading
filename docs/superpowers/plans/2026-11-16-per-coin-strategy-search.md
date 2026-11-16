# Per-Coin Strategy Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tìm, cho từng coin BTC-USDT-SWAP và ETH-USDT-SWAP, một chiến lược đóng băng (hoặc kết luận "không có") với SL theo ATR, TP 2R/3R, rủi ro 1% vốn mỗi lệnh, được chọn bằng walk-forward + hiệu chỉnh theo số lần thử + test đóng băng.

**Architecture:** Engine "trade-based" tham chiếu bằng Python (`trade_engine.py`) là chuẩn đúng; engine numba (`search/engine_nb.py`) chạy tìm kiếm và phải khớp engine tham chiếu. Bốn họ chiến lược sinh `(side, atr)` causal; mỗi tổ hợp chạy một lần trên giai đoạn dev rồi cắt theo cửa sổ; bộ chọn lọc theo độ bền, kiểm định thống kê, rồi test đóng băng đúng một lần.

**Tech Stack:** Python 3.13, pandas, numpy, numba, pytest. Dữ liệu trong `data/` (CSV, không commit).

**Spec:** `docs/superpowers/specs/2026-11-16-per-coin-strategy-search-design.md`

## Global Constraints

- Rủi ro mỗi lệnh = 1% equity; `notional = 0.01 * equity / sl_distance_pct`; bỏ lệnh nếu đòn bẩy cần > 3.0x hoặc `chi phí khứ hồi / khoảng cách SL > 0.15`.
- SL = `k × ATR`; TP = `rr × SL`, `rr ∈ {2.0, 3.0}`; một vị thế mỗi coin; dừng trong ngày nếu lỗ ngày ≥ 3% equity.
- Cùng nến chạm cả SL và TP ⇒ SL trước. TP chỉ khớp khi giá đi qua mức TP và khớp ở giá TP. SL gap ⇒ khớp ở giá mở. Thanh lý theo mark price.
- Mặc định chi phí: taker 0.0005 cho vào lệnh, SL **và TP**; slippage 2 bps + 0.05 × biên độ nến trước; funding thật.
- Dev: `2022-01-16` → `2025-06-30` (`DEV_END_TS = 1751328000000`, loại trừ); test đóng băng: `ts >= 1751328000000`, mỗi coin chạm đúng một lần. Walk-forward train 12 tháng / test 3 tháng.
- Bộ lọc bền vững: ≥ 80 lệnh; win rate ≥ 0.40; kỳ vọng ≥ +0.15R/lệnh; profit factor ≥ 1.3; lãi trong ≥ 60% cửa sổ test; max drawdown ≤ 20%; dương trong ≥ 2 trong 3 pha (2022, 2023, 2024 theo năm dương lịch); độ nhạy ±20% tham số không làm kỳ vọng âm. Test đóng băng: kỳ vọng ≥ +0.1R và win rate ≥ 0.38.
- Không commit/push file dưới `docs/` (spec, plan) trừ khi người dùng yêu cầu. Không push bất cứ gì khi chưa được yêu cầu.

## Review Focus

1. ATR bằng 0 hoặc NaN (giai đoạn khởi động) ⇒ bỏ lệnh, không chia cho 0.
2. SL quá hẹp khiến đòn bẩy cần > 3x hoặc chi phí > 0.15R ⇒ bỏ lệnh và đếm `skipped`.
3. Nến chạm cả SL lẫn TP, và nến gap qua SL ⇒ SL trước, khớp ở giá mở khi gap.
4. Chạy test đóng băng lần thứ hai cho cùng coin ⇒ từ chối.
5. Không họ chiến lược nào qua bộ lọc ⇒ báo cáo "không có chiến lược đạt", không sinh file đặc tả, thoát mã 0.

---

### Task 1: Dữ liệu ETH + hằng số dùng chung

**Files:**
- Create: `okxlab/search/__init__.py`, `okxlab/search/constants.py`
- Test: `tests/test_constants.py`

**Interfaces:**
- Produces (`constants.py`): `DEV_END_TS: int = 1751328000000`, `COINS: dict[str, str] = {"BTC": "BTC-USDT-SWAP", "ETH": "ETH-USDT-SWAP"}`, `load_coin(coin: str, bar: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` trả `(candles, funding, mark)` đã cắt bắt đầu tại `max(candles[0], funding[0])`, đọc từ `data/`.

- [ ] **Step 1:** Tải ETH: `python -m okxlab download --inst ETH-USDT-SWAP --bar 15m --start 2022-01-01 --mark`, rồi `--bar 1H ... --mark`. Chạy `python -m okxlab validate --inst ETH-USDT-SWAP --bar 15m` và `--bar 1H`; mong đợi mã thoát 0. Nếu có lỗi `critical` (khoảng trống, funding bắt đầu muộn), dừng và báo người dùng.
- [ ] **Step 2: Viết test** `test_load_coin_trims_to_funding_and_returns_three_frames` (dùng file `data/` thật của BTC 1H: `candles.ts.iloc[0] >= funding.ts.iloc[0]`, `mark` có cột `ts, open, high, low, close`) và `test_load_coin_unknown_coin_raises` (`pytest.raises(KeyError)`).
- [ ] **Step 3:** Chạy test, mong đợi FAIL; cài `numba` (`python -m pip install numba`).
- [ ] **Step 4:** Implement `constants.py`; chạy test, mong đợi PASS.
- [ ] **Step 5:** Commit `feat: ETH data download and search constants` (chỉ code, không `data/*.csv`).

### Task 2: Engine tham chiếu theo lệnh (TP/SL, rủi ro 1%)

**Files:**
- Create: `okxlab/trade_engine.py`
- Test: `tests/test_trade_engine.py`

**Interfaces:**
- Consumes: `okxlab.backtest.Costs` (thêm field `tp_fee: float | None = None`, nghĩa là dùng `taker_fee`), `okxlab.download.BAR_MS`.
- Produces:
  - `@dataclass(frozen=True) Rules(sl_atr: float, rr: float, max_hold_bars: int, risk_pct: float = 0.01, max_leverage: float = 3.0, max_cost_r: float = 0.15, daily_loss_halt: float = 0.03)`
  - `@dataclass TradeResult(trades: pd.DataFrame, equity: np.ndarray, skipped: int)`; cột `trades`: `entry_ts, exit_ts, side, entry, exit, r_multiple, net, fee, funding, reason, bars` với `reason ∈ {"tp","sl","time","liquidation","open_at_end"}`.
  - `run_trades(df: pd.DataFrame, funding: pd.DataFrame, mark: pd.DataFrame | None, side: np.ndarray, atr: np.ndarray, rules: Rules, costs: Costs, bar: str, initial_equity: float = 10_000.0) -> TradeResult`

Ngữ nghĩa: `side[i]`/`atr[i]` chốt ở đóng cửa nến i, khớp ở mở cửa nến i+1; `r_multiple = net / (risk_pct * equity_at_entry)`; dùng lại logic funding/slippage/thanh lý của `backtest.run`.

- [ ] **Step 1: Viết test (mỗi test dựng dữ liệu giả như `tests/test_engine.py`):**
  - `test_tp_hit_gives_about_plus_rr` (giá đi thẳng lên, `rr=2`, costs bằng 0 ⇒ `r_multiple ≈ 2.0`, `reason == "tp"`)
  - `test_sl_hit_gives_about_minus_one_r` (costs 0 ⇒ `r_multiple ≈ -1.0`)
  - `test_same_bar_sl_and_tp_resolves_to_sl`
  - `test_gap_through_sl_fills_at_open` (`exit == open` của nến gap, `r_multiple < -1.0`)
  - `test_position_size_risks_one_percent` (SL hit, costs 0 ⇒ `net ≈ -0.01 * initial_equity`)
  - `test_skips_trade_when_leverage_needed_exceeds_cap` (SL 0.2% ⇒ `skipped == 1`, không có lệnh)
  - `test_skips_trade_when_cost_over_max_cost_r` (SL hẹp, costs mặc định)
  - `test_atr_zero_or_nan_skips_without_error`
  - `test_daily_loss_halt_blocks_further_entries_same_day`
  - `test_time_stop_exits_at_max_hold_bars`
  - `test_no_lookahead_truncation` (cắt dữ liệu giữa chừng, các lệnh đã đóng trước điểm cắt giống hệt)
- [ ] **Step 2:** Chạy `pytest tests/test_trade_engine.py -v`, mong đợi FAIL (module chưa có).
- [ ] **Step 3:** Implement `run_trades`. Một vòng lặp theo nến: funding → kiểm tra thoát (liquidation theo mark, SL, TP, time-stop; SL ưu tiên) → vào lệnh mới tại mở cửa từ tín hiệu nến trước nếu chưa có vị thế, chưa bị halt trong ngày UTC, và qua hai kiểm tra bỏ lệnh. Cuối dữ liệu đóng vị thế ở giá đóng với `reason="open_at_end"`.
- [ ] **Step 4:** Chạy test, mong đợi PASS; chạy `pytest -q` toàn bộ, mong đợi không hỏng test cũ.
- [ ] **Step 5:** Commit `feat: trade-based engine with TP/SL and 1% risk sizing`.

### Task 3: Engine numba khớp engine tham chiếu

**Files:**
- Create: `okxlab/search/engine_nb.py`
- Test: `tests/test_engine_nb_parity.py`

**Interfaces:**
- Consumes: `Rules`, `Costs`, `TradeResult`, `run_trades` (Task 2).
- Produces: `run_trades_fast(df, funding, mark, side, atr, rules, costs, bar, initial_equity=10_000.0) -> TradeResult` — cùng chữ ký và kết quả như `run_trades`; hàm `@njit` lõi nhận mảng numpy thuần.

- [ ] **Step 1: Viết test** `test_parity_with_reference_on_random_cases`: 30 trường hợp ngẫu nhiên (seed cố định) trên random walk, `side` ngẫu nhiên thưa, nhiều giá trị `sl_atr`, `rr ∈ {2,3}`, `max_hold_bars`; assert số lệnh bằng nhau, `np.allclose(trades[["entry","exit","r_multiple","net"]], ..., atol=1e-9)`, `reason` giống hệt, `skipped` bằng nhau, `np.allclose(equity)`. Thêm `test_parity_on_real_btc_1h_slice` (5.000 nến đầu của dữ liệu thật). Thêm `test_fast_is_at_least_10x_faster` trên 40.000 nến (chỉ cần `t_ref / t_fast >= 10` sau lần chạy làm nóng).
- [ ] **Step 2:** Chạy, mong đợi FAIL.
- [ ] **Step 3:** Implement bằng cách chuyển vòng lặp của Task 2 sang hàm `@njit(cache=True)` trả mảng lệnh; wrapper dựng `TradeResult`. Không đổi ngữ nghĩa.
- [ ] **Step 4:** Chạy test, mong đợi PASS (nếu lệch, sửa engine numba, không nới dung sai).
- [ ] **Step 5:** Commit `feat: numba trade engine with parity tests`.

### Task 4: Chỉ báo và bốn họ chiến lược

**Files:**
- Create: `okxlab/search/indicators.py`, `okxlab/search/families.py`
- Test: `tests/test_families.py`

**Interfaces:**
- Produces (`indicators.py`): `atr(df, n: int) -> np.ndarray`, `ema(x: np.ndarray, n: int) -> np.ndarray`, `rsi(close, n)`, `adx(df, n)`, `bollinger(close, n, k) -> tuple[mid, upper, lower]`, `align_htf(df_entry: pd.DataFrame, df_htf: pd.DataFrame, values: np.ndarray, entry_bar: str, htf_bar: str) -> np.ndarray` — chỉ dùng giá trị HTF của nến đã đóng tại thời điểm đóng của nến vào lệnh.
- Produces (`families.py`): `@dataclass Family(name: str, grid: list[dict], signal: Callable[[pd.DataFrame, pd.DataFrame, dict], tuple[np.ndarray, np.ndarray]])`; `FAMILIES: dict[str, Family]` với khóa `trend_breakout`, `trend_pullback`, `range_reversion`, `squeeze_breakout`. `signal(df_entry, df_htf, params) -> (side, atr)`; mọi `params` đều có `entry_bar ∈ {"15m","1H"}`, `sl_atr`, `rr ∈ {2.0, 3.0}`, `max_hold_bars` cộng ≤ 3 tham số riêng của họ. Mỗi lưới ≤ 200 tổ hợp.

- [ ] **Step 1: Viết test:** `test_signals_are_causal[family]` (tham số hóa 4 họ: tín hiệu trên dữ liệu cắt tại k bằng tiền tố của tín hiệu trên dữ liệu đầy đủ, ở nhiều k), `test_align_htf_never_uses_unclosed_htf_bar`, `test_grid_sizes_within_limit`, `test_side_values_in_minus1_0_1_and_atr_nan_only_in_warmup`, `test_indicator_values_match_reference_on_known_series` (RSI/ATR/EMA trên chuỗi nhỏ tính tay).
- [ ] **Step 2:** Chạy, mong đợi FAIL.
- [ ] **Step 3:** Implement. Ý nghĩa họ: breakout = phá Donchian N cùng hướng EMA HTF; pullback = EMA nhanh/chậm xác định xu hướng, vào khi giá chạm EMA nhanh và RSI quay lại; range = ADX < ngưỡng và giá ra ngoài Bollinger rồi vào lại; squeeze = độ rộng Bollinger dưới phân vị rồi phá băng.
- [ ] **Step 4:** Chạy test, mong đợi PASS.
- [ ] **Step 5:** Commit `feat: indicators and four strategy families`.

### Task 5: Sổ thí nghiệm

**Files:**
- Create: `okxlab/search/ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Produces: `data_fingerprint(paths: list[pathlib.Path]) -> str` (sha256 ghép), `code_fingerprint() -> str` (sha256 các file `trade_engine.py`, `search/engine_nb.py`, `search/families.py`), `class Ledger(path: pathlib.Path)` với `append(record: dict) -> None` (thêm `data_hash`, `code_hash`, `ts` UTC) và `count(coin: str, family: str) -> int` (số tổ hợp đã thử, dùng cho hiệu chỉnh).

- [ ] **Step 1: Viết test:** `test_append_and_count`, `test_fingerprint_changes_when_file_changes` (tmp file), `test_ledger_is_append_only_jsonl` (mỗi dòng là JSON hợp lệ, hai lần ghi ⇒ hai dòng).
- [ ] **Step 2–4:** Chạy FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: experiment ledger`.

### Task 6: Chọn lọc và kiểm định thống kê

**Files:**
- Create: `okxlab/search/select.py`
- Test: `tests/test_select.py`

**Interfaces:**
- Consumes: `TradeResult.trades` (Task 2).
- Produces:
  - `walk_forward_windows(first_ts: int, end_ts: int, train_months: int = 12, test_months: int = 3) -> list[tuple[int, int, int, int]]` (train_start, train_end, test_start, test_end theo ts ms, cuối khoảng loại trừ).
  - `trade_stats(trades: pd.DataFrame, risk_pct: float = 0.01) -> dict` với khóa `n_trades, win_rate, expectancy_r, profit_factor, max_drawdown, sharpe` (equity = tích `1 + risk_pct * r_multiple`; bỏ lệnh `open_at_end`).
  - `phase_positive_count(trades: pd.DataFrame) -> int` (số năm dương lịch trong {2022, 2023, 2024} có tổng R dương).
  - `passes_robust(stats: dict, window_win_pct: float, phases: int, sens_ok: bool) -> tuple[bool, list[str]]` (danh sách lý do loại).
  - `deflated_sharpe_prob(sharpe: float, n_obs: int, n_trials: int, skew: float, kurt: float) -> float`
  - `reality_check_pvalue(trial_returns: np.ndarray, n_boot: int = 1000, seed: int = 0) -> float` (`trial_returns` shape `(n_trials, n_obs)`; bootstrap theo khối).
  - `select_params_walk_forward(per_combo_trades: dict[int, pd.DataFrame], windows) -> tuple[pd.DataFrame, dict]` trả các lệnh OOS ghép lại và `{"window_win_pct": float, "chosen": list[int]}` (mỗi cửa sổ chọn combo có `expectancy_r` train cao nhất với ≥ 20 lệnh train).

- [ ] **Step 1: Viết test:** `test_windows_do_not_overlap_and_are_ordered`, `test_trade_stats_known_example` (4 lệnh R = [2,-1,-1,2] ⇒ `win_rate=0.5`, `expectancy_r=0.5`, `profit_factor=2.0`), `test_trade_stats_excludes_open_at_end`, `test_phase_positive_count`, `test_passes_robust_reports_each_failed_rule`, `test_deflated_sharpe_decreases_with_more_trials`, `test_reality_check_noise_not_significant` (1.000 chuỗi nhiễu trung bình 0 ⇒ p > 0.1), `test_reality_check_detects_planted_edge` (một chuỗi drift lớn ⇒ p < 0.05), `test_select_walk_forward_uses_only_train_to_choose` (combo tốt ở test nhưng tệ ở train không được chọn).
- [ ] **Step 2–4:** Chạy FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: selection filters and multiple-testing checks`.

### Task 7: Pipeline tìm kiếm cho từng coin + CLI

**Files:**
- Create: `okxlab/search/pipeline.py`
- Modify: `okxlab/__main__.py` (thêm lệnh con `search`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `load_coin`, `DEV_END_TS` (Task 1), `run_trades_fast` (3), `FAMILIES` (4), `Ledger` (5), `select` (6).
- Produces: `search_coin(coin: str, out_dir: pathlib.Path, families: list[str] | None = None, data_loader=load_coin) -> dict` — chạy mọi tổ hợp của từng họ trên giai đoạn dev (một lần mỗi tổ hợp, cắt theo cửa sổ), ghi từng lần thử vào sổ, áp bộ lọc + sensitivity + deflated Sharpe + reality check; trả `{"coin", "passed": list[dict], "best": dict | None, "table": list[dict]}` và ghi `out_dir/<coin>_search.json`. Mỗi phần tử `passed`/`best` có `family, params, oos_stats`.
- CLI: `python -m okxlab search --coin BTC|ETH [--families a,b]`.

- [ ] **Step 1: Viết test** (inject `data_loader` giả để chạy nhanh): `test_search_on_pure_noise_returns_no_passed_candidates` (random walk 3 năm, `best is None`, file JSON vẫn được ghi), `test_search_finds_planted_edge` (dữ liệu tổng hợp có xu hướng kéo dài đủ để họ trend_breakout qua bộ lọc; nếu quá khó dựng, đánh dấu skip có lý do và thay bằng test của `passes_robust`), `test_every_tried_combo_is_in_ledger`, `test_dev_only_no_test_period_data_used` (assert mọi `ts` đưa vào engine < `DEV_END_TS`).
- [ ] **Step 2–4:** Chạy FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: per-coin search pipeline and CLI`.

### Task 8: Test đóng băng, danh mục chung và báo cáo

**Files:**
- Create: `okxlab/search/frozen.py`, `okxlab/search/report.py`
- Modify: `okxlab/__main__.py` (lệnh con `frozen`, `report`)
- Test: `tests/test_frozen_report.py`

**Interfaces:**
- Produces:
  - `run_frozen_test(coin: str, spec: dict, out_dir: pathlib.Path, data_loader=load_coin) -> dict` — chạy chiến lược `spec` (`family, params`) trên `ts >= DEV_END_TS` (kèm đoạn khởi động trước đó chỉ để tính chỉ báo), ghi `out_dir/frozen_<coin>.lock` **trước** khi chạy; nếu file khóa tồn tại ⇒ `raise RuntimeError("frozen test already used for <coin>")`. Trả `{"stats": dict, "accepted": bool}` với `accepted = expectancy_r >= 0.1 and win_rate >= 0.38`.
  - `combine_portfolio(trades_by_coin: dict[str, pd.DataFrame], risk_pct: float = 0.01, daily_loss_halt: float = 0.03) -> dict` — ghép lệnh hai coin theo thời gian trên một equity chung, áp halt lỗ ngày chung, trả `trade_stats` của danh mục.
  - `write_report(coin_results: dict, out_path: pathlib.Path) -> None` — Markdown: bảng mọi họ (số tổ hợp, số qua bộ lọc), ứng viên tốt nhất, kết quả test đóng băng, kết luận trung thực nếu không có ứng viên, cùng đường dẫn `<coin>_spec.json` (chỉ sinh khi `accepted`).

- [ ] **Step 1: Viết test:** `test_frozen_test_refuses_second_run` (lần hai ⇒ `RuntimeError`), `test_frozen_lock_written_before_run` (loader giả ném lỗi ⇒ file khóa vẫn tồn tại), `test_accept_thresholds`, `test_combine_portfolio_applies_shared_daily_halt`, `test_report_states_no_strategy_passed_when_none`.
- [ ] **Step 2–4:** Chạy FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: frozen test guard, portfolio combine and report`.

### Task 9: Chạy thật BTC và ETH, báo cáo kết quả

**Files:**
- Create (không commit nếu là kết quả lớn): `results/BTC_search.json`, `results/ETH_search.json`, `results/report.md`, `results/<coin>_spec.json` (nếu có)

- [ ] **Step 1:** `python -m okxlab search --coin BTC` rồi `--coin ETH`. Ghi lại thời gian chạy và số tổ hợp đã thử (từ sổ).
- [ ] **Step 2:** Với mỗi coin có `best`, hỏi người dùng xác nhận **trước khi** chạy test đóng băng (đây là lần chạm duy nhất); sau đó `python -m okxlab frozen --coin <COIN>`.
- [ ] **Step 3:** `python -m okxlab report`; đưa `results/report.md` cho người dùng, nêu rõ nếu không coin nào có ứng viên được chấp nhận.
- [ ] **Step 4:** `pytest -q` toàn bộ phải xanh. Commit code (không commit `docs/`, không push khi chưa được yêu cầu).
