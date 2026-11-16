"""Chọn lọc: cửa sổ walk-forward, thống kê lệnh, bộ lọc bền vững, deflated Sharpe, reality check."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329
MIN_TRAIN_TRADES = 20
PHASE_YEARS = (2022, 2023, 2024)


def walk_forward_windows(first_ts: int, end_ts: int, train_months: int = 12, test_months: int = 3):
    """[(train_start, train_end, test_start, test_end)] theo ms; test liền kề nhau, cuối khoảng loại trừ."""
    start = pd.Timestamp(first_ts, unit="ms", tz="UTC")
    end = pd.Timestamp(end_ts, unit="ms", tz="UTC")
    out, k = [], 0
    while True:
        a = start + pd.DateOffset(months=test_months * k)
        b = a + pd.DateOffset(months=train_months)
        d = b + pd.DateOffset(months=test_months)
        if d > end:
            break
        out.append(tuple(int(x.timestamp() * 1000) for x in (a, b, b, d)))
        k += 1
    return out


def _closed(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[trades["reason"] != "open_at_end"] if len(trades) else trades


def trade_stats(trades: pd.DataFrame, risk_pct: float = 0.01) -> dict:
    t = _closed(trades)
    if len(t) == 0:
        return dict(n_trades=0, win_rate=0.0, expectancy_r=0.0, profit_factor=0.0, max_drawdown=0.0, sharpe=0.0)
    r = t.sort_values("exit_ts")["r_multiple"].to_numpy(float)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumprod(1.0 + risk_pct * r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return dict(n_trades=int(len(r)), win_rate=float((r > 0).mean()), expectancy_r=float(r.mean()),
                profit_factor=float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0),
                max_drawdown=float((eq / peak - 1).min()), sharpe=float(r.mean() / sd) if sd > 0 else 0.0)


def phase_positive_count(trades: pd.DataFrame) -> int:
    t = _closed(trades)
    if len(t) == 0:
        return 0
    year = pd.to_datetime(t["entry_ts"], unit="ms", utc=True).dt.year
    by = t.groupby(year)["r_multiple"].sum()
    return int(sum(by.get(y, 0.0) > 0 for y in PHASE_YEARS))


def passes_robust(stats: dict, window_win_pct: float, phases: int, sens_ok: bool) -> tuple[bool, list[str]]:
    why = []
    if stats["n_trades"] < 80: why.append(f"n_trades {stats['n_trades']} < 80")
    if stats["win_rate"] < 0.40: why.append(f"win_rate {stats['win_rate']:.3f} < 0.40")
    if stats["expectancy_r"] < 0.15: why.append(f"expectancy_r {stats['expectancy_r']:.3f} < 0.15")
    if stats["profit_factor"] < 1.3: why.append(f"profit_factor {stats['profit_factor']:.2f} < 1.3")
    if stats["max_drawdown"] < -0.20: why.append(f"max_drawdown {stats['max_drawdown']:.3f} < -0.20")
    if window_win_pct < 0.60: why.append(f"window_win_pct {window_win_pct:.2f} < 0.60")
    if phases < 2: why.append(f"phases {phases} < 2")
    if not sens_ok: why.append("sensitivity fails (±20% params make expectancy negative)")
    return (not why), why


def deflated_sharpe_prob(sharpe: float, n_obs: int, n_trials: int, skew: float, kurt: float) -> float:
    """Xác suất Sharpe thật > 0 sau khi trừ kỳ vọng cực đại của n_trials lần thử nhiễu (Bailey & López de Prado).
    `sharpe` là Sharpe trên mỗi quan sát (không quy năm); kurt là kurtosis thường (chuẩn = 3)."""
    sd0 = 1.0 / np.sqrt(max(n_obs - 1, 1))
    if n_trials > 1:
        sr0 = sd0 * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1.0 / n_trials)
                     + EULER_GAMMA * _N.inv_cdf(1 - 1.0 / (n_trials * np.e)))
    else:
        sr0 = 0.0
    denom = np.sqrt(max(1 - skew * sharpe + (kurt - 1) / 4.0 * sharpe ** 2, 1e-12))
    return float(_N.cdf((sharpe - sr0) * np.sqrt(max(n_obs - 1, 1)) / denom))


def reality_check_pvalue(trial_returns: np.ndarray, n_boot: int = 1000, seed: int = 0) -> float:
    """White's reality check: p-value cho H0 'không chiến lược nào trong n_trials có kỳ vọng > 0'.
    trial_returns shape (n_trials, n_obs), các hàng cùng trục thời gian. Bootstrap theo khối (stationary-like)."""
    x = np.asarray(trial_returns, float)
    k, n = x.shape
    mean = x.mean(axis=1)
    stat = np.sqrt(n) * mean.max()
    rng = np.random.default_rng(seed)
    block = max(int(np.sqrt(n)), 1)
    n_blocks = int(np.ceil(n / block))
    count = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        boot = x[:, idx].mean(axis=1)
        count += np.sqrt(n) * (boot - mean).max() >= stat
    return float((count + 1) / (n_boot + 1))


def select_params_walk_forward(per_combo_trades: dict[int, pd.DataFrame], windows):
    """Mỗi cửa sổ: chọn combo có expectancy_r train cao nhất (≥ MIN_TRAIN_TRADES lệnh train), lấy lệnh test của combo đó."""
    oos, chosen, wins, scored = [], [], 0, 0
    for tr_a, tr_b, te_a, te_b in windows:
        best, best_score = None, -np.inf
        for cid, t in per_combo_trades.items():
            c = _closed(t)
            tr = c[(c["entry_ts"] >= tr_a) & (c["entry_ts"] < tr_b)]
            if len(tr) < MIN_TRAIN_TRADES:
                continue
            score = tr["r_multiple"].mean()
            if score > best_score:
                best, best_score = cid, score
        chosen.append(best)
        if best is None:
            continue
        c = _closed(per_combo_trades[best])
        te = c[(c["entry_ts"] >= te_a) & (c["entry_ts"] < te_b)]
        if len(te):
            oos.append(te)
            scored += 1
            wins += te["r_multiple"].sum() > 0
    out = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame(columns=["entry_ts", "exit_ts", "r_multiple", "reason"])
    return out, {"window_win_pct": float(wins / scored) if scored else 0.0, "chosen": chosen}
