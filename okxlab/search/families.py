"""Bốn họ chiến lược. `signal(df_entry, df_htf, params) -> (side, atr)`.

side[i] != 0 là *sự kiện vào lệnh* chốt ở đóng cửa nến i (xem trade_engine). Khung HTF: 1H cho entry 15m,
4H cho entry 1H. Mọi tham số lookback tính theo số nến của khung vào lệnh.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind

HTF_OF = {"15m": "1H", "1H": "4H"}
ATR_N = 14


@dataclass
class Family:
    name: str
    grid: list
    signal: Callable


def _grid(own: dict) -> list[dict]:
    common = dict(entry_bar=["15m", "1H"], sl_atr=[1.5, 2.0, 3.0], rr=[2.0, 3.0], max_hold_bars=[48, 96])
    space = {**own, **common}
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*space.values())]


def _events(long_c: np.ndarray, short_c: np.ndarray) -> np.ndarray:
    """Chỉ lấy nến *đầu tiên* điều kiện bắt đầu đúng (cạnh lên)."""
    def edge(x):
        prev = np.concatenate([[False], x[:-1]])
        return x & ~prev
    return edge(long_c).astype(float) - edge(short_c).astype(float)


def _finish(side: np.ndarray, atr: np.ndarray, warmup: int) -> tuple[np.ndarray, np.ndarray]:
    side = side.copy()
    side[:warmup] = 0.0
    side[~np.isfinite(atr)] = 0.0
    return side, atr


def _trend_breakout(e, htf, p):
    n = p["n"]
    c = e["close"].to_numpy()
    up = e["high"].rolling(n).max().shift(1).to_numpy()
    dn = e["low"].rolling(n).min().shift(1).to_numpy()
    f = ind.align_htf(e, htf, ind.ema(htf["close"].to_numpy(), p["htf_ema"]), p["entry_bar"], HTF_OF[p["entry_bar"]])
    with np.errstate(invalid="ignore"):
        side = _events((c > up) & (c > f), (c < dn) & (c < f))
    return _finish(side, ind.atr(e, ATR_N), n + 1)


def _trend_pullback(e, htf, p):
    c = e["close"].to_numpy()
    ef, es = ind.ema(c, p["fast"]), ind.ema(c, p["slow"])
    r = ind.rsi(c, 14)
    rp = np.concatenate([[np.nan], r[:-1]])
    th = p["rsi_th"]
    with np.errstate(invalid="ignore"):
        long_c = (ef > es) & (e["low"].to_numpy() <= ef) & (c > ef) & (rp < th) & (r >= th)
        short_c = (ef < es) & (e["high"].to_numpy() >= ef) & (c < ef) & (rp > 100 - th) & (r <= 100 - th)
    return _finish(_events(long_c, short_c), ind.atr(e, ATR_N), 2 * p["slow"])


def _range_reversion(e, htf, p):
    c = e["close"].to_numpy()
    _, up, lo = ind.bollinger(c, p["bb_n"], p["bb_k"])
    a = ind.adx(e, 14)
    cp = np.concatenate([[np.nan], c[:-1]])
    upp = np.concatenate([[np.nan], up[:-1]])
    lop = np.concatenate([[np.nan], lo[:-1]])
    with np.errstate(invalid="ignore"):
        long_c = (a < p["adx_max"]) & (cp < lop) & (c > lo)
        short_c = (a < p["adx_max"]) & (cp > upp) & (c < up)
    return _finish(_events(long_c, short_c), ind.atr(e, ATR_N), 3 * 14 + p["bb_n"])


def _squeeze_breakout(e, htf, p):
    c = e["close"].to_numpy()
    mid, up, lo = ind.bollinger(c, 20, p["bb_k"])
    with np.errstate(invalid="ignore", divide="ignore"):
        width = pd.Series((up - lo) / mid)
    thr = width.rolling(p["lookback"]).quantile(p["q"] / 100.0).shift(1).to_numpy()
    sq = (width.shift(1).to_numpy() <= thr)
    with np.errstate(invalid="ignore"):
        side = _events(sq & (c > up), sq & (c < lo))
    return _finish(side, ind.atr(e, ATR_N), p["lookback"] + 21)


FAMILIES: dict[str, Family] = {
    "trend_breakout": Family("trend_breakout", _grid(dict(n=[20, 40, 80], htf_ema=[20, 50])), _trend_breakout),
    "trend_pullback": Family("trend_pullback", _grid(dict(fast=[20, 50], slow=[100, 200], rsi_th=[40, 50])), _trend_pullback),
    "range_reversion": Family("range_reversion", _grid(dict(adx_max=[20, 25], bb_n=[20, 40], bb_k=[2.0, 2.5])), _range_reversion),
    "squeeze_breakout": Family("squeeze_breakout", _grid(dict(q=[10, 20], lookback=[100, 200], bb_k=[2.0, 2.5])), _squeeze_breakout),
}
