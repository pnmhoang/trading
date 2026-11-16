"""Chỉ báo causal: giá trị tại nến i chỉ dùng dữ liệu đến hết nến i."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..download import BAR_MS


def _ewm(x: pd.Series, alpha: float, n: int) -> pd.Series:
    return x.ewm(alpha=alpha, adjust=False, min_periods=n).mean()


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def atr(df: pd.DataFrame, n: int) -> np.ndarray:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return _ewm(tr, 1.0 / n, n).to_numpy()


def rsi(close: np.ndarray, n: int) -> np.ndarray:
    d = pd.Series(close).diff()
    gain = _ewm(d.clip(lower=0), 1.0 / n, n)
    loss = _ewm((-d).clip(lower=0), 1.0 / n, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
        out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(loss != 0, 100.0).where(gain.notna())
    return out.to_numpy()


def adx(df: pd.DataFrame, n: int) -> np.ndarray:
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = pd.Series(atr(df, n), index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * _ewm(plus, 1.0 / n, n) / a
        mdi = 100 * _ewm(minus, 1.0 / n, n) / a
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return _ewm(dx.fillna(0.0).where(a.notna()), 1.0 / n, n).to_numpy()


def bollinger(close: np.ndarray, n: int, k: float):
    s = pd.Series(close)
    mid = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    return mid.to_numpy(), (mid + k * sd).to_numpy(), (mid - k * sd).to_numpy()


def resample_ohlc(df: pd.DataFrame, bar_ms: int) -> pd.DataFrame:
    g = df.groupby(df["ts"] // bar_ms * bar_ms)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "vol": g["vol"].sum()})
    out.index.name = "ts"
    return out.reset_index()


def align_htf(df_entry: pd.DataFrame, df_htf: pd.DataFrame, values: np.ndarray, entry_bar: str, htf_bar: str) -> np.ndarray:
    """Với mỗi nến vào lệnh, lấy giá trị của nến HTF *đã đóng* tại thời điểm nến vào lệnh đóng."""
    entry_close = df_entry["ts"].to_numpy(np.int64) + BAR_MS[entry_bar]
    htf_close = df_htf["ts"].to_numpy(np.int64) + BAR_MS[htf_bar]
    idx = np.searchsorted(htf_close, entry_close, side="right") - 1
    out = np.full(len(df_entry), np.nan)
    ok = idx >= 0
    out[ok] = np.asarray(values, float)[idx[ok]]
    return out
