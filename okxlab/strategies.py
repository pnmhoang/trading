"""Chiến lược mẫu. Mỗi hàm trả Series target ∈ {-1,0,1} căn theo index của df, chốt ở đóng cửa nến i.
Chỉ dùng dữ liệu đến hết nến i (kiểm tra bằng tests/test_lookahead.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def donchian(df: pd.DataFrame, entry_n: int = 55, exit_n: int = 20, allow_short: bool = True) -> pd.Series:
    up = df["high"].rolling(entry_n).max().shift(1).to_numpy()
    dn = df["low"].rolling(entry_n).min().shift(1).to_numpy()
    xlo = df["low"].rolling(exit_n).min().shift(1).to_numpy()
    xhi = df["high"].rolling(exit_n).max().shift(1).to_numpy()
    c = df["close"].to_numpy()
    out = np.zeros(len(df)); pos = 0
    for i in range(len(df)):
        if np.isnan(up[i]):
            out[i] = 0; continue
        if pos == 1 and c[i] < xlo[i]:
            pos = 0
        elif pos == -1 and c[i] > xhi[i]:
            pos = 0
        if pos <= 0 and c[i] > up[i]:
            pos = 1
        elif pos >= 0 and c[i] < dn[i] and allow_short:
            pos = -1
        out[i] = pos
    return pd.Series(out, index=df.index)


def ema_trend(df: pd.DataFrame, fast: int = 20, slow: int = 100, allow_short: bool = True) -> pd.Series:
    f = df["close"].ewm(span=fast, adjust=False).mean()
    s = df["close"].ewm(span=slow, adjust=False).mean()
    sig = np.where(f > s, 1.0, -1.0 if allow_short else 0.0)
    sig[: slow] = 0.0  # warmup
    return pd.Series(sig, index=df.index)


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


STRATEGIES = {"donchian": donchian, "ema_trend": ema_trend, "buy_hold": lambda df: buy_and_hold(df)}
