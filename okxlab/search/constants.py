from __future__ import annotations

import pandas as pd

from ..download import DATA_DIR, candle_path, funding_path, mark_path

DEV_END_TS: int = 1751328000000  # 2025-07-01 00:00 UTC; từ mốc này là test đóng băng
COINS: dict[str, str] = {"BTC": "BTC-USDT-SWAP", "ETH": "ETH-USDT-SWAP"}


def load_coin(coin: str, bar: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(candles, funding, mark), nến bắt đầu từ max(nến đầu, funding đầu) để không bỏ sót chi phí funding."""
    inst = COINS[coin]
    df, funding = pd.read_csv(candle_path(inst, bar)), pd.read_csv(funding_path(inst))
    mark = pd.read_csv(mark_path(inst, bar))
    start = max(int(df["ts"].iloc[0]), int(funding["ts"].iloc[0]))
    df = df[df["ts"] >= start].reset_index(drop=True)
    mark = mark[mark["ts"] >= start].reset_index(drop=True)
    return df, funding, mark
