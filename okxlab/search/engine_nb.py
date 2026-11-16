"""Engine numba cho tìm kiếm: biên dịch đúng `trade_engine._core` (cùng ngữ nghĩa, không viết lại)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ..backtest import Costs
from ..trade_engine import Rules, TradeResult, _core, build_result, core_args, prepare_inputs

_core_nb = njit(cache=True)(_core)


def run_prepared(prep: dict, side: np.ndarray, atr: np.ndarray, rules: Rules, costs: Costs,
                 initial_equity: float = 10_000.0, interpreted: bool = False) -> TradeResult:
    """Chạy lõi trên mảng đã chuẩn bị bởi `prepare_inputs` (chuẩn bị một lần, dùng cho nhiều tổ hợp)."""
    fn = _core if interpreted else _core_nb
    out, eq, skipped = fn(prep["ts"], prep["o"], prep["h"], prep["l"], prep["c"], prep["mh"], prep["ml"],
                          prep["slip"], prep["fb"], prep["fi"], np.asarray(side, float), np.asarray(atr, float),
                          *core_args(rules, costs, initial_equity))
    return build_result(out, eq, skipped)


def run_trades_fast(df: pd.DataFrame, funding: pd.DataFrame, mark: pd.DataFrame | None, side: np.ndarray,
                    atr: np.ndarray, rules: Rules, costs: Costs, bar: str, initial_equity: float = 10_000.0) -> TradeResult:
    return run_prepared(prepare_inputs(df, funding, mark, bar, costs), side, atr, rules, costs, initial_equity)
