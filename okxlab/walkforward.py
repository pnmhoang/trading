"""Walk-forward + độ nhạy tham số: công cụ chính để chống overfit."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .backtest import Config, Costs, run


def grid(**ranges) -> list[dict]:
    keys = list(ranges)
    return [dict(zip(keys, v)) for v in itertools.product(*ranges.values())]


def _slice(df, funding, a, b):
    d = df.iloc[a:b].reset_index(drop=True)
    f = funding[(funding["ts"] >= d["ts"].iloc[0]) & (funding["ts"] <= d["ts"].iloc[-1])]
    return d, f


def walk_forward(df, funding, strat_fn, params_grid, bar, train_bars, test_bars,
                 cfg=Config(), costs=Costs(), min_trades=8, objective="sharpe"):
    """Với mỗi cửa sổ: chọn tham số tốt nhất trên TRAIN, áp dụng cố định lên TEST kế tiếp.
    Tín hiệu tính trên toàn đoạn [train_start, test_end] (causal) nên không rò rỉ tương lai."""
    rows, oos_returns = [], []
    start = 0
    while start + train_bars + test_bars <= len(df):
        a, m, b = start, start + train_bars, start + train_bars + test_bars
        seg, fseg = _slice(df, funding, a, b)
        best, best_score = None, -np.inf
        for p in params_grid:
            tgt = strat_fn(seg, **p)
            tr = run(seg.iloc[: m - a], fseg, tgt.iloc[: m - a], bar, cfg, costs)
            if tr.stats["trades"] < min_trades:
                continue
            score = tr.stats[objective]
            if score > best_score:
                best, best_score = p, score
        if best is None:
            start += test_bars; continue
        tgt = strat_fn(seg, **best)
        test_seg = seg.iloc[m - a:].reset_index(drop=True)
        te = run(test_seg, fseg, tgt.iloc[m - a:].reset_index(drop=True), bar, cfg, costs)
        rows.append({"test_start": int(test_seg["ts"].iloc[0]), "params": best, "train_" + objective: best_score,
                     "test_return": te.stats["total_return"], "test_sharpe": te.stats["sharpe"],
                     "test_dd": te.stats["max_drawdown"], "test_trades": te.stats["trades"]})
        oos_returns.append(te.equity.pct_change().fillna(0.0))
        start += test_bars
    wf = pd.DataFrame(rows)
    summary = {"windows": len(wf), "trials_total": len(params_grid) * len(wf)}
    if len(wf):
        r = pd.concat(oos_returns)
        eq = (1 + r).cumprod()
        summary.update({
            "oos_total_return": float(eq.iloc[-1] - 1),
            "oos_max_drawdown": float((eq / eq.cummax() - 1).min()),
            "windows_profitable_pct": float((wf["test_return"] > 0).mean()),
            "median_test_sharpe": float(wf["test_sharpe"].median()),
            "avg_train_vs_test_sharpe_gap": float(wf["train_" + objective].mean() - wf["test_sharpe"].mean()),
        })
    return wf, summary


def sensitivity(df, funding, strat_fn, base: dict, key: str, values, bar, cfg=Config(), costs=Costs()) -> pd.DataFrame:
    """Đổi 1 tham số quanh giá trị gốc. Nếu kết quả sụp khi lệch ±20% => tham số overfit."""
    rows = []
    for v in values:
        p = {**base, key: v}
        r = run(df, funding, strat_fn(df, **p), bar, cfg, costs)
        rows.append({key: v, "return": r.stats["total_return"], "sharpe": r.stats["sharpe"],
                     "max_dd": r.stats["max_drawdown"], "trades": r.stats["trades"]})
    return pd.DataFrame(rows)
