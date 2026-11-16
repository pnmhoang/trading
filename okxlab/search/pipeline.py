"""Pipeline tìm kiếm cho một coin: mọi tổ hợp của từng họ chạy một lần trên giai đoạn dev, rồi chọn lọc."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest import Costs
from ..download import BAR_MS, candle_path, funding_path, mark_path
from ..trade_engine import Rules, prepare_inputs
from . import select as sel
from .constants import COINS, DEV_END_TS, load_coin
from .engine_nb import run_prepared
from .families import FAMILIES, HTF_OF
from .indicators import resample_ohlc
from .ledger import Ledger, data_fingerprint

NON_SIGNAL_KEYS = ("sl_atr", "rr", "max_hold_bars")
DSR_MIN = 0.90
RC_MAX_P = 0.10
SENS_MIN_TRADES = 30
DAY_MS = 86_400_000


def _data_hash(coin: str, bars) -> str:
    inst = COINS.get(coin)
    paths = []
    for b in bars:
        paths += [candle_path(inst, b), mark_path(inst, b)]
    paths.append(funding_path(inst))
    return data_fingerprint([p for p in paths if Path(p).exists()])


def _neighbors(grid: list[dict], j: int) -> list[int]:
    """Tổ hợp khác tổ hợp j đúng một tham số số học (entry_bar giữ nguyên)."""
    base = grid[j]
    out = []
    for i, p in enumerate(grid):
        if i == j or p["entry_bar"] != base["entry_bar"]:
            continue
        diff = [k for k in base if base[k] != p[k]]
        if len(diff) == 1:
            out.append(i)
    return out


def _daily_r(trades: pd.DataFrame, first_ts: int, n_days: int) -> np.ndarray:
    t = trades[trades["reason"] != "open_at_end"] if len(trades) else trades
    out = np.zeros(n_days)
    if len(t):
        d = ((t["exit_ts"].to_numpy() - first_ts) // DAY_MS).astype(int).clip(0, n_days - 1)
        np.add.at(out, d, t["r_multiple"].to_numpy())
    return out


def search_coin(coin: str, out_dir: Path, families: list[str] | None = None, entry_bars=("15m", "1H"),
                data_loader=load_coin, costs: Costs | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    costs = costs or Costs()
    fams = [FAMILIES[f] for f in (families or FAMILIES)]

    # dữ liệu dev, chuẩn bị mảng một lần cho mỗi khung vào lệnh
    frames, preps, first_ts = {}, {}, 0
    for bar in entry_bars:
        df, funding, mark = data_loader(coin, bar)
        df = df[df["ts"] < DEV_END_TS].reset_index(drop=True)
        mark = None if mark is None else mark[mark["ts"] < DEV_END_TS].reset_index(drop=True)
        htf = resample_ohlc(df, BAR_MS[HTF_OF[bar]])
        frames[bar] = (df, htf)
        preps[bar] = prepare_inputs(df, funding, mark, bar, costs)
        first_ts = max(first_ts, int(df["ts"].iloc[0]))
    last_ts = max(int(f[0]["ts"].iloc[-1]) for f in frames.values())
    windows = sel.walk_forward_windows(first_ts, DEV_END_TS)
    n_days = int((last_ts - first_ts) // DAY_MS) + 1
    ledger = Ledger(out_dir / "ledger.jsonl", data_hash=_data_hash(coin, entry_bars))

    results, all_daily = {}, []
    for fam in fams:
        grid = [p for p in fam.grid if p["entry_bar"] in entry_bars]
        sig_cache, per_combo, full_stats = {}, {}, {}
        for j, p in enumerate(grid):
            key = tuple(sorted((k, v) for k, v in p.items() if k not in NON_SIGNAL_KEYS))
            if key not in sig_cache:
                e, htf = frames[p["entry_bar"]]
                sig_cache[key] = fam.signal(e, htf, p)
            side, atr = sig_cache[key]
            rules = Rules(sl_atr=p["sl_atr"], rr=p["rr"], max_hold_bars=p["max_hold_bars"])
            res = run_prepared(preps[p["entry_bar"]], side, atr, rules, costs)
            per_combo[j] = res.trades
            full_stats[j] = sel.trade_stats(res.trades)
            ledger.append({"coin": coin, "family": fam.name, "params": p, "stage": "dev_full",
                           "stats": full_stats[j], "skipped": res.skipped})
            all_daily.append(_daily_r(res.trades, first_ts, n_days))
        results[fam.name] = (grid, per_combo, full_stats)

    n_trials = sum(len(g) for g, _, _ in results.values())
    rc_p = sel.reality_check_pvalue(np.vstack(all_daily), n_boot=500) if all_daily else 1.0

    table, passed = [], []
    for fam in fams:
        grid, per_combo, full_stats = results[fam.name]
        oos, info = sel.select_params_walk_forward(per_combo, windows)
        stats = sel.trade_stats(oos)
        eligible = [j for j, s in full_stats.items() if s["n_trades"] >= 80]
        final = max(eligible, key=lambda j: full_stats[j]["expectancy_r"]) if eligible else None
        row = {"family": fam.name, "n_combos": len(grid), "n_oos_trades": stats["n_trades"], "oos_stats": stats,
               "window_win_pct": info["window_win_pct"], "final_params": None, "phases": 0, "sens_ok": False,
               "dsr_prob": 0.0, "rc_pvalue": rc_p, "passed": False, "reasons": []}
        if final is None:
            row["reasons"] = ["no combo with >= 80 trades on dev"]
        else:
            nb = [i for i in _neighbors(grid, final) if full_stats[i]["n_trades"] >= SENS_MIN_TRADES]
            sens_ok = bool(nb) and all(full_stats[i]["expectancy_r"] > 0 for i in nb)
            phases = sel.phase_positive_count(per_combo[final])
            r = oos["r_multiple"].to_numpy(float) if len(oos) else np.array([])
            dsr = (sel.deflated_sharpe_prob(stats["sharpe"], len(r), n_trials, float(pd.Series(r).skew()),
                                            float(pd.Series(r).kurt()) + 3.0) if len(r) > 3 else 0.0)
            ok, why = sel.passes_robust(stats, info["window_win_pct"], phases, sens_ok)
            if dsr < DSR_MIN:
                ok = False; why.append(f"deflated Sharpe prob {dsr:.2f} < {DSR_MIN}")
            if rc_p > RC_MAX_P:
                ok = False; why.append(f"reality check p {rc_p:.3f} > {RC_MAX_P}")
            row.update(final_params=grid[final], phases=phases, sens_ok=sens_ok, dsr_prob=float(dsr),
                       passed=ok, reasons=why)
            if ok:
                passed.append({"family": fam.name, "params": grid[final], "oos_stats": stats})
        table.append(row)

    best = max(passed, key=lambda c: c["oos_stats"]["expectancy_r"]) if passed else None
    out = {"coin": coin, "dev_range": [first_ts, last_ts], "n_trials": n_trials, "rc_pvalue": rc_p,
           "passed": passed, "best": best, "table": table}
    (out_dir / f"{coin}_search.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out
