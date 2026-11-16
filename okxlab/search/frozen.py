"""Test đóng băng: chạm đúng một lần cho mỗi coin; danh mục chung BTC+ETH với halt lỗ ngày chung."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..backtest import Costs
from ..download import BAR_MS
from ..trade_engine import Rules, prepare_inputs
from . import select as sel
from .constants import DEV_END_TS, load_coin
from .engine_nb import run_prepared
from .families import FAMILIES, HTF_OF
from .indicators import resample_ohlc

ACCEPT_MIN_EXPECTANCY_R = 0.10
ACCEPT_MIN_WIN_RATE = 0.38
DAY_MS = 86_400_000


def is_accepted(stats: dict) -> bool:
    return (stats["n_trades"] > 0 and stats["expectancy_r"] >= ACCEPT_MIN_EXPECTANCY_R
            and stats["win_rate"] >= ACCEPT_MIN_WIN_RATE)


def run_frozen_test(coin: str, spec: dict, out_dir: Path, data_loader=load_coin, costs: Costs | None = None) -> dict:
    """spec = {"family", "params"}. Tín hiệu tính trên toàn bộ dữ liệu (causal, cần phần dev để khởi động chỉ báo),
    nhưng mọi tín hiệu trước DEV_END_TS bị xóa nên chỉ có lệnh trong giai đoạn test."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / f"frozen_{coin}.lock"
    if lock.exists():
        raise RuntimeError(f"frozen test already used for {coin}")
    lock.write_text(json.dumps({"coin": coin, "spec": spec, "ts": datetime.now(timezone.utc).isoformat()}, default=float),
                    encoding="utf-8")  # khóa ghi TRƯỚC khi chạm dữ liệu test

    costs = costs or Costs()
    p = spec["params"]
    bar = p["entry_bar"]
    df, funding, mark = data_loader(coin, bar)
    htf = resample_ohlc(df, BAR_MS[HTF_OF[bar]])
    side, atr = FAMILIES[spec["family"]].signal(df, htf, p)
    side = side.copy()
    side[df["ts"].to_numpy() < DEV_END_TS] = 0.0
    rules = Rules(sl_atr=p["sl_atr"], rr=p["rr"], max_hold_bars=p["max_hold_bars"])
    res = run_prepared(prepare_inputs(df, funding, mark, bar, costs), side, atr, rules, costs)
    res.trades.to_csv(out_dir / f"frozen_{coin}_trades.csv", index=False)
    stats = sel.trade_stats(res.trades)
    accepted = is_accepted(stats)
    out = {"coin": coin, "spec": spec, "stats": stats, "accepted": accepted, "skipped": res.skipped}
    (out_dir / f"frozen_{coin}.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    if accepted:
        (out_dir / f"{coin}_spec.json").write_text(json.dumps(spec, indent=2, default=float), encoding="utf-8")
    return out


def combine_portfolio(trades_by_coin: dict[str, pd.DataFrame], risk_pct: float = 0.01, daily_loss_halt: float = 0.03) -> dict:
    """Ghép lệnh các coin trên một equity chung; sau khi lỗ trong ngày UTC chạm ngưỡng thì chặn lệnh mới vào cùng ngày."""
    frames = [t[t["reason"] != "open_at_end"] for t in trades_by_coin.values() if len(t)]
    if not frames:
        return {"stats": sel.trade_stats(pd.DataFrame(columns=["entry_ts", "exit_ts", "r_multiple", "reason"])),
                "skipped_by_halt": 0}
    allt = pd.concat(frames, ignore_index=True).sort_values("exit_ts")
    day_pnl: dict[int, float] = {}
    halt_ts: dict[int, int] = {}
    kept, skipped = [], 0
    for row in allt.itertuples():
        d = int(row.entry_ts // DAY_MS)
        if d in halt_ts and row.entry_ts >= halt_ts[d]:
            skipped += 1
            continue
        kept.append(row.Index)
        xd = int(row.exit_ts // DAY_MS)
        day_pnl[xd] = day_pnl.get(xd, 0.0) + risk_pct * row.r_multiple
        if day_pnl[xd] <= -daily_loss_halt and xd not in halt_ts:
            halt_ts[xd] = int(row.exit_ts)
    return {"stats": sel.trade_stats(allt.loc[kept], risk_pct), "skipped_by_halt": skipped}
