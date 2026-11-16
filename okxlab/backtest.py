"""Backtester futures USDT-margined (linear perpetual) có chi phí thực tế.

Quy ước chống look-ahead:
  * `target[i]` là quyết định chốt ở *đóng cửa* nến i; engine chỉ khớp ở *mở cửa* nến i+1.
  * Slippage = slip_bps cố định + k * biên độ nến *trước đó* (chỉ dùng thông tin đã biết).
Chi phí mô hình hóa: phí taker, slippage, funding (đúng chiều/độ lớn vị thế), thanh lý (giá xấu nhất trong nến),
stop-loss (khớp ở giá stop hoặc giá mở nếu gap, cộng slippage).
Chưa mô hình hóa: lot size/min size, tier margin theo kích thước vị thế, hàng đợi lệnh limit, thanh khoản sổ lệnh.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .download import BAR_MS


@dataclass
class Costs:
    taker_fee: float = 0.0005      # OKX regular taker 0.05% (kiểm tra tier tài khoản của bạn)
    slip_bps: float = 2.0          # slippage cố định mỗi lần khớp (bps)
    vol_slip_k: float = 0.05       # + k * biên độ nến trước
    mmr: float = 0.005             # maintenance margin rate (xấp xỉ; OKX thật tăng theo tier)
    liq_fee: float = 0.005         # phí thanh lý / notional
    tp_fee: float | None = None    # phí lệnh TP; None = dùng taker_fee (thận trọng)


@dataclass
class Config:
    initial_equity: float = 10_000.0
    exposure: float = 1.0          # notional = exposure * equity lúc vào lệnh (đòn bẩy hiệu dụng)
    max_leverage: float = 3.0
    stop_pct: float | None = None  # stop cứng tính từ giá vào, vd 0.05 = 5%


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)


def run(df: pd.DataFrame, funding: pd.DataFrame, target: pd.Series, bar: str,
        cfg: Config = Config(), costs: Costs = Costs(), mark: pd.DataFrame | None = None) -> Result:
    """`mark`: nến mark price (cột ts, high, low). Nếu có, thanh lý xét theo mark; stop vẫn theo last price."""
    if cfg.exposure > cfg.max_leverage:
        raise ValueError("exposure vượt max_leverage")
    n = len(df)
    ts = df["ts"].to_numpy(dtype=np.int64)
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    if mark is not None:
        m = df[["ts"]].merge(mark[["ts", "high", "low"]], on="ts", how="left")
        mh = m["high"].fillna(df["high"]).to_numpy(float); ml = m["low"].fillna(df["low"]).to_numpy(float)
    else:
        mh, ml = h, l
    desired = np.nan_to_num(target.shift(1).to_numpy(float), nan=0.0)  # quyết định hôm trước -> khớp ở mở cửa nến này
    desired = np.sign(desired)

    # gán funding vào nến: before = trùng đúng giờ mở nến (dùng vị thế trước khi đổi); inside = giữa nến
    f_before: dict[int, float] = {}
    f_inside: dict[int, float] = {}
    if len(funding):
        for f_ts, rate in zip(funding["ts"].to_numpy(np.int64), funding["realized"].to_numpy(float)):
            if f_ts < ts[0] or f_ts >= ts[-1] + BAR_MS[bar]:
                continue
            idx = int(np.searchsorted(ts, f_ts, side="right") - 1)
            bucket = f_before if f_ts == ts[idx] else f_inside
            bucket[idx] = bucket.get(idx, 0.0) + rate

    cash = cfg.initial_equity
    qty = 0.0; entry = 0.0; blocked = 0.0; busted = False
    fees = slip_cost = funding_total = 0.0; liqs = 0
    tr_fee = tr_fund = 0.0; tr_open_ts = 0; tr_open_i = 0
    trades: list[dict] = []
    eq = np.empty(n); in_pos = np.zeros(n, dtype=bool)

    def slip_at(i: int) -> float:
        prev = (h[i - 1] - l[i - 1]) / c[i - 1] if i > 0 else 0.0
        return costs.slip_bps / 1e4 + costs.vol_slip_k * prev

    def close_pos(i: int, price: float, reason: str, extra_fee: float = 0.0):
        nonlocal cash, qty, entry, fees, slip_cost, tr_fee, tr_fund, blocked
        fee = costs.taker_fee * abs(qty) * price + extra_fee
        pnl_gross = qty * (price - entry)
        cash += pnl_gross - fee
        fees += fee; tr_fee += fee
        trades.append({"entry_ts": tr_open_ts, "exit_ts": int(ts[i]), "side": int(np.sign(qty)), "entry": entry,
                       "exit": price, "qty": abs(qty), "gross": pnl_gross, "fee": tr_fee, "funding": tr_fund,
                       "net": pnl_gross - tr_fee + tr_fund, "bars": i - tr_open_i, "reason": reason})
        qty = 0.0; tr_fee = tr_fund = 0.0

    for i in range(n):
        # 1) funding trùng giờ mở nến: dùng vị thế cũ
        if qty != 0 and i in f_before:
            cf = -qty * o[i] * f_before[i]
            cash += cf; funding_total += cf; tr_fund += cf

        # 2) rebalance ở giá mở cửa
        if not busted:
            want = desired[i]
            if blocked != 0 and want != blocked:
                blocked = 0.0
            cur = np.sign(qty)
            if want != cur and not (want != 0 and want == blocked):
                s = slip_at(i)
                if qty != 0:
                    px = o[i] * (1 - cur * s)
                    slip_cost += abs(px - o[i]) * abs(qty)
                    close_pos(i, px, "signal")
                if want != 0 and cash > 0:
                    px = o[i] * (1 + want * s)
                    notional = cfg.exposure * cash
                    qty = want * notional / px
                    fee = costs.taker_fee * abs(qty) * px
                    cash -= fee; fees += fee; tr_fee = fee; tr_fund = 0.0
                    slip_cost += abs(px - o[i]) * abs(qty)
                    entry = px; tr_open_ts = int(ts[i]); tr_open_i = i

        # 3) funding giữa nến
        if qty != 0 and i in f_inside:
            cf = -qty * o[i] * f_inside[i]
            cash += cf; funding_total += cf; tr_fund += cf

        # 4) thanh lý / stop trong nến (giá xấu nhất)
        if qty != 0:
            q = abs(qty)
            if qty > 0:
                p_liq = (q * entry - cash) / (q * (1 - costs.mmr))
                p_stop = entry * (1 - cfg.stop_pct) if cfg.stop_pct else -np.inf
                liq_hit, stop_hit = ml[i] <= p_liq, l[i] <= p_stop
                if liq_hit and (not stop_hit or p_liq >= p_stop):
                    liqs += 1; px = min(p_liq, o[i]); close_pos(i, px, "liquidation", costs.liq_fee * q * px)
                    blocked = 1.0
                elif stop_hit:
                    px = min(p_stop, o[i]) * (1 - slip_at(i)); close_pos(i, px, "stop")
                    blocked = 1.0
            else:
                p_liq = (cash + q * entry) / (q * (1 + costs.mmr))
                p_stop = entry * (1 + cfg.stop_pct) if cfg.stop_pct else np.inf
                liq_hit, stop_hit = mh[i] >= p_liq, h[i] >= p_stop
                if liq_hit and (not stop_hit or p_liq <= p_stop):
                    liqs += 1; px = max(p_liq, o[i]); close_pos(i, px, "liquidation", costs.liq_fee * q * px)
                    blocked = -1.0
                elif stop_hit:
                    px = max(p_stop, o[i]) * (1 + slip_at(i)); close_pos(i, px, "stop")
                    blocked = -1.0
            if cash <= 0:
                cash = 0.0; busted = True; qty = 0.0

        eq[i] = cash + qty * (c[i] - entry) if qty != 0 else cash
        in_pos[i] = qty != 0

    if qty != 0:  # đóng vị thế mở cuối kỳ ở giá đóng cửa để thống kê (không phí) — đánh dấu rõ
        trades.append({"entry_ts": tr_open_ts, "exit_ts": int(ts[-1]), "side": int(np.sign(qty)), "entry": entry,
                       "exit": c[-1], "qty": abs(qty), "gross": qty * (c[-1] - entry), "fee": tr_fee,
                       "funding": tr_fund, "net": qty * (c[-1] - entry) - tr_fee + tr_fund,
                       "bars": n - 1 - tr_open_i, "reason": "open_at_end"})

    equity = pd.Series(eq, index=df["ts"].values, name="equity")
    tdf = pd.DataFrame(trades)
    stats = _stats(equity, tdf, df, bar, cfg, fees, slip_cost, funding_total, liqs, in_pos, busted)
    return Result(equity, tdf, stats)


def _stats(equity, tdf, df, bar, cfg, fees, slip_cost, funding_total, liqs, in_pos, busted) -> dict:
    bpy = 365 * 86_400_000 / BAR_MS[bar]
    rets = equity.pct_change().fillna(0.0)
    years = len(equity) / bpy
    final = float(equity.iloc[-1])
    total = final / cfg.initial_equity - 1
    cagr = (final / cfg.initial_equity) ** (1 / years) - 1 if years > 0 and final > 0 else -1.0
    dd = (equity / equity.cummax() - 1).min()
    sd = rets.std()
    sharpe = float(rets.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0
    neg = rets[rets < 0].std()
    sortino = float(rets.mean() / neg * np.sqrt(bpy)) if neg and neg > 0 else 0.0
    bh = float(df["close"].iloc[-1] / df["open"].iloc[0] - 1)
    s = {"total_return": total, "cagr": cagr, "max_drawdown": float(dd), "sharpe": sharpe, "sortino": sortino,
         "calmar": float(cagr / abs(dd)) if dd < 0 else 0.0, "buy_hold_return": bh,
         "fees": fees, "slippage_cost": slip_cost, "funding_pnl": funding_total, "liquidations": liqs,
         "time_in_market": float(in_pos.mean()), "busted": busted, "trades": int(len(tdf))}
    if len(tdf):
        w = tdf[tdf["net"] > 0]["net"].sum(); lo = -tdf[tdf["net"] < 0]["net"].sum()
        s.update({"win_rate": float((tdf["net"] > 0).mean()), "profit_factor": float(w / lo) if lo > 0 else float("inf"),
                  "avg_trade_net": float(tdf["net"].mean()), "gross_pnl": float(tdf["gross"].sum())})
    return s
