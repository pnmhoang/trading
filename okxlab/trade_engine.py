"""Engine theo lệnh: SL theo ATR, TP = rr × R, khối lượng để mất risk_pct equity nếu chạm SL.

Quy ước tín hiệu: side[i] != 0 là *sự kiện vào lệnh* chốt ở đóng cửa nến i (atr[i] dùng để đặt SL); khớp ở mở
cửa nến i+1. Tín hiệu bị bỏ qua khi đang có vị thế. Thứ tự trong một nến: funding -> vào lệnh -> thanh lý /
SL / TP / time-stop. Cùng nến chạm SL và TP => SL. TP chỉ khớp khi giá đi *qua* mức TP và khớp ở giá TP.

`_core` chỉ dùng mảng numpy và số (không pandas/dict) để `search.engine_nb` biên dịch bằng numba mà không đổi ngữ nghĩa.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import Costs
from .download import BAR_MS

REASONS = ["tp", "sl", "time", "liquidation", "open_at_end"]
TRADE_COLS = ["entry_ts", "exit_ts", "side", "entry", "exit", "qty", "gross", "fee", "funding", "net",
              "r_multiple", "bars", "reason_code", "equity_at_entry"]
DAY_MS = 86_400_000


@dataclass(frozen=True)
class Rules:
    sl_atr: float
    rr: float
    max_hold_bars: int
    risk_pct: float = 0.01
    max_leverage: float = 3.0
    max_cost_r: float = 0.15
    daily_loss_halt: float = 0.03


@dataclass
class TradeResult:
    trades: pd.DataFrame
    equity: np.ndarray
    skipped: int


def _core(ts, o, h, l, c, mh, ml, slip, fb, fi, side, atr,
          taker, mmr, liq_fee, tp_fee, sl_atr, rr, max_hold, risk, max_lev, max_cost_r, halt, equity0):
    n = len(ts)
    out = np.empty((n + 1, 14))
    nt = 0
    eq = np.empty(n)
    cash = equity0
    pos = 0
    qty = 0.0
    entry = 0.0
    sl = 0.0
    tp = 0.0
    entry_i = 0
    entry_ts = 0
    eq_at_entry = 0.0
    tr_fee = 0.0
    tr_fund = 0.0
    skipped = 0
    cur_day = -1
    day_start = cash
    halted = False
    for i in range(n):
        day = ts[i] // DAY_MS
        if day != cur_day:
            cur_day = day
            day_start = cash
            halted = False
        # 1) funding trùng giờ mở nến dùng vị thế cũ
        if pos != 0 and fb[i] != 0.0:
            cf = -qty * o[i] * fb[i]
            cash += cf
            tr_fund += cf
        # 2) vào lệnh ở mở cửa từ tín hiệu nến trước
        if pos == 0 and i >= 1 and side[i - 1] != 0.0 and (not halted) and cash > 0.0:
            s = 1.0 if side[i - 1] > 0 else -1.0
            a = atr[i - 1]
            ok = np.isfinite(a) and a > 0.0
            fill = 0.0
            sl_dist = 0.0
            sl_pct = 1.0
            if ok:
                fill = o[i] * (1.0 + s * slip[i])
                sl_dist = sl_atr * a
                sl_pct = sl_dist / fill
                if risk / sl_pct > max_lev:
                    ok = False
                elif (taker + tp_fee + 2.0 * slip[i]) / sl_pct > max_cost_r:
                    ok = False
            if not ok:
                skipped += 1
            else:
                eq_at_entry = cash
                qty = s * (risk * cash / sl_pct) / fill
                fee = taker * abs(qty) * fill
                cash -= fee
                tr_fee = fee
                tr_fund = 0.0
                entry = fill
                entry_i = i
                entry_ts = ts[i]
                sl = fill - s * sl_dist
                tp = fill + s * rr * sl_dist
                pos = 1 if s > 0 else -1
        # 3) funding giữa nến (sau khi vào lệnh)
        if pos != 0 and fi[i] != 0.0:
            cf = -qty * o[i] * fi[i]
            cash += cf
            tr_fund += cf
        # 4) thoát lệnh
        if pos != 0:
            q = abs(qty)
            reason = -1
            px = 0.0
            rate = taker
            extra = 0.0
            if pos > 0:
                p_liq = (q * entry - cash) / (q * (1.0 - mmr))
                liq_hit = ml[i] <= p_liq
                sl_hit = l[i] <= sl
                tp_hit = h[i] > tp
                if liq_hit and ((not sl_hit) or p_liq >= sl):
                    reason = 3
                    px = min(p_liq, o[i])
                    extra = liq_fee * q * px
                elif sl_hit:
                    reason = 1
                    px = min(sl, o[i]) * (1.0 - slip[i])
                elif tp_hit:
                    reason = 0
                    px = tp
                    rate = tp_fee
            else:
                p_liq = (cash + q * entry) / (q * (1.0 + mmr))
                liq_hit = mh[i] >= p_liq
                sl_hit = h[i] >= sl
                tp_hit = l[i] < tp
                if liq_hit and ((not sl_hit) or p_liq <= sl):
                    reason = 3
                    px = max(p_liq, o[i])
                    extra = liq_fee * q * px
                elif sl_hit:
                    reason = 1
                    px = max(sl, o[i]) * (1.0 + slip[i])
                elif tp_hit:
                    reason = 0
                    px = tp
                    rate = tp_fee
            if reason < 0 and i - entry_i >= max_hold:
                reason = 2
                px = c[i] * (1.0 - pos * slip[i])
            if reason >= 0:
                fee = rate * q * px + extra
                gross = qty * (px - entry)
                cash += gross - fee
                tr_fee += fee
                net = gross - tr_fee + tr_fund
                out[nt, 0] = entry_ts
                out[nt, 1] = ts[i]
                out[nt, 2] = pos
                out[nt, 3] = entry
                out[nt, 4] = px
                out[nt, 5] = q
                out[nt, 6] = gross
                out[nt, 7] = tr_fee
                out[nt, 8] = tr_fund
                out[nt, 9] = net
                out[nt, 10] = net / (risk * eq_at_entry)
                out[nt, 11] = i - entry_i
                out[nt, 12] = reason
                out[nt, 13] = eq_at_entry
                nt += 1
                pos = 0
                qty = 0.0
                if cash <= 0.0:
                    cash = 0.0
                if (cash - day_start) / day_start <= -halt:
                    halted = True
        if pos != 0:
            eq[i] = cash + qty * (c[i] - entry)
        else:
            eq[i] = cash
    if pos != 0:  # vị thế còn mở ở cuối dữ liệu: ghi nhận để thống kê, loại khỏi chỉ số
        gross = qty * (c[n - 1] - entry)
        net = gross - tr_fee + tr_fund
        out[nt, 0] = entry_ts
        out[nt, 1] = ts[n - 1]
        out[nt, 2] = pos
        out[nt, 3] = entry
        out[nt, 4] = c[n - 1]
        out[nt, 5] = abs(qty)
        out[nt, 6] = gross
        out[nt, 7] = tr_fee
        out[nt, 8] = tr_fund
        out[nt, 9] = net
        out[nt, 10] = net / (risk * eq_at_entry)
        out[nt, 11] = n - 1 - entry_i
        out[nt, 12] = 4
        out[nt, 13] = eq_at_entry
        nt += 1
    return out[:nt], eq, skipped


def prepare_inputs(df: pd.DataFrame, funding: pd.DataFrame, mark: pd.DataFrame | None, bar: str, costs: Costs) -> dict:
    """Mảng đầu vào dùng chung cho engine Python và numba."""
    n = len(df)
    ts = df["ts"].to_numpy(np.int64)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    if mark is not None:
        m = df[["ts"]].merge(mark[["ts", "high", "low"]], on="ts", how="left")
        mh = m["high"].fillna(df["high"]).to_numpy(float)
        ml = m["low"].fillna(df["low"]).to_numpy(float)
    else:
        mh, ml = h, l
    rng = np.zeros(n)
    rng[1:] = (h[:-1] - l[:-1]) / c[:-1]
    slip = costs.slip_bps / 1e4 + costs.vol_slip_k * rng
    fb, fi = np.zeros(n), np.zeros(n)
    if len(funding):
        for f_ts, rate in zip(funding["ts"].to_numpy(np.int64), funding["realized"].to_numpy(float)):
            if f_ts < ts[0] or f_ts >= ts[-1] + BAR_MS[bar]:
                continue
            idx = int(np.searchsorted(ts, f_ts, side="right") - 1)
            (fb if f_ts == ts[idx] else fi)[idx] += rate
    return dict(ts=ts, o=o, h=h, l=l, c=c, mh=mh, ml=ml, slip=slip, fb=fb, fi=fi)


def build_result(out: np.ndarray, eq: np.ndarray, skipped: int) -> TradeResult:
    t = pd.DataFrame(out, columns=TRADE_COLS)
    t["reason"] = [REASONS[int(x)] for x in t["reason_code"]]
    for k in ("entry_ts", "exit_ts", "side", "bars"):
        t[k] = t[k].astype("int64")
    return TradeResult(t.drop(columns=["reason_code"]), eq, int(skipped))


def core_args(rules: Rules, costs: Costs, equity0: float) -> tuple:
    tp_fee = costs.taker_fee if costs.tp_fee is None else costs.tp_fee
    return (costs.taker_fee, costs.mmr, costs.liq_fee, tp_fee, rules.sl_atr, rules.rr, rules.max_hold_bars,
            rules.risk_pct, rules.max_leverage, rules.max_cost_r, rules.daily_loss_halt, equity0)


def run_trades(df: pd.DataFrame, funding: pd.DataFrame, mark: pd.DataFrame | None, side: np.ndarray,
               atr: np.ndarray, rules: Rules, costs: Costs, bar: str, initial_equity: float = 10_000.0) -> TradeResult:
    a = prepare_inputs(df, funding, mark, bar, costs)
    out, eq, skipped = _core(a["ts"], a["o"], a["h"], a["l"], a["c"], a["mh"], a["ml"], a["slip"], a["fb"], a["fi"],
                             np.asarray(side, float), np.asarray(atr, float), *core_args(rules, costs, initial_equity))
    return build_result(out, eq, skipped)
