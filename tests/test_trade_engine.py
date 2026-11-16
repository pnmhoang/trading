import numpy as np
import pandas as pd
import pytest

from okxlab.backtest import Costs
from okxlab.trade_engine import Rules, run_trades

H = 3_600_000
ZERO = Costs(taker_fee=0.0, slip_bps=0.0, vol_slip_k=0.0, liq_fee=0.0)
NOFUND = pd.DataFrame({"ts": pd.Series(dtype="int64"), "rate": [], "realized": []})


def bars(o, h=None, l=None, c=None):
    o = np.asarray(o, float)
    h = o + 0.5 if h is None else np.asarray(h, float)
    l = o - 0.5 if l is None else np.asarray(l, float)
    c = o if c is None else np.asarray(c, float)
    return pd.DataFrame({"ts": np.arange(len(o), dtype="int64") * H, "open": o, "high": h, "low": l, "close": c,
                         "vol": 1.0})


def signal(n, at, side=1, atr=1.0):
    s = np.zeros(n); a = np.full(n, np.nan)
    for i in np.atleast_1d(at):
        s[i] = side; a[i] = atr
    return s, a


def go(df, s, a, rules=None, costs=ZERO, funding=NOFUND):
    rules = rules or Rules(sl_atr=1.0, rr=2.0, max_hold_bars=100)
    return run_trades(df, funding, None, s, a, rules, costs, "1H")


def flat(n=8):
    return dict(o=[100.0] * n)


def test_tp_hit_gives_plus_rr():
    h = [100.5, 100.5, 103, 100.5, 100.5, 100.5, 100.5, 100.5]
    df = bars(**{**flat(), "h": h})
    s, a = signal(8, 0)
    r = go(df, s, a)
    t = r.trades.iloc[0]
    assert t["reason"] == "tp" and t["exit"] == pytest.approx(102.0)
    assert t["r_multiple"] == pytest.approx(2.0)


def test_sl_hit_gives_minus_one_r_and_risks_one_percent():
    l = [99.5, 99.5, 98.5, 99.5, 99.5, 99.5, 99.5, 99.5]
    df = bars(**{**flat(), "l": l})
    s, a = signal(8, 0)
    r = go(df, s, a)
    t = r.trades.iloc[0]
    assert t["reason"] == "sl" and t["exit"] == pytest.approx(99.0)
    assert t["r_multiple"] == pytest.approx(-1.0)
    assert t["net"] == pytest.approx(-0.01 * 10_000)


def test_same_bar_sl_and_tp_resolves_to_sl():
    h = [100.5, 103, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    l = [99.5, 98, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5]
    df = bars(**{**flat(), "h": h, "l": l})
    s, a = signal(8, 0)
    t = go(df, s, a).trades.iloc[0]
    assert t["reason"] == "sl"


def test_gap_through_sl_fills_at_open():
    o = [100, 100, 95, 95, 95, 95, 95, 95]
    df = bars(o)
    s, a = signal(8, 0)
    t = go(df, s, a).trades.iloc[0]
    assert t["reason"] == "sl" and t["exit"] == pytest.approx(95.0)
    assert t["r_multiple"] < -1.0


def test_short_tp_and_sl_mirror_long():
    l = [99.5, 99.5, 97, 99.5, 99.5, 99.5, 99.5, 99.5]
    df = bars(**{**flat(), "l": l})
    s, a = signal(8, 0, side=-1)
    t = go(df, s, a).trades.iloc[0]
    assert t["reason"] == "tp" and t["r_multiple"] == pytest.approx(2.0) and t["side"] == -1


def test_skips_when_leverage_needed_exceeds_cap():
    df = bars(**flat())
    s, a = signal(8, 0, atr=0.2)  # SL 0.2% => cần 5x
    r = go(df, s, a)
    assert len(r.trades) == 0 and r.skipped == 1


def test_skips_when_cost_over_max_cost_r_but_allows_wide_sl():
    df = bars(**flat())
    costs = Costs(vol_slip_k=0.0)
    tight = go(df, *signal(8, 0, atr=0.5), costs=costs)  # cost ~0.28R
    wide = go(df, *signal(8, 0, atr=1.0), costs=costs)   # cost ~0.14R
    assert tight.skipped == 1 and len(tight.trades) == 0
    assert wide.skipped == 0 and len(wide.trades) == 1


@pytest.mark.parametrize("bad", [0.0, np.nan, -1.0])
def test_bad_atr_skips_without_error(bad):
    df = bars(**flat())
    s = np.zeros(8); s[0] = 1.0
    a = np.full(8, bad)
    r = go(df, s, a)
    assert len(r.trades) == 0 and r.skipped == 1


def test_daily_loss_halt_blocks_entries_same_day_only():
    n = 60
    o = [100.0] * n
    l = [99.5] * n
    for i in (3, 6, 9, 30):  # nến SL cho các lệnh vào ở i-1... (xem tín hiệu bên dưới)
        l[i] = 98.0
    df = bars(o, l=l)
    s = np.zeros(n); a = np.full(n, np.nan)
    for i in (2, 5, 8, 29):  # vào ở mở nến 3, 6, 9 (ngày 0) và 30 (ngày 1 vì nến 24+)
        s[i] = 1.0; a[i] = 1.0
    rules = Rules(sl_atr=1.0, rr=2.0, max_hold_bars=100, daily_loss_halt=0.015)
    t = go(df, s, a, rules).trades
    entries = list(t["entry_ts"] // H)
    assert entries == [3, 6, 30]  # lệnh thứ 3 trong ngày 0 bị chặn sau 2 lần lỗ ~2% > 1.5%


def test_time_stop_exits_at_close_after_max_hold_bars():
    df = bars(**flat())
    s, a = signal(8, 0)
    rules = Rules(sl_atr=1.0, rr=2.0, max_hold_bars=3)
    t = go(df, s, a, rules).trades.iloc[0]
    assert t["reason"] == "time" and t["bars"] == 3 and t["exit_ts"] == 4 * H


def test_funding_charged_on_open_position():
    df = bars(**flat())
    f = pd.DataFrame({"ts": [3 * H], "rate": [0.001], "realized": [0.001]})
    s, a = signal(8, 0)
    r = go(df, s, a, funding=f)
    assert r.trades.iloc[0]["funding"] == pytest.approx(-10.0)  # qty 100 * giá 100 * 0.1%


def test_open_position_at_end_is_flagged():
    df = bars(**flat())
    s, a = signal(8, 5)
    t = go(df, s, a).trades.iloc[0]
    assert t["reason"] == "open_at_end"


def test_no_lookahead_truncation():
    rng = np.random.default_rng(3)
    n = 1500
    o = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    df = bars(o, h=o * 1.003, l=o * 0.997, c=o)
    s = np.zeros(n); a = np.full(n, np.nan)
    idx = rng.choice(n, 120, replace=False)
    s[idx] = rng.choice([-1.0, 1.0], 120); a[idx] = o[idx] * 0.012
    rules = Rules(sl_atr=1.0, rr=2.0, max_hold_bars=40)
    full = go(df, s, a, rules, Costs())
    k = 900
    cut = go(df.iloc[:k].reset_index(drop=True), s[:k], a[:k], rules, Costs())
    cutoff = df["ts"].iloc[k - 2]
    f = full.trades[full.trades["exit_ts"] <= cutoff].reset_index(drop=True)
    c = cut.trades[cut.trades["exit_ts"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(f, c)
    assert np.allclose(full.equity[: k - 1], cut.equity[: k - 1])
