import numpy as np
import pandas as pd
import pytest

from okxlab.backtest import Config, Costs, run
from okxlab.strategies import donchian, ema_trend
from okxlab.validate import validate_candles

H = 3_600_000
ZERO = Costs(taker_fee=0.0, slip_bps=0.0, vol_slip_k=0.0, liq_fee=0.0)
NOFUND = pd.DataFrame({"ts": pd.Series(dtype="int64"), "rate": [], "realized": []})


def make(prices, start=0):
    p = np.asarray(prices, float)
    return pd.DataFrame({"ts": start + np.arange(len(p), dtype="int64") * H, "open": p, "high": p * 1.001,
                         "low": p * 0.999, "close": p, "vol": 1.0, "vol_ccy": 1.0, "vol_quote": 1.0})


def random_walk(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return make(p)


def test_fill_at_next_open_not_signal_bar():
    df = make([100, 100, 110, 110, 110])
    for col in ("high", "low", "close"):
        df[col] = df["open"]  # phẳng trong nến
    tgt = pd.Series([0, 1, 0, 0, 0.0])  # tín hiệu long ở đóng cửa nến 1 -> khớp ở mở cửa nến 2 (110)
    r = run(df, NOFUND, tgt, "1H", Config(), ZERO)
    t = r.trades.iloc[0]
    assert t["entry"] == pytest.approx(110.0)


def test_costs_reduce_equity_on_flat_market():
    df = make([100.0] * 50)
    tgt = pd.Series([0.0] * 5 + [1.0] * 20 + [0.0] * 25)
    r = run(df, NOFUND, tgt, "1H", Config(), Costs(taker_fee=0.0005, slip_bps=2.0, vol_slip_k=0.0))
    # 2 lần khớp: mỗi lần ~ (5 + 2) bps => ~ -14bps
    assert -0.0020 < r.stats["total_return"] < -0.0008
    assert r.stats["fees"] > 0 and r.stats["slippage_cost"] > 0


def test_funding_long_pays_positive_rate():
    df = make([100.0] * 30)
    f = pd.DataFrame({"ts": [10 * H], "rate": [0.001], "realized": [0.001]})
    r = run(df, f, pd.Series([1.0] * 30), "1H", Config(exposure=1.0), ZERO)
    assert r.stats["funding_pnl"] == pytest.approx(-10_000 * 0.001, rel=1e-6)
    short = run(df, f, pd.Series([-1.0] * 30), "1H", Config(exposure=1.0), ZERO)
    assert short.stats["funding_pnl"] == pytest.approx(+10.0, rel=1e-6)


def test_liquidation_on_crash_and_no_negative_equity():
    prices = [100.0] * 5 + [100, 90, 70, 50, 50, 50]
    df = make(prices)
    r = run(df, NOFUND, pd.Series([1.0] * len(prices)), "1H", Config(exposure=3.0, max_leverage=3.0), Costs())
    assert r.stats["liquidations"] == 1
    assert (r.equity >= 0).all()
    assert r.trades.iloc[0]["reason"] == "liquidation"


def test_stop_loss_gap_fills_at_open():
    prices = [100.0] * 4 + [80.0, 80.0]
    df = make(prices)
    r = run(df, NOFUND, pd.Series([1.0] * 6), "1H", Config(stop_pct=0.05), ZERO)
    assert r.trades.iloc[0]["reason"] == "stop"
    assert r.trades.iloc[0]["exit"] == pytest.approx(80.0)  # gap xuống dưới stop => khớp ở giá mở, không ở giá stop


@pytest.mark.parametrize("fn,kw", [(donchian, dict(entry_n=30, exit_n=10)), (ema_trend, dict(fast=10, slow=50))])
def test_signals_have_no_lookahead(fn, kw):
    df = random_walk()
    full = fn(df, **kw)
    for k in (300, 700, 1200):
        part = fn(df.iloc[:k].reset_index(drop=True), **kw)
        assert np.array_equal(part.to_numpy(), full.iloc[:k].to_numpy())


def test_engine_has_no_lookahead():
    df = random_walk()
    tgt = donchian(df, 30, 10)
    k = 900
    full = run(df, NOFUND, tgt, "1H", Config(), Costs())
    cut = run(df.iloc[:k].reset_index(drop=True), NOFUND, tgt.iloc[:k], "1H", Config(), Costs())
    assert np.allclose(full.equity.to_numpy()[: k - 1], cut.equity.to_numpy()[: k - 1])


def test_validate_detects_problems():
    df = make(np.linspace(100, 110, 100))
    assert validate_candles(df, "1H")["ok"]
    gap = df.drop(index=range(40, 60)).reset_index(drop=True)
    codes = {i["code"] for i in validate_candles(gap, "1H")["issues"]}
    assert "gaps" in codes
    dup = pd.concat([df, df.iloc[[5]]]).reset_index(drop=True)
    assert not validate_candles(dup, "1H")["ok"]
    bad = df.copy(); bad.loc[10, "high"] = bad.loc[10, "low"] - 1
    assert not validate_candles(bad, "1H")["ok"]


def test_liquidation_uses_mark_price_when_given():
    prices = [100.0] * 8
    df = make(prices)
    df.loc[5, "low"] = 60.0  # last price wick xuống 60 (3x long thanh lý ~ 67)
    tgt = pd.Series([1.0] * 8)
    cfg = Config(exposure=3.0, max_leverage=3.0)
    by_last = run(df, NOFUND, tgt, "1H", cfg, Costs())
    assert by_last.stats["liquidations"] == 1
    mark = df[["ts", "high", "low"]].copy()
    mark["low"] = 99.0  # mark price không chạm -> không thanh lý
    by_mark = run(df, NOFUND, tgt, "1H", cfg, Costs(), mark)
    assert by_mark.stats["liquidations"] == 0
