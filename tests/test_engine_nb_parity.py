import time

import numpy as np
import pandas as pd
import pytest

from okxlab.backtest import Costs
from okxlab.search.constants import load_coin
from okxlab.search.engine_nb import run_prepared, run_trades_fast
from okxlab.trade_engine import Rules, prepare_inputs, _core, core_args, build_result, run_trades

H = 3_600_000


def random_case(seed, n=3000):
    rng = np.random.default_rng(seed)
    o = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    spread = np.abs(rng.normal(0, 0.004, n))
    df = pd.DataFrame({"ts": np.arange(n, dtype="int64") * H, "open": o, "high": o * (1 + spread),
                       "low": o * (1 - spread), "close": o * (1 + rng.normal(0, 0.001, n)), "vol": 1.0})
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    k = int(rng.integers(20, 200))
    idx = rng.choice(n, k, replace=False)
    side = np.zeros(n); atr = np.full(n, np.nan)
    side[idx] = rng.choice([-1.0, 1.0], k)
    atr[idx] = o[idx] * rng.uniform(0.003, 0.03, k)
    funding = pd.DataFrame({"ts": np.arange(0, n, 8, dtype="int64") * H,
                            "realized": rng.normal(0.0001, 0.0003, len(range(0, n, 8)))})
    funding["rate"] = funding["realized"]
    rules = Rules(sl_atr=float(rng.uniform(0.8, 3)), rr=float(rng.choice([2.0, 3.0])),
                  max_hold_bars=int(rng.integers(5, 100)))
    return df, funding, side, atr, rules


def assert_same(a, b):
    assert len(a.trades) == len(b.trades) and a.skipped == b.skipped
    num = ["entry_ts", "exit_ts", "side", "entry", "exit", "r_multiple", "net", "fee", "funding", "bars"]
    assert np.allclose(a.trades[num].to_numpy(float), b.trades[num].to_numpy(float), atol=1e-9, rtol=1e-9)
    assert list(a.trades["reason"]) == list(b.trades["reason"])
    assert np.allclose(a.equity, b.equity, atol=1e-9, rtol=1e-9)


def interpreted(df, funding, side, atr, rules, costs, bar="1H"):
    a = prepare_inputs(df, funding, None, bar, costs)
    out, eq, sk = _core(a["ts"], a["o"], a["h"], a["l"], a["c"], a["mh"], a["ml"], a["slip"], a["fb"], a["fi"],
                        side, atr, *core_args(rules, costs, 10_000.0))
    return build_result(out, eq, sk)


@pytest.mark.parametrize("seed", range(30))
def test_parity_with_interpreted_reference_on_random_cases(seed):
    df, funding, side, atr, rules = random_case(seed)
    costs = Costs()
    fast = run_trades_fast(df, funding, None, side, atr, rules, costs, "1H")
    assert_same(interpreted(df, funding, side, atr, rules, costs), fast)
    assert len(fast.trades) > 0


def test_parity_on_real_btc_slice():
    df, funding, mark = load_coin("BTC", "1H")
    df = df.iloc[:5000].reset_index(drop=True)
    mark = mark.iloc[:5000]
    rng = np.random.default_rng(1)
    idx = rng.choice(5000, 150, replace=False)
    side = np.zeros(5000); atr = np.full(5000, np.nan)
    side[idx] = rng.choice([-1.0, 1.0], 150); atr[idx] = df["close"].to_numpy()[idx] * 0.015
    rules = Rules(sl_atr=1.5, rr=2.0, max_hold_bars=48)
    costs = Costs()
    fast = run_trades_fast(df, funding, mark, side, atr, rules, costs, "1H")
    ref = run_trades(df, funding, mark, side, atr, rules, costs, "1H")
    assert_same(ref, fast)


def test_prepared_core_is_at_least_10x_faster_than_interpreted():
    df, funding, side, atr, rules = random_case(7, n=40_000)
    rng = np.random.default_rng(5)
    idx = rng.choice(40_000, 3_000, replace=False)  # tín hiệu dày để vị thế mở thường xuyên
    side[:] = 0.0; atr[:] = np.nan
    side[idx] = rng.choice([-1.0, 1.0], 3_000); atr[idx] = df["open"].to_numpy()[idx] * 0.01
    costs = Costs()
    prep = prepare_inputs(df, funding, None, "1H", costs)
    run_prepared(prep, side, atr, rules, costs)  # làm nóng JIT
    t0 = time.perf_counter(); fast = run_prepared(prep, side, atr, rules, costs); t_fast = time.perf_counter() - t0
    t0 = time.perf_counter(); ref = run_prepared(prep, side, atr, rules, costs, interpreted=True); t_ref = time.perf_counter() - t0
    assert_same(ref, fast)
    assert t_ref / t_fast >= 10, (t_ref, t_fast)
