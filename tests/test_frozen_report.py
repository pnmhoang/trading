import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, "tests")
from test_pipeline import synth_loader  # noqa: E402

from okxlab.backtest import Costs  # noqa: E402
from okxlab.search.constants import DEV_END_TS  # noqa: E402
from okxlab.search.families import FAMILIES  # noqa: E402
from okxlab.search.frozen import combine_portfolio, is_accepted, run_frozen_test  # noqa: E402
from okxlab.search.report import write_report  # noqa: E402

H, DAY = 3_600_000, 86_400_000


def spec():
    p = next(q for q in FAMILIES["trend_breakout"].grid if q["entry_bar"] == "1H")
    return {"family": "trend_breakout", "params": p}


def test_frozen_test_refuses_second_run(tmp_path):
    run_frozen_test("BTC", spec(), tmp_path, data_loader=synth_loader())
    with pytest.raises(RuntimeError, match="frozen test already used for BTC"):
        run_frozen_test("BTC", spec(), tmp_path, data_loader=synth_loader())


def test_frozen_lock_written_before_run(tmp_path):
    def boom(coin, bar):
        raise ValueError("loader failed")
    with pytest.raises(ValueError):
        run_frozen_test("ETH", spec(), tmp_path, data_loader=boom)
    assert (tmp_path / "frozen_ETH.lock").exists()


def test_frozen_only_trades_in_test_period(tmp_path):
    free = Costs(taker_fee=0.0, slip_bps=0.0, vol_slip_k=0.0, liq_fee=0.0)  # tránh bộ lọc chi phí loại hết tín hiệu
    res = run_frozen_test("BTC", spec(), tmp_path, data_loader=synth_loader(), costs=free)
    t = pd.read_csv(tmp_path / "frozen_BTC_trades.csv")
    assert len(t) > 0 and (t["entry_ts"] >= DEV_END_TS).all()
    assert res["stats"]["n_trades"] == len(t[t["reason"] != "open_at_end"])


def test_accept_thresholds():
    ok = dict(expectancy_r=0.10, win_rate=0.38, n_trades=30)
    assert is_accepted(ok)
    assert not is_accepted({**ok, "expectancy_r": 0.09})
    assert not is_accepted({**ok, "win_rate": 0.37})
    assert not is_accepted({**ok, "n_trades": 0})


def t(entry_h, r, day=0):
    e = day * DAY + entry_h * H
    return {"entry_ts": e, "exit_ts": e + H, "r_multiple": r, "reason": "sl" if r < 0 else "tp"}


def test_combine_portfolio_applies_shared_daily_halt():
    a = pd.DataFrame([t(1, -1.0), t(5, -1.0)])      # BTC: hai lệnh lỗ trong ngày 0
    b = pd.DataFrame([t(8, 2.0), t(30, 2.0)])       # ETH: lệnh giờ 8 (ngày 0, sau halt) bị chặn; lệnh giờ 30 (ngày 1) còn
    out = combine_portfolio({"BTC": a, "ETH": b}, risk_pct=0.01, daily_loss_halt=0.015)
    assert out["skipped_by_halt"] == 1
    assert out["stats"]["n_trades"] == 3


def test_report_states_no_strategy_passed_when_none(tmp_path):
    row = {"family": "trend_breakout", "n_combos": 96, "passed": False, "reasons": ["win_rate 0.30 < 0.40"],
           "oos_stats": {"n_trades": 100, "win_rate": 0.3, "expectancy_r": -0.1, "profit_factor": 0.8, "max_drawdown": -0.3}}
    res = {"BTC": {"search": {"best": None, "table": [row], "n_trials": 96, "rc_pvalue": 0.9}, "frozen": None},
           "ETH": {"search": {"best": None, "table": [row], "n_trials": 96, "rc_pvalue": 0.8}, "frozen": None}}
    out = tmp_path / "report.md"
    write_report(res, out)
    text = out.read_text(encoding="utf-8")
    assert "Không có chiến lược nào đạt" in text and "win_rate 0.30 < 0.40" in text
