import numpy as np
import pandas as pd
import pytest

from okxlab.search import select as sel

DAY = 86_400_000
Y = {2022: 1640995200000, 2023: 1672531200000, 2024: 1704067200000, 2025: 1735689600000}


def trades(rs, start=Y[2023], step=DAY, reason="tp"):
    n = len(rs)
    return pd.DataFrame({"entry_ts": start + np.arange(n, dtype="int64") * step,
                         "exit_ts": start + np.arange(n, dtype="int64") * step + 3_600_000,
                         "r_multiple": np.asarray(rs, float), "reason": reason})


def test_windows_do_not_overlap_and_are_ordered():
    w = sel.walk_forward_windows(Y[2022], Y[2025] + 182 * DAY)
    assert len(w) == 10
    for (a, b, c, d), nxt in zip(w, w[1:] + [None]):
        assert a < b == c < d
        if nxt:
            assert d == nxt[2]  # test liền kề, không chồng
    assert w[-1][3] <= Y[2025] + 182 * DAY


def test_trade_stats_known_example():
    s = sel.trade_stats(trades([2, -1, -1, 2]))
    assert s["n_trades"] == 4 and s["win_rate"] == 0.5
    assert s["expectancy_r"] == pytest.approx(0.5) and s["profit_factor"] == pytest.approx(2.0)
    assert s["max_drawdown"] == pytest.approx(0.99 * 0.99 - 1)


def test_trade_stats_excludes_open_at_end_and_handles_empty():
    t = pd.concat([trades([2, -1]), trades([5.0], reason="open_at_end")], ignore_index=True)
    assert sel.trade_stats(t)["n_trades"] == 2
    e = sel.trade_stats(trades([]))
    assert e["n_trades"] == 0 and e["win_rate"] == 0 and e["expectancy_r"] == 0


def test_trade_stats_drawdown_value():
    s = sel.trade_stats(trades([-1.0] * 5))  # 5 lần lỗ 1% liên tiếp
    assert s["max_drawdown"] == pytest.approx(0.99 ** 5 - 1)


def test_phase_positive_count():
    t = pd.concat([trades([2, -1], start=Y[2022]), trades([-1, -1, 1], start=Y[2023]),
                   trades([3], start=Y[2024])], ignore_index=True)
    assert sel.phase_positive_count(t) == 2  # 2022 (+1) và 2024 (+3); 2023 = -1


def good_stats(**kw):
    s = dict(n_trades=120, win_rate=0.45, expectancy_r=0.25, profit_factor=1.6, max_drawdown=-0.12, sharpe=0.1)
    return {**s, **kw}


def test_passes_robust_accepts_good_and_reports_each_failed_rule():
    ok, why = sel.passes_robust(good_stats(), 0.7, 3, True)
    assert ok and why == []
    bad = good_stats(n_trades=50, win_rate=0.3, expectancy_r=0.05, profit_factor=1.0, max_drawdown=-0.4)
    ok, why = sel.passes_robust(bad, 0.4, 1, False)
    assert not ok and len(why) == 8


def test_deflated_sharpe_decreases_with_more_trials():
    a = sel.deflated_sharpe_prob(0.15, 300, 1, 0.0, 3.0)
    b = sel.deflated_sharpe_prob(0.15, 300, 200, 0.0, 3.0)
    assert 0 <= b < a <= 1


def test_reality_check_noise_not_significant():
    rng = np.random.default_rng(0)
    assert sel.reality_check_pvalue(rng.normal(0, 1, (300, 400)), n_boot=500) > 0.1


def test_reality_check_detects_planted_edge():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 1, (300, 400))
    m[17] += 0.4
    assert sel.reality_check_pvalue(m, n_boot=500) < 0.05


def test_select_walk_forward_uses_only_train_to_choose():
    tr, te = Y[2023], Y[2023] + 365 * DAY
    # combo 0: tốt ở train, tệ ở test. combo 1: vừa phải ở train, tốt ở test
    c0 = pd.concat([trades([1.0] * 30, start=tr), trades([-1.0] * 10, start=te)], ignore_index=True)
    c1 = pd.concat([trades([0.2] * 30, start=tr), trades([2.0] * 10, start=te)], ignore_index=True)
    windows = [(tr, tr + 30 * DAY, te, te + 30 * DAY)]
    oos, info = sel.select_params_walk_forward({0: c0, 1: c1}, windows)
    assert info["chosen"] == [0]
    assert oos["r_multiple"].sum() == pytest.approx(-10.0)
    assert info["window_win_pct"] == 0.0


def test_select_skips_window_when_no_combo_has_enough_train_trades():
    tr = Y[2023]
    c = trades([1.0] * 5, start=tr)
    oos, info = sel.select_params_walk_forward({0: c}, [(tr, tr + 30 * DAY, tr + 30 * DAY, tr + 60 * DAY)])
    assert info["chosen"] == [None] and len(oos) == 0
