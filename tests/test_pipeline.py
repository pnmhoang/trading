import json

import numpy as np
import pandas as pd
import pytest

from okxlab.search.constants import DEV_END_TS
from okxlab.search.families import FAMILIES
from okxlab.search.pipeline import search_coin

H = 3_600_000
START = 1642348800000  # 2022-01-16 16:00 UTC


def synth_loader(kind="noise", seed=0, n=33_000):
    """~3,8 năm nến 1H, kéo dài qua DEV_END_TS để kiểm tra pipeline chỉ dùng dữ liệu dev."""
    rng = np.random.default_rng(seed)
    if kind == "noise":
        ret = rng.normal(0, 0.004, n)
    else:  # chế độ xu hướng đổi chiều mỗi ~250 nến
        drift = np.repeat(rng.choice([-1, 1], n // 250 + 1) * 0.0012, 250)[:n]
        ret = drift + rng.normal(0, 0.003, n)
    c = 100 * np.exp(np.cumsum(ret))
    o = np.concatenate([[100.0], c[:-1]])
    wick = np.abs(rng.normal(0, 0.002, n))
    df = pd.DataFrame({"ts": START + np.arange(n, dtype="int64") * H, "open": o,
                       "high": np.maximum(o, c) * (1 + wick), "low": np.minimum(o, c) * (1 - wick),
                       "close": c, "vol": 1.0})
    funding = pd.DataFrame({"ts": df["ts"].iloc[::8].to_numpy(), "rate": 0.0001, "realized": 0.0001})

    def loader(coin, bar):
        assert bar == "1H"
        return df, funding, None
    return loader


def run(tmp_path, kind="noise", families=None, **kw):
    return search_coin("BTC", tmp_path, families=families, entry_bars=("1H",), data_loader=synth_loader(kind), **kw)


def test_search_on_pure_noise_returns_no_passed_candidates(tmp_path):
    res = run(tmp_path)
    assert res["best"] is None and res["passed"] == []
    assert (tmp_path / "BTC_search.json").exists()
    saved = json.loads((tmp_path / "BTC_search.json").read_text(encoding="utf-8"))
    assert saved["best"] is None and len(saved["table"]) == 4


def test_every_tried_combo_is_in_ledger(tmp_path):
    run(tmp_path, families=["trend_breakout", "range_reversion"])
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    expected = sum(len([p for p in FAMILIES[f].grid if p["entry_bar"] == "1H"]) for f in ("trend_breakout", "range_reversion"))
    assert len(lines) == expected
    rec = json.loads(lines[0])
    assert rec["coin"] == "BTC" and "params" in rec and "code_hash" in rec


def test_dev_only_no_test_period_data_used(tmp_path):
    res = run(tmp_path, families=["trend_breakout"])
    assert res["dev_range"][1] < DEV_END_TS
    loader_last = synth_loader()("BTC", "1H")[0]["ts"].iloc[-1]
    assert loader_last > DEV_END_TS  # loader có dữ liệu sau mốc test; pipeline phải bỏ qua


def test_search_finds_planted_trend_edge(tmp_path):
    res = run(tmp_path, kind="trend", families=["trend_breakout", "squeeze_breakout"])
    assert res["rc_pvalue"] < 0.05
    assert res["best"] is not None and res["best"]["family"] in ("trend_breakout", "squeeze_breakout")
    assert res["best"]["oos_stats"]["expectancy_r"] >= 0.15
    assert all("reasons" in r and isinstance(r["dsr_prob"], float) for r in res["table"])
