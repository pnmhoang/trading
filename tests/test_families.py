import numpy as np
import pandas as pd
import pytest

from okxlab.download import BAR_MS
from okxlab.search import indicators as ind
from okxlab.search.families import FAMILIES


def synth(n, step_ms, seed=0):
    rng = np.random.default_rng(seed)
    o = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)) + 0.3 * np.sin(np.arange(n) / 150))
    c = o * (1 + rng.normal(0, 0.002, n))
    spread = np.abs(rng.normal(0, 0.003, n))
    return pd.DataFrame({"ts": np.arange(n, dtype="int64") * step_ms, "open": o,
                         "high": np.maximum(o, c) * (1 + spread), "low": np.minimum(o, c) * (1 - spread),
                         "close": c, "vol": 1.0})


def frames(entry_bar, n):
    htf_bar = "1H" if entry_bar == "15m" else "4H"
    e = synth(n, BAR_MS[entry_bar])
    return e, ind.resample_ohlc(e, BAR_MS[htf_bar])


def test_indicators_known_values():
    df = pd.DataFrame({"high": [11.0] * 30, "low": [9.0] * 30, "close": [10.0] * 30})
    assert np.allclose(ind.atr(df, 14)[14:], 2.0)
    assert np.isnan(ind.atr(df, 14)[:13]).all()
    assert np.allclose(ind.ema(np.full(20, 5.0), 10), 5.0)
    up = ind.rsi(np.arange(1.0, 60.0), 14)
    assert up[-1] == pytest.approx(100.0)
    mid, upper, lower = ind.bollinger(np.full(30, 7.0), 20, 2.0)
    assert mid[-1] == 7.0 and upper[-1] == 7.0 and lower[-1] == 7.0


def test_resample_ohlc_aggregates():
    df = synth(8, BAR_MS["1H"])
    r = ind.resample_ohlc(df, BAR_MS["4H"])
    assert len(r) == 2
    assert r["open"].iloc[0] == df["open"].iloc[0] and r["close"].iloc[0] == df["close"].iloc[3]
    assert r["high"].iloc[1] == df["high"].iloc[4:8].max()


def test_align_htf_never_uses_unclosed_htf_bar():
    e = synth(8, BAR_MS["15m"])
    htf = ind.resample_ohlc(e, BAR_MS["1H"])
    values = np.array([10.0, 20.0])  # giá trị của HTF bar 0 và 1
    out = ind.align_htf(e, htf, values, "15m", "1H")
    # nến 15m thứ 4 (index 3) đóng cùng lúc HTF bar 0 đóng => dùng được; các nến trước đó thì chưa
    assert np.isnan(out[:3]).all()
    assert out[3] == 10.0 and out[4] == 10.0 and out[6] == 10.0 and out[7] == 20.0


@pytest.mark.parametrize("name", list(FAMILIES))
def test_grid_sizes_within_limit(name):
    g = FAMILIES[name].grid
    assert 0 < len(g) <= 200
    assert all(p["rr"] in (2.0, 3.0) and p["entry_bar"] in ("15m", "1H") for p in g)
    assert len({tuple(sorted(p.items())) for p in g}) == len(g)


@pytest.mark.parametrize("name", list(FAMILIES))
@pytest.mark.parametrize("entry_bar", ["15m", "1H"])
def test_signals_are_causal(name, entry_bar):
    fam = FAMILIES[name]
    params = next(p for p in fam.grid if p["entry_bar"] == entry_bar)
    e, htf = frames(entry_bar, 4000)
    side, atr = fam.signal(e, htf, params)
    assert len(side) == len(atr) == len(e)
    assert set(np.unique(side)) <= {-1.0, 0.0, 1.0}
    for k in (900, 1700, 3100):
        ek = e.iloc[:k].reset_index(drop=True)
        hk = ind.resample_ohlc(ek, BAR_MS["1H"] if entry_bar == "15m" else BAR_MS["4H"])
        sk, ak = fam.signal(ek, hk, params)
        assert np.array_equal(sk, side[:k])
        assert np.allclose(ak, atr[:k], equal_nan=True)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_side_nonzero_only_where_atr_is_valid_and_some_signals_exist(name):
    fam = FAMILIES[name]
    e, htf = frames("1H", 12_000)
    counts = []
    for p in [q for q in fam.grid if q["entry_bar"] == "1H"][:40:7]:
        side, atr = fam.signal(e, htf, p)
        assert np.isfinite(atr[side != 0]).all()
        counts.append(int((side != 0).sum()))
    assert max(counts) > 0
