import pytest

from okxlab.search.constants import COINS, DEV_END_TS, load_coin


def test_load_coin_trims_to_funding_and_returns_three_frames():
    df, funding, mark = load_coin("BTC", "1H")
    assert df["ts"].iloc[0] >= funding["ts"].iloc[0]
    assert {"ts", "open", "high", "low", "close"} <= set(mark.columns)
    assert len(df) > 1000 and len(funding) > 100


def test_load_coin_unknown_coin_raises():
    with pytest.raises(KeyError):
        load_coin("DOGE", "1H")


def test_constants():
    assert DEV_END_TS == 1751328000000
    assert COINS["ETH"] == "ETH-USDT-SWAP"
