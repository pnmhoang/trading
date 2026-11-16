import json

from okxlab.search.ledger import Ledger, code_fingerprint, data_fingerprint


def test_append_and_count(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append({"coin": "BTC", "family": "trend_breakout", "params": {"n": 20}})
    led.append({"coin": "BTC", "family": "trend_breakout", "params": {"n": 40}})
    led.append({"coin": "ETH", "family": "trend_breakout", "params": {"n": 20}})
    assert led.count("BTC", "trend_breakout") == 2
    assert led.count("ETH", "trend_breakout") == 1
    assert led.count("BTC", "range_reversion") == 0


def test_records_are_jsonl_with_hashes(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", data_hash="d1", code_hash="c1")
    led.append({"coin": "BTC", "family": "f"})
    led.append({"coin": "BTC", "family": "f"})
    lines = (tmp_path / "l.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["data_hash"] == "d1" and rec["code_hash"] == "c1" and "ts" in rec


def test_fingerprint_changes_when_file_changes(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x")
    h1 = data_fingerprint([f])
    f.write_text("y")
    assert data_fingerprint([f]) != h1
    assert len(code_fingerprint()) == 64
