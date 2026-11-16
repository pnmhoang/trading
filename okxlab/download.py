"""Tải dữ liệu công khai của OKX (không cần API key): nến, funding rate, thông số hợp đồng.

Mọi file tải về đều được ghi vào data/manifest.json cùng sha256 để truy vết phiên bản dữ liệu.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.okx.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CANDLE_COLS = ["ts", "open", "high", "low", "close", "vol", "vol_ccy", "vol_quote", "confirm"]

BAR_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1H": 3_600_000, "2H": 7_200_000, "4H": 14_400_000, "1D": 86_400_000,
}


def _get(path: str, params: dict, retries: int = 6) -> list:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
            body = r.json()
            if body.get("code") == "0":
                return body["data"]
            if body.get("code") == "50011":  # rate limit
                time.sleep(1 + attempt)
                continue
            raise RuntimeError(f"OKX error {body.get('code')}: {body.get('msg')}")
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"Request failed after {retries} tries: {path} {params} ({last})")


def _paginate(path: str, params: dict, start_ms: int, end_ms: int, ts_of, sleep: float = 0.11) -> list:
    """OKX trả dữ liệu mới -> cũ; `after` = lấy bản ghi cũ hơn mốc này."""
    out: list = []
    after = end_ms
    while True:
        data = _get(path, {**params, "after": after, "limit": 100})
        if not data:
            break
        out.extend(data)
        oldest = min(ts_of(x) for x in data)
        if oldest <= start_ms or oldest >= after:
            break
        after = oldest
        time.sleep(sleep)
    return out


def fetch_candles(inst: str, bar: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = _paginate("/api/v5/market/history-candles", {"instId": inst, "bar": bar},
                     start_ms, end_ms, lambda x: int(x[0]))
    df = pd.DataFrame(rows, columns=CANDLE_COLS)
    if df.empty:
        return df.assign(ts=pd.Series(dtype="int64"))
    df = df[df["confirm"] == "1"].drop(columns="confirm")  # bỏ nến chưa đóng
    df["ts"] = df["ts"].astype("int64")
    for c in ["open", "high", "low", "close", "vol", "vol_ccy", "vol_quote"]:
        df[c] = df[c].astype(float)
    df = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_mark_candles(inst: str, bar: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Nến mark price (OKX tính thanh lý theo mark price, không theo last price)."""
    rows = _paginate("/api/v5/market/history-mark-price-candles", {"instId": inst, "bar": bar},
                     start_ms, end_ms, lambda x: int(x[0]))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "confirm"])
    if df.empty:
        return df.drop(columns="confirm").assign(ts=pd.Series(dtype="int64"))
    df = df[df["confirm"] == "1"].drop(columns="confirm")
    df["ts"] = df["ts"].astype("int64")
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_funding(inst: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = _paginate("/api/v5/public/funding-rate-history", {"instId": inst},
                     start_ms, end_ms, lambda x: int(x["fundingTime"]))
    df = pd.DataFrame([{"ts": int(r["fundingTime"]), "rate": float(r["fundingRate"]),
                        "realized": float(r["realizedRate"] or r["fundingRate"])} for r in rows])
    if df.empty:
        return pd.DataFrame({"ts": pd.Series(dtype="int64"), "rate": [], "realized": []})
    df = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_funding_files(inst: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Funding dài hạn từ file tháng của OKX (API lịch sử chỉ giữ ~3 tháng).
    File tải về coi là dữ liệu không tin cậy: chỉ đọc CSV trong zip, giới hạn dung lượng, parse bằng pandas."""
    import io
    import zipfile

    family = inst.replace("-SWAP", "")
    frames, window = [], 9 * 30 * 86_400_000
    a = start_ms
    while a < end_ms:
        b = min(a + window, end_ms)
        data = _get("/api/v5/public/market-data-history",
                    {"module": 3, "instType": "SWAP", "instFamilyList": family,
                     "dateAggrType": "monthly", "begin": a, "end": b})
        for block in data:
            for det in block.get("details", []):
                for g in det.get("groupDetails", []):
                    url = g["url"]
                    if not url.startswith("https://static.okx.com/"):
                        continue
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        for info in z.infolist():
                            if info.filename.endswith(".csv") and info.file_size < 5_000_000:
                                frames.append(pd.read_csv(io.BytesIO(z.read(info))))
                    time.sleep(0.2)
        a = b
    if not frames:
        return pd.DataFrame({"ts": pd.Series(dtype="int64"), "rate": [], "realized": []})
    df = pd.concat(frames)
    df = df[df["instrument_name"] == inst]
    out = pd.DataFrame({"ts": df["funding_time"].astype("int64"), "rate": df["funding_rate"].astype(float)})
    out["realized"] = out["rate"]
    out = out[(out["ts"] >= start_ms) & (out["ts"] < end_ms)]
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_instrument(inst: str) -> dict:
    data = _get("/api/v5/public/instruments", {"instType": "SWAP", "instId": inst})
    if not data:
        raise RuntimeError(f"Không tìm thấy instrument {inst}")
    return data[0]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_manifest(path: Path, df: pd.DataFrame) -> None:
    mpath = DATA_DIR / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest[path.name] = {
        "sha256": _sha256(path),
        "rows": int(len(df)),
        "first_ts": int(df["ts"].iloc[0]) if len(df) else None,
        "last_ts": int(df["ts"].iloc[-1]) if len(df) else None,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    mpath.write_text(json.dumps(manifest, indent=2))


def _merge_save(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new]).drop_duplicates("ts", keep="last")
    new = new.sort_values("ts").reset_index(drop=True)
    new.to_csv(path, index=False)
    _update_manifest(path, new)
    return new


def candle_path(inst: str, bar: str) -> Path:
    return DATA_DIR / f"{inst}_{bar}_candles.csv"


def funding_path(inst: str) -> Path:
    return DATA_DIR / f"{inst}_funding.csv"


def mark_path(inst: str, bar: str) -> Path:
    return DATA_DIR / f"{inst}_{bar}_mark.csv"


def download_mark(inst: str, bar: str, start: str, end: str | None = None) -> int:
    DATA_DIR.mkdir(exist_ok=True)
    to_ms = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)
    start_ms, end_ms = to_ms(start), (to_ms(end) if end else int(time.time() * 1000))
    p = mark_path(inst, bar)
    if p.exists():
        last = int(pd.read_csv(p, usecols=["ts"])["ts"].iloc[-1])
        if last >= start_ms:
            start_ms = last + BAR_MS[bar]
    return len(_merge_save(p, fetch_mark_candles(inst, bar, start_ms, end_ms)))


def download(inst: str, bar: str, start: str, end: str | None = None) -> dict:
    """start/end dạng YYYY-MM-DD (UTC). Nếu file đã có, tải bù phần còn thiếu ở cuối."""
    DATA_DIR.mkdir(exist_ok=True)
    to_ms = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)
    start_ms = to_ms(start)
    end_ms = to_ms(end) if end else int(time.time() * 1000)

    cpath = candle_path(inst, bar)
    if cpath.exists():
        last = int(pd.read_csv(cpath, usecols=["ts"])["ts"].iloc[-1])
        fetch_from = min(start_ms, last + BAR_MS[bar]) if last < start_ms else last + BAR_MS[bar]
    else:
        fetch_from = start_ms
    candles = _merge_save(cpath, fetch_candles(inst, bar, fetch_from, end_ms))

    fpath = funding_path(inst)
    # file tháng (dài hạn) trước, API (mới nhất, có realizedRate) ghi đè sau — keep="last" trong _merge_save
    _merge_save(fpath, fetch_funding_files(inst, start_ms, end_ms))
    funding = _merge_save(fpath, fetch_funding(inst, start_ms, end_ms))

    inst_info = fetch_instrument(inst)
    (DATA_DIR / f"{inst}_instrument.json").write_text(json.dumps(inst_info, indent=2))
    return {"candles": len(candles), "funding": len(funding), "instrument": inst_info}
