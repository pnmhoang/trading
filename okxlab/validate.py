"""Kiểm tra chất lượng dữ liệu trước khi backtest. Lỗi 'critical' => không nên backtest."""
from __future__ import annotations

import pandas as pd

from .download import BAR_MS


def validate_candles(df: pd.DataFrame, bar: str, max_move: float = 0.25) -> dict:
    step = BAR_MS[bar]
    issues: list[dict] = []

    def add(level: str, code: str, detail):
        issues.append({"level": level, "code": code, "detail": detail})

    if df.empty:
        add("critical", "empty", "Không có dữ liệu")
        return {"ok": False, "rows": 0, "issues": issues}

    ts = df["ts"]
    dups = int(ts.duplicated().sum())
    if dups:
        add("critical", "duplicate_ts", dups)
    if not ts.is_monotonic_increasing:
        add("critical", "not_sorted", "ts không tăng dần")
    if (ts % step != 0).any():
        add("critical", "misaligned_ts", int((ts % step != 0).sum()))

    # khoảng trống
    d = ts.diff().dropna()
    gaps = d[d > step]
    missing = int(((gaps // step) - 1).sum())
    if missing:
        biggest = gaps.sort_values(ascending=False).head(5)
        detail = {"missing_bars": missing, "gap_count": int(len(gaps)),
                  "largest": [{"after_ts": int(ts[i - 1]), "bars": int(g // step - 1)} for i, g in biggest.items()]}
        pct = missing / (missing + len(df))
        add("critical" if pct > 0.005 else "warning", "gaps", {**detail, "missing_pct": round(pct * 100, 3)})

    # OHLC hợp lệ
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bad_ohlc = int(((h < pd.concat([o, c], axis=1).max(axis=1)) | (l > pd.concat([o, c], axis=1).min(axis=1)) | (h < l)).sum())
    if bad_ohlc:
        add("critical", "bad_ohlc", bad_ohlc)
    nonpos = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if nonpos:
        add("critical", "non_positive_price", nonpos)

    # nến "chết": không volume, giá phẳng
    dead = int(((df["vol"] == 0) & (h == l)).sum())
    if dead:
        add("warning", "zero_volume_flat", dead)

    # biến động bất thường close->close và wick quá lớn
    ret = c.pct_change().abs()
    jumps = ret[ret > max_move]
    if len(jumps):
        add("warning", "extreme_return", [{"ts": int(ts[i]), "ret": round(float(r), 4)} for i, r in jumps.head(5).items()])
    wick = ((h - l) / o)
    big_wick = int((wick > 0.5).sum())
    if big_wick:
        add("warning", "huge_range_bar", big_wick)

    ok = not any(i["level"] == "critical" for i in issues)
    return {"ok": ok, "rows": int(len(df)), "first_ts": int(ts.iloc[0]), "last_ts": int(ts.iloc[-1]), "issues": issues}


def validate_funding(fdf: pd.DataFrame, cdf: pd.DataFrame) -> dict:
    issues: list[dict] = []
    if fdf.empty:
        return {"ok": False, "rows": 0, "issues": [{"level": "critical", "code": "empty", "detail": "Không có funding"}]}
    d = fdf["ts"].diff().dropna()
    common = d.mode().iloc[0]
    irregular = int((d != common).sum())
    if irregular:
        issues.append({"level": "warning", "code": "irregular_interval",
                       "detail": {"common_ms": int(common), "irregular": irregular}})
    if fdf["ts"].duplicated().any():
        issues.append({"level": "critical", "code": "duplicate_ts", "detail": int(fdf["ts"].duplicated().sum())})
    if (fdf["rate"].abs() > 0.02).any():
        issues.append({"level": "warning", "code": "extreme_rate", "detail": int((fdf["rate"].abs() > 0.02).sum())})
    # funding có phủ hết giai đoạn nến không?
    if len(cdf):
        gap_start = int(fdf["ts"].iloc[0] - cdf["ts"].iloc[0])
        if gap_start > common:
            issues.append({"level": "critical", "code": "funding_starts_late",
                           "detail": f"Funding bắt đầu muộn hơn nến {gap_start / 3_600_000:.1f}h — backtest sẽ bỏ sót chi phí funding giai đoạn đầu"})
    ok = not any(i["level"] == "critical" for i in issues)
    return {"ok": ok, "rows": int(len(fdf)), "first_ts": int(fdf["ts"].iloc[0]), "last_ts": int(fdf["ts"].iloc[-1]), "issues": issues}
