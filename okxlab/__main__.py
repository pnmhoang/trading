"""CLI:
  python -m okxlab download  --inst BTC-USDT-SWAP --bar 1H --start 2022-01-01
  python -m okxlab validate  --inst BTC-USDT-SWAP --bar 1H
  python -m okxlab backtest  --inst BTC-USDT-SWAP --bar 1H --strategy donchian --params entry_n=55,exit_n=20
  python -m okxlab walkforward --inst BTC-USDT-SWAP --bar 1H --train-days 365 --test-days 90
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .backtest import Config, Costs, run
from .download import BAR_MS, candle_path, download, download_mark, funding_path, mark_path
from .strategies import STRATEGIES
from .validate import validate_candles, validate_funding
from .walkforward import grid, sensitivity, walk_forward


def load(inst, bar, trim=True):
    df, f = pd.read_csv(candle_path(inst, bar)), pd.read_csv(funding_path(inst))
    if trim and len(f) and df["ts"].iloc[0] < f["ts"].iloc[0]:
        # chỉ backtest trong giai đoạn có đủ funding (nếu không sẽ bỏ sót chi phí)
        df = df[df["ts"] >= f["ts"].iloc[0]].reset_index(drop=True)
        print(f"[info] Cắt nến để bắt đầu từ {pd.to_datetime(df['ts'].iloc[0], unit='ms')} (khi funding bắt đầu)")
    return df, f


def parse_params(s):
    out = {}
    for kv in filter(None, (s or "").split(",")):
        k, v = kv.split("=")
        out[k] = int(v) if v.lstrip("-").isdigit() else (v == "true" if v in ("true", "false") else float(v))
    return out


def fmt(d):
    return json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}, indent=2)


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="okxlab")
    sub = p.add_subparsers(dest="cmd", required=True)
    for nm in ("frozen", "report"):
        q = sub.add_parser(nm)
        if nm == "frozen":
            q.add_argument("--coin", required=True, choices=["BTC", "ETH"])
        q.add_argument("--out", default="results")
    sp = sub.add_parser("search")
    sp.add_argument("--coin", required=True, choices=["BTC", "ETH"])
    sp.add_argument("--families")
    sp.add_argument("--entry-bars", default="15m,1H")
    sp.add_argument("--out", default="results")
    for name in ("download", "validate", "backtest", "walkforward"):
        s = sub.add_parser(name)
        s.add_argument("--inst", default="BTC-USDT-SWAP")
        s.add_argument("--bar", default="1H", choices=list(BAR_MS))
        if name == "download":
            s.add_argument("--start", required=True); s.add_argument("--end")
            s.add_argument("--mark", action="store_true", help="tải thêm mark price cùng khung")
            s.add_argument("--mark-only", action="store_true")
        if name in ("backtest", "walkforward"):
            s.add_argument("--strategy", default="donchian", choices=list(STRATEGIES))
            s.add_argument("--exposure", type=float, default=1.0)
            s.add_argument("--stop-pct", type=float)
            s.add_argument("--taker-fee", type=float, default=0.0005)
            s.add_argument("--slip-bps", type=float, default=2.0)
            s.add_argument("--allow-invalid-data", action="store_true")
            s.add_argument("--use-mark", action="store_true", help="thanh lý theo mark price (backtest)")
        if name == "backtest":
            s.add_argument("--params", default="")
            s.add_argument("--sensitivity", help="tên tham số để quét ±20%%, vd entry_n")
        if name == "walkforward":
            s.add_argument("--train-days", type=int, default=365); s.add_argument("--test-days", type=int, default=90)
    a = p.parse_args()

    if a.cmd == "frozen":
        from pathlib import Path

        from .search.frozen import run_frozen_test
        sj = Path(a.out) / f"{a.coin}_search.json"
        best = json.loads(sj.read_text(encoding="utf-8"))["best"]
        if best is None:
            print(f"{a.coin}: không có ứng viên qua bộ lọc; không chạy test đóng băng.")
            return
        res = run_frozen_test(a.coin, {"family": best["family"], "params": best["params"]}, Path(a.out))
        print(fmt(res["stats"]), "ACCEPTED" if res["accepted"] else "REJECTED")
        return

    if a.cmd == "report":
        from pathlib import Path

        from .search.frozen import combine_portfolio
        from .search.report import write_report
        out = Path(a.out)
        results, trades = {}, {}
        for coin in ("BTC", "ETH"):
            sj = out / f"{coin}_search.json"
            if not sj.exists():
                continue
            fj = out / f"frozen_{coin}.json"
            fz = json.loads(fj.read_text(encoding="utf-8")) if fj.exists() else None
            results[coin] = {"search": json.loads(sj.read_text(encoding="utf-8")), "frozen": fz}
            if fz and fz["accepted"]:
                trades[coin] = pd.read_csv(out / f"frozen_{coin}_trades.csv")
        port = combine_portfolio(trades) if len(trades) == 2 else None
        write_report(results, out / "report.md", port)
        print(f"Đã ghi {out / 'report.md'}")
        return

    if a.cmd == "search":
        from pathlib import Path

        from .search.pipeline import search_coin
        res = search_coin(a.coin, Path(a.out), a.families.split(",") if a.families else None, tuple(a.entry_bars.split(",")))
        for t in res["table"]:
            s = t["oos_stats"]
            print(f"{t['family']:17s} oos_n={s['n_trades']:4d} win={s['win_rate']:.2f} exp={s['expectancy_r']:+.2f}R "
                  f"pf={s['profit_factor']:.2f} dd={s['max_drawdown']:.2f} dsr={t['dsr_prob']:.2f} "
                  f"{'PASS' if t['passed'] else 'FAIL: ' + '; '.join(t['reasons'][:3])}")
        print(f"trials={res['n_trials']} reality_check_p={res['rc_pvalue']:.3f} best={'none' if res['best'] is None else res['best']['family']}")
        return

    if a.cmd == "download":
        if a.mark_only:
            print(f"Đã lưu {download_mark(a.inst, a.bar, a.start, a.end)} nến mark price")
            return
        res = download(a.inst, a.bar, a.start, a.end)
        if a.mark:
            print(f"Đã lưu {download_mark(a.inst, a.bar, a.start, a.end)} nến mark price")
        print(f"Đã lưu: {res['candles']} nến, {res['funding']} funding. ctVal={res['instrument'].get('ctVal')} lotSz={res['instrument'].get('lotSz')}")
        return
    df, f = load(a.inst, a.bar)
    if a.cmd == "validate":
        rc, rf = validate_candles(df, a.bar), validate_funding(f, df)
        print("CANDLES:", fmt({"ok": rc["ok"], "rows": rc["rows"]})); print(json.dumps(rc["issues"], indent=1))
        print("FUNDING:", fmt({"ok": rf["ok"], "rows": rf["rows"]})); print(json.dumps(rf["issues"], indent=1))
        raise SystemExit(0 if rc["ok"] and rf["ok"] else 1)

    rc, rf = validate_candles(df, a.bar), validate_funding(f, df)
    if not (rc["ok"] and rf["ok"]) and not a.allow_invalid_data:
        raise SystemExit("Dữ liệu không đạt kiểm tra. Chạy `validate` để xem chi tiết (hoặc --allow-invalid-data).")
    cfg = Config(exposure=a.exposure, max_leverage=max(3.0, a.exposure), stop_pct=a.stop_pct)
    costs = Costs(taker_fee=a.taker_fee, slip_bps=a.slip_bps)
    fn = STRATEGIES[a.strategy]

    if a.cmd == "backtest":
        params = parse_params(a.params)
        mark = pd.read_csv(mark_path(a.inst, a.bar)) if a.use_mark else None
        r = run(df, f, fn(df, **params), a.bar, cfg, costs, mark)
        print(fmt(r.stats))
        if a.sensitivity:
            base = params[a.sensitivity]
            vals = sorted({max(2, int(round(base * m))) for m in (0.6, 0.8, 1.0, 1.2, 1.4)})
            print(sensitivity(df, f, fn, params, a.sensitivity, vals, a.bar, cfg, costs).to_string(index=False))
    else:
        per_day = 86_400_000 // BAR_MS[a.bar]
        g = grid(entry_n=[20, 40, 55, 80], exit_n=[10, 20, 30]) if a.strategy == "donchian" else grid(fast=[10, 20, 40], slow=[80, 120, 200])
        wf, summ = walk_forward(df, f, fn, g, a.bar, a.train_days * per_day, a.test_days * per_day, cfg, costs)
        print(wf.to_string(index=False)); print(fmt(summ))


if __name__ == "__main__":
    main()
