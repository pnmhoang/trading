"""Báo cáo Markdown kết quả tìm kiếm + test đóng băng."""
from __future__ import annotations

from pathlib import Path


def _fmt(s: dict) -> str:
    return (f"{s['n_trades']} lệnh, win {s['win_rate']:.0%}, kỳ vọng {s['expectancy_r']:+.2f}R, "
            f"PF {s['profit_factor']:.2f}, maxDD {s['max_drawdown']:.1%}")


def write_report(coin_results: dict, out_path: Path, portfolio: dict | None = None) -> None:
    lines = ["# Kết quả tìm chiến lược theo từng coin", ""]
    accepted_any = False
    for coin, r in coin_results.items():
        s, fz = r["search"], r.get("frozen")
        lines += [f"## {coin}", "",
                  f"Số tổ hợp đã thử: {s['n_trials']} · Reality-check p = {s['rc_pvalue']:.3f}", "",
                  "| Họ chiến lược | Số tổ hợp | OOS walk-forward | Kết quả |", "|---|---|---|---|"]
        for t in s["table"]:
            verdict = "ĐẠT bộ lọc" if t["passed"] else "Loại: " + "; ".join(t["reasons"])
            lines.append(f"| {t['family']} | {t['n_combos']} | {_fmt(t['oos_stats'])} | {verdict} |")
        lines.append("")
        if s["best"] is None:
            lines += ["Không có họ chiến lược nào qua bộ lọc trên dữ liệu dev; test đóng băng **không** được chạy.", ""]
        elif fz is None:
            lines += [f"Ứng viên tốt nhất: `{s['best']['family']}` {s['best']['params']} (chưa chạy test đóng băng).", ""]
        else:
            ok = fz["accepted"]
            accepted_any |= ok
            lines += [f"Test đóng băng `{fz['spec']['family']}`: {_fmt(fz['stats'])} → **{'CHẤP NHẬN' if ok else 'KHÔNG ĐẠT'}**."]
            if ok:
                lines.append(f"Đặc tả đóng băng: `{coin}_spec.json`.")
            lines.append("")
    if portfolio:
        lines += ["## Danh mục BTC+ETH (equity chung, rủi ro 1%/lệnh, halt lỗ ngày 3%)", "",
                  _fmt(portfolio["stats"]) + f"; lệnh bị halt chặn: {portfolio['skipped_by_halt']}", ""]
    if not accepted_any:
        lines += ["## Kết luận", "", "**Không có chiến lược nào đạt** tiêu chuẩn sau test đóng băng. "
                  "Đây là kết quả hợp lệ: không nên giao dịch tiền thật với các cấu hình đã thử.", ""]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
