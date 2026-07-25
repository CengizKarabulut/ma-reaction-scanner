#!/usr/bin/env python3
"""Send the deduplicated market summary and complete CSV to Telegram."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx
import pandas as pd


def _number(value: object, digits: int = 2) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(number) else f"{float(number):.{digits}f}"


def format_summary(frame: pd.DataFrame, label: str, top: int = 25) -> str:
    selected = frame.head(max(1, top))
    lines = [
        f"<b>{label}</b>",
        f"Toplam: <b>{frame['symbol'].nunique()}</b> varlik - her varlik tek satir",
        "",
        "<pre>",
        f"{'Hisse':<7} {'TF':<4} {'MA':<8} {'Fiyat':>8} {'MA Deg':>8} "
        f"{'Uzk%':>6} {'ATR':>6} {'Durum':<5}",
        "-" * 71,
    ]
    for _, row in selected.iterrows():
        eligible = str(row.get("filter_status", "Uygun")) == "Uygun"
        lines.append(
            f"{str(row['symbol']):<7} {str(row.get('best_timeframe','-')):<4} "
            f"{str(row.get('best_ma','-')):<8} "
            f"{_number(row.get('current_price')):>8} "
            f"{_number(row.get('best_ma_value')):>8} "
            f"{_number(row.get('best_distance_pct')):>6} "
            f"{_number(row.get('best_distance_atr')):>6} "
            f"{'Uygun' if eligible else 'Disi':<5}"
        )
    lines.extend(["</pre>", ""])
    if "filter_status" in selected.columns:
        excluded = selected[selected["filter_status"] != "Uygun"]
    else:
        excluded = selected.iloc[0:0]
    if not excluded.empty:
        lines.append("<b>Filtre disi nedenleri</b>")
        for _, row in excluded.iterrows():
            reasons = str(row.get("filter_reasons", "")).strip() or "Esik disi"
            lines.append(f"{row['symbol']}: {reasons}")
        lines.append("")
    lines.extend(
        [
            "Uzk% = (MA - fiyat) / fiyat. ATR = volatiliteye gore uzaklik.",
            "Fiyat ve MA Deg ayni zaman dilimi ve veri anina aittir.",
            "Tam rapor CSV eki ve GitHub artifact icindedir.",
        ]
    )
    return "\n".join(lines)


def send(summary_path: Path, label: str, top: int) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    frame = pd.read_csv(summary_path)
    base = f"https://api.telegram.org/bot{token}"
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{base}/sendMessage",
            data={"chat_id": chat_id, "text": format_summary(frame, label, top), "parse_mode": "HTML"},
        )
        response.raise_for_status()
        with summary_path.open("rb") as handle:
            response = client.post(
                f"{base}/sendDocument",
                data={"chat_id": chat_id, "caption": f"{label} — tam tekilleştirilmiş rapor"},
                files={"document": (summary_path.name, handle, "text/csv")},
            )
        response.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--label", default="MA Trend ve Tepki")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    send(Path(args.summary), args.label, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
