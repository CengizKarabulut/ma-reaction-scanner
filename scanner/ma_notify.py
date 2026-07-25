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
    lines = [
        f"📊 <b>{label}</b>",
        f"Toplam: <b>{frame['symbol'].nunique()}</b> varlık · her varlık tek satır",
        "",
        "<pre>",
        f"{'Hisse':<7} {'TF':<4} {'MA':<8} {'Yön':<6} {'Uyum':<12} {'MedR':>6} {'Uzk':>6}",
        "-" * 58,
    ]
    for _, row in frame.head(max(1, top)).iterrows():
        lines.append(
            f"{str(row['symbol']):<7} {str(row.get('best_timeframe','-')):<4} "
            f"{str(row.get('best_ma','-')):<8} {str(row.get('best_side','-')):<6} "
            f"{str(row.get('best_compatibility','-'))[:12]:<12} "
            f"{_number(row.get('best_median_net_r')):>6} "
            f"{_number(row.get('best_distance_atr')):>6}"
        )
    lines.extend(["</pre>", "", "Tam rapor CSV eki ve GitHub artifact içindedir."])
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
