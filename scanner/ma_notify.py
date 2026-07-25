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


_TELEGRAM_TEXT_BUDGET = 4_000
_FOOTER = [
    "Uzk% = (MA - fiyat) / fiyat. ATR = volatiliteye gore uzaklik.",
    "Fiyat ve MA Deg ayni zaman dilimi ve veri anina aittir.",
    "Tam rapor CSV eki ve GitHub artifact icindedir.",
]


def _fits(lines: list[str], additions: list[str]) -> bool:
    return len("\n".join([*lines, *additions, *_FOOTER])) <= _TELEGRAM_TEXT_BUDGET


def format_summary(frame: pd.DataFrame, label: str, top: int = 25) -> str:
    selected = frame.head(max(1, top))
    safe_label = str(label)[:200]
    lines = [
        f"<b>{safe_label}</b>",
        f"Toplam: <b>{frame['symbol'].nunique()}</b> varlik - her varlik tek satir",
        "",
        "<pre>",
        f"{'Hisse':<7} {'TF':<4} {'MA':<8} {'Fiyat':>8} {'MA Deg':>8} "
        f"{'Uzk%':>6} {'ATR':>6} {'Durum':<5}",
        "-" * 71,
    ]
    has_exclusions = (
        "filter_status" in selected.columns
        and bool((selected["filter_status"] != "Uygun").any())
    )
    reason_reserve = ["Filtre disi nedenleri tam CSV'de.", ""] if has_exclusions else []
    displayed_indices: list[object] = []
    for index, row in selected.iterrows():
        eligible = str(row.get("filter_status", "Uygun")) == "Uygun"
        table_row = (
            f"{str(row['symbol']):<7} {str(row.get('best_timeframe','-')):<4} "
            f"{str(row.get('best_ma','-')):<8} "
            f"{_number(row.get('current_price')):>8} "
            f"{_number(row.get('best_ma_value')):>8} "
            f"{_number(row.get('best_distance_pct')):>6} "
            f"{_number(row.get('best_distance_atr')):>6} "
            f"{'Uygun' if eligible else 'Disi':<5}"
        )
        if not _fits(lines, [table_row, "</pre>", "", *reason_reserve]):
            break
        lines.append(table_row)
        displayed_indices.append(index)
    lines.extend(["</pre>", ""])
    if len(displayed_indices) < len(selected):
        note = "Diger tablo satirlari tam CSV'de."
        if _fits(lines, [note, ""]):
            lines.extend([note, ""])

    displayed = selected.loc[displayed_indices]
    if "filter_status" in displayed.columns:
        excluded = displayed[displayed["filter_status"] != "Uygun"]
    else:
        excluded = displayed.iloc[0:0]
    if not excluded.empty and _fits(lines, ["<b>Filtre disi nedenleri</b>"]):
        lines.append("<b>Filtre disi nedenleri</b>")
        omitted = False
        for _, row in excluded.iterrows():
            reasons = str(row.get("filter_reasons", "")).strip() or "Esik disi"
            detail = f"{row['symbol']}: {reasons}"
            if not _fits(lines, [detail, "Diger filtre nedenleri tam CSV'de.", ""]):
                omitted = True
                break
            lines.append(detail)
        if omitted:
            note = "Diger filtre nedenleri tam CSV'de."
            if _fits(lines, [note, ""]):
                lines.append(note)
        lines.append("")
    elif has_exclusions and _fits(lines, reason_reserve):
        lines.extend(reason_reserve)
    lines.extend(_FOOTER)
    return "\n".join(lines)


def format_single_detail(frame: pd.DataFrame, label: str, top: int = 20) -> str:
    lines = [
        f"<b>{str(label)[:200]}</b>",
        f"Seçilen kombinasyonlar: <b>{len(frame)}</b> destek/direnç satırı",
        "En güçlü satırlar aşağıdadır; tam tablo CSV ve HTML ekindedir.",
        "",
        "<pre>",
        f"{'TF':<4} {'MA':<8} {'Taraf':<7} {'Tms':>3} {'Kor%':>5} "
        f"{'Kaz%':>5} {'MedR':>5} {'Edge':>5}",
        "-" * 53,
    ]
    displayed = 0
    footer = [
        "</pre>",
        "",
        "Kor% = MA'nın beklenen tarafında geçirilen süre.",
        "Tms = bağımsız temas. MedR/Edge = R cinsinden tepki kalitesi.",
    ]
    for _, row in frame.head(max(1, top)).iterrows():
        table_row = (
            f"{str(row.get('Zaman Dilimi', '-')):<4} "
            f"{str(row.get('MA', '-')):<8} "
            f"{str(row.get('Taraf', '-')):<7} "
            f"{_number(row.get('Temas'), 0):>3} "
            f"{_number(row.get('Taraf Koruma %')):>5} "
            f"{_number(row.get('Kazanma %')):>5} "
            f"{_number(row.get('Medyan R')):>5} "
            f"{_number(row.get('Edge R')):>5}"
        )
        if len("\n".join([*lines, table_row, *footer])) > _TELEGRAM_TEXT_BUDGET:
            break
        lines.append(table_row)
        displayed += 1
    lines.extend(footer)
    if displayed < min(len(frame), max(1, top)):
        lines.append("Diğer satırlar tam CSV/HTML tablosundadır.")
    return "\n".join(lines)


def send(
    summary_path: Path,
    label: str,
    top: int,
    detail_path: Path | None = None,
    report_path: Path | None = None,
) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    message_path = detail_path or summary_path
    frame = pd.read_csv(message_path)
    message = (
        format_single_detail(frame, label, top)
        if detail_path is not None
        else format_summary(frame, label, top)
    )
    base = f"https://api.telegram.org/bot{token}"
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{base}/sendMessage",
            data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
        )
        response.raise_for_status()
        attachments = [message_path]
        if report_path is not None and report_path.exists():
            attachments.append(report_path)
        for attachment in attachments:
            with attachment.open("rb") as handle:
                response = client.post(
                    f"{base}/sendDocument",
                    data={"chat_id": chat_id, "caption": f"{label} — tam tablo"},
                    files={"document": (attachment.name, handle)},
                )
            response.raise_for_status()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--label", default="MA Trend ve Tepki")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--detail")
    parser.add_argument("--report")
    args = parser.parse_args()
    send(
        Path(args.summary),
        args.label,
        args.top,
        detail_path=Path(args.detail) if args.detail else None,
        report_path=Path(args.report) if args.report else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
