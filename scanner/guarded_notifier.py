#!/usr/bin/env python3
"""Telegram reporting for one-instrument-one-row guarded MA summaries."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

try:
    from .notifier import render_table_image, send_photo, send_telegram
except ImportError:  # direct script execution
    from notifier import render_table_image, send_photo, send_telegram


_REQUIRED_COLUMNS = {
    "symbol",
    "asset_label",
    "current_price",
    "tested_level_count",
    "certified_level_count",
    "actionable_level_count",
    "certification_rate_pct",
    "avg_holdout_hit_rate_pct",
    "avg_holdout_return_atr",
}


def select_top_instruments(summary: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Rank unique instruments; prefer certified rows and never duplicate symbols."""

    missing = sorted(_REQUIRED_COLUMNS - set(summary.columns))
    if missing:
        raise ValueError(f"guarded summary columns missing: {missing}")
    ranked = summary.copy()
    ranked["symbol"] = ranked["symbol"].astype(str).str.upper()
    numeric = [
        "certified_level_count",
        "actionable_level_count",
        "certification_rate_pct",
        "avg_holdout_hit_rate_pct",
        "avg_holdout_return_atr",
    ]
    for column in numeric:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked.sort_values(
        numeric + ["symbol"],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    )
    ranked = ranked.drop_duplicates(["asset_label", "symbol"], keep="first")
    certified = ranked[ranked["certified_level_count"] > 0]
    chosen = certified if not certified.empty else ranked
    return chosen if top_n == 0 else chosen.head(top_n)


def _number(value: object, decimals: int = 1, suffix: str = "") -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "—"
    return f"{number:.{decimals}f}{suffix}"


def _price(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "—"
    if abs(number) < 10:
        return f"{number:.4f}"
    if abs(number) < 100:
        return f"{number:.3f}"
    return f"{number:.2f}"


def build_guarded_table(summary: pd.DataFrame, top_n: int = 20) -> tuple[list[str], list[list[str]], dict[int, list[str | None]], str]:
    """Build the compact Telegram table from already-deduplicated summary data."""

    top = select_top_instruments(summary, top_n=top_n)
    rows: list[list[str]] = []
    colors: dict[int, list[str | None]] = {0: []}
    for _, row in top.iterrows():
        tested = int(row["tested_level_count"])
        certified = int(row["certified_level_count"])
        actionable = int(row["actionable_level_count"])
        rows.append(
            [
                str(row["symbol"]),
                str(row["asset_label"]),
                _price(row["current_price"]),
                f"{certified}/{tested}",
                _number(row["certification_rate_pct"], 1, "%"),
                str(actionable),
                _number(row["avg_holdout_hit_rate_pct"], 1, "%"),
                _number(row["avg_holdout_return_atr"], 2),
            ]
        )
        colors[0].append("#7fc97f" if certified > 0 else None)
    has_certified = bool((summary["certified_level_count"] > 0).any())
    title = (
        f"Sertifikalı Top {len(rows)} — Her Varlık Tek Satır"
        if has_certified
        else f"Sertifikalı Seviye Yok — En Yakın {len(rows)} Aday"
    )
    headers = ["Varlık", "Tür", "Fiyat", "Sert./Test", "Oran", "Hazır", "Holdout WR", "Holdout ATR"]
    return headers, rows, colors, title


def send_guarded_summary(
    token: str,
    chat_id: str,
    summary: pd.DataFrame,
    *,
    label: str,
    top_n: int = 20,
) -> bool:
    unique_count = len(summary.drop_duplicates(["asset_label", "symbol"]))
    certified_count = int((summary["certified_level_count"] > 0).sum())
    actionable_count = int((summary["actionable_level_count"] > 0).sum())
    header = (
        f"📊 *{label}*\n\n"
        f"Benzersiz varlık: *{unique_count}*\n"
        f"Sertifikalı varlık: *{certified_count}*\n"
        f"Yakın/aksiyon alınabilir: *{actionable_count}*\n\n"
        "Her varlık tabloda yalnızca bir kez gösterilir."
    )
    ok = send_telegram(token, chat_id, header, parse_mode="Markdown")
    headers, rows, colors, title = build_guarded_table(summary, top_n=top_n)
    if not rows:
        return ok
    images = render_table_image(headers, rows, title=title, col_colors=colors)
    if images:
        for index, image in enumerate(images, 1):
            caption = title if len(images) == 1 else f"{title} — Sayfa {index}/{len(images)}"
            ok = send_photo(token, chat_id, image, caption=caption) and ok
        return ok
    lines = [title, "```", "Varlık   Tür       Sert/Test  Oran   H-WR   H-ATR"]
    for row in rows:
        lines.append(f"{row[0]:<8} {row[1]:<9} {row[3]:<10} {row[4]:<6} {row[6]:<6} {row[7]}")
    lines.append("```")
    return send_telegram(token, chat_id, "\n".join(lines), parse_mode="Markdown") and ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="instrument_summary.csv path")
    parser.add_argument("--label", default="Guarded MA Research")
    parser.add_argument("--top", type=int, default=20, help="0 shows all certified instruments")
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID secrets are required", file=sys.stderr)
        return 2
    if not os.path.exists(args.summary):
        print(f"summary not found: {args.summary}", file=sys.stderr)
        return 2
    summary = pd.read_csv(args.summary)
    if summary.empty:
        send_telegram(token, chat_id, f"⚠️ {args.label}: özet boş", parse_mode=None)
        return 1
    return 0 if send_guarded_summary(token, chat_id, summary, label=args.label, top_n=args.top) else 1


if __name__ == "__main__":
    raise SystemExit(main())
