#!/usr/bin/env python3
"""Telegram reporting for one-instrument-one-row guarded MA summaries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
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
    "low_confidence_level_count",
    "actionable_level_count",
    "certification_rate_pct",
    "max_sr_strength_score",
    "avg_holdout_hit_rate_pct",
    "avg_holdout_return_atr",
    "avg_holdout_net_return_atr",
    "nearest_timeframe",
    "nearest_side",
    "nearest_ma",
    "nearest_period",
    "nearest_level",
    "nearest_distance_pct",
    "nearest_abs_distance_atr",
    "nearest_status",
    "nearest_discovery_events",
    "nearest_sr_strength_score",
}

_STATUS_LABELS = {
    "insufficient_history": "Veri/olay yetersiz",
    "unverified_candidate": "Kanıt kapıları geçilmedi",
    "validation_failed": "Validation geçmedi",
    "holdout_failed": "Holdout geçmedi",
    "low_confidence_fast": "Dusuk guven: zayif kanit profili",
    "certified_thin_holdout": "Dusuk guven: az holdout",
    "certified_but_far": "Sertifikalı ama uzak",
    "certified": "Sertifikalı",
    "none": "Aday yok",
}
_SIDE_LABELS = {"support": "Destek", "resistance": "Direnç"}
_MIN_DISPLAY_TOUCHES = 10


def select_top_instruments(
    summary: pd.DataFrame, top_n: int = 20, min_touches: int = _MIN_DISPLAY_TOUCHES
) -> pd.DataFrame:
    """Rank unique instruments; uncertified fallback is ordered by proximity."""

    missing = sorted(_REQUIRED_COLUMNS - set(summary.columns))
    if missing:
        raise ValueError(f"guarded summary columns missing: {missing}")
    min_touches = max(1, int(min_touches))
    ranked = summary.copy()
    ranked["symbol"] = ranked["symbol"].astype(str).str.upper()
    descending = [
        "certified_level_count",
        "actionable_level_count",
        "max_sr_strength_score",
        "certification_rate_pct",
        "avg_holdout_hit_rate_pct",
        "avg_holdout_net_return_atr",
    ]
    if "nearest_total_touch_events" not in ranked:
        ranked["nearest_total_touch_events"] = ranked["nearest_discovery_events"]
    for column in [*descending, "nearest_abs_distance_atr", "nearest_total_touch_events"]:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked.drop_duplicates(["asset_label", "symbol"], keep="first")
    certified = ranked[ranked["certified_level_count"] > 0]
    if not certified.empty:
        chosen = certified.sort_values(
            descending + ["nearest_abs_distance_atr", "symbol"],
            ascending=[False, False, False, False, False, False, True, True],
            na_position="last",
        )
    else:
        chosen = ranked[
            (ranked["nearest_total_touch_events"].fillna(0) >= min_touches)
            & (ranked["nearest_status"].astype(str) != "none")
        ].sort_values(
            ["nearest_abs_distance_atr", "nearest_total_touch_events", "max_sr_strength_score", "symbol"],
            ascending=[True, False, False, True],
            na_position="last",
        )
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


def _period(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "—" if pd.isna(number) else str(int(number))


def _candidate_rows(top: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for _, row in top.iterrows():
        candidate = f"{row['nearest_timeframe']}:{row['nearest_ma']}{_period(row['nearest_period'])}"
        rows.append(
            [
                str(row["symbol"]),
                _price(row["current_price"]),
                candidate,
                _SIDE_LABELS.get(str(row["nearest_side"]), str(row["nearest_side"])),
                _price(row["nearest_level"]),
                _number(row["nearest_abs_distance_atr"], 2),
                _number(abs(float(row["nearest_distance_pct"])), 2, "%"),
                _number(row["nearest_sr_strength_score"], 1),
                str(int(row.get("nearest_total_touch_events", row["nearest_discovery_events"]))),
                _STATUS_LABELS.get(
                    str(row["nearest_status"]), str(row["nearest_status"])
                ),
            ]
        )
    return rows


def build_guarded_table(
    summary: pd.DataFrame, top_n: int = 20, min_touches: int = _MIN_DISPLAY_TOUCHES
) -> tuple[list[str], list[list[str]], dict[int, list[str | None]], str]:
    """Build a compact certified table or a useful nearest-candidate fallback."""

    top = select_top_instruments(summary, top_n=top_n, min_touches=min_touches)
    has_certified = bool((summary["certified_level_count"] > 0).any())
    colors: dict[int, list[str | None]] = {0: []}
    if not has_certified:
        rows = _candidate_rows(top)
        colors[0] = [None] * len(rows)
        headers = [
            "Varlık",
            "Fiyat",
            "En Yakın Aday",
            "Taraf",
            "Seviye",
            "Uzak ATR",
            "Uzak %",
            "SR Guc",
            "Olay",
            "Durum",
        ]
        title = f"Sertifikalı Seviye Yok — En Yakın {len(rows)} Güncel Aday"
        return headers, rows, colors, title

    rows: list[list[str]] = []
    for _, row in top.iterrows():
        tested = int(row["tested_level_count"])
        certified = int(row["certified_level_count"])
        low_confidence = int(row["low_confidence_level_count"])
        actionable = int(row["actionable_level_count"])
        rows.append(
            [
                str(row["symbol"]),
                str(row["asset_label"]),
                _price(row["current_price"]),
                f"{certified}+{low_confidence}/{tested}",
                _number(row["max_sr_strength_score"], 1),
                _number(row["certification_rate_pct"], 1, "%"),
                str(actionable),
                _number(row["avg_holdout_hit_rate_pct"], 1, "%"),
                _number(row["avg_holdout_return_atr"], 2),
                _number(row["avg_holdout_net_return_atr"], 2),
            ]
        )
        colors[0].append("#7fc97f")
    headers = [
        "Varlik",
        "Tur",
        "Fiyat",
        "Sert.+DG/Test",
        "SR Guc",
        "Oran",
        "Hazir",
        "Holdout WR",
        "Holdout ATR",
        "Net ATR",
    ]
    title = f"Sertifikalı Top {len(rows)} — Her Varlık Tek Satır"
    return headers, rows, colors, title


def _behavior_rows(
    table: pd.DataFrame, max_rows: int, score_column: str = "reaction_quality_score"
) -> list[list[str]]:
    rows: list[list[str]] = []
    for _, row in table.head(max_rows).iterrows():
        distance_pct = pd.to_numeric(
            pd.Series([row.get("distance_pct", 0.0)]), errors="coerce"
        ).fillna(0.0).iloc[0]
        touch_events = pd.to_numeric(
            pd.Series([row.get("total_touch_events", 0)]), errors="coerce"
        ).fillna(0).iloc[0]
        rows.append(
            [
                str(row.get("symbol", "")),
                _price(row.get("current_price")),
                _SIDE_LABELS.get(str(row.get("side", "")), str(row.get("side", ""))),
                str(row.get("ma_label", "")),
                _price(row.get("current_ma")),
                _number(row.get("abs_distance_atr"), 2),
                _number(abs(float(distance_pct)), 2, "%"),
                str(int(touch_events)),
                _number(row.get("reaction_hit_rate_pct"), 1, "%"),
                _number(row.get("reaction_median_fixed_atr"), 2),
                _number(row.get(score_column), 1),
            ]
        )
    return rows


def send_behavior_tables(
    token: str,
    chat_id: str,
    behavior_dir: str | Path,
    *,
    top_n: int = 20,
    min_touches: int = _MIN_DISPLAY_TOUCHES,
    notify_empty: bool = False,
) -> bool:
    min_touches = max(1, int(min_touches))
    folder = Path(behavior_dir)
    if not folder.exists():
        if notify_empty:
            return send_telegram(
                token,
                chat_id,
                "Davranış tabloları bulunamadı: çıktı klasöründe ma_behavior_*.csv yok.",
                parse_mode=None,
            )
        return True
    specs = [
        (
            "ma_behavior_most_visited.csv",
            "En Cok Temas Alan Ortalamalar",
            "reaction_quality_score",
        ),
        (
            "ma_behavior_best_reactions.csv",
            "En Iyi Tepki Veren Ortalamalar",
            "reaction_quality_score",
        ),
        (
            "ma_behavior_near_price.csv",
            "Fiyata Yakin Guclu Temasli Destek/Direnc",
            "near_action_score",
        ),
    ]
    headers = [
        "Varlik",
        "Fiyat",
        "Taraf",
        "MA",
        "Seviye",
        "Uzak ATR",
        "Uzak %",
        "Temas",
        "Tepki%",
        "MedATR",
        "Skor",
    ]
    ok = True
    sent_any = False
    max_rows = max(int(top_n), 10)
    for filename, title, score_column in specs:
        path = folder / filename
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if table.empty:
            continue
        touch_events = pd.to_numeric(
            table.get("total_touch_events", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        table = table[touch_events >= min_touches].copy()
        if table.empty:
            continue
        rows = _behavior_rows(table, max_rows=max_rows, score_column=score_column)
        if not rows:
            continue
        sent_any = True
        images = render_table_image(headers, rows, title=title)
        if images:
            for index, image in enumerate(images, 1):
                caption = title if len(images) == 1 else f"{title} - Sayfa {index}/{len(images)}"
                ok = send_photo(token, chat_id, image, caption=caption) and ok
        else:
            lines = [title, "```", " | ".join(headers), "-" * 60]
            lines.extend(" | ".join(row) for row in rows)
            lines.append("```")
            ok = send_telegram(token, chat_id, "\n".join(lines), parse_mode="Markdown") and ok
    if notify_empty and not sent_any:
        ok = send_telegram(
            token,
            chat_id,
            f"Davranış tablosu gönderilmedi: minimum {min_touches} ham temas eşiğini "
            "geçen MA bulunamadı. Workflow'da behavior_min_touches değerini "
            "düşürürsen Telegram da aynı eşiği kullanır.",
            parse_mode=None,
        ) and ok
    return ok


def send_guarded_summary(
    token: str,
    chat_id: str,
    summary: pd.DataFrame,
    *,
    label: str,
    top_n: int = 20,
    behavior_dir: str | Path | None = None,
    behavior_min_touches: int = _MIN_DISPLAY_TOUCHES,
) -> bool:
    behavior_min_touches = max(1, int(behavior_min_touches))
    unique_count = len(summary.drop_duplicates(["asset_label", "symbol"]))
    tested_levels = int(summary["tested_level_count"].sum())
    discovery_levels = int(summary.get("discovery_pass_count", pd.Series(dtype=float)).sum())
    certified_levels = int(summary["certified_level_count"].sum())
    low_confidence_levels = int(summary.get("low_confidence_level_count", pd.Series(dtype=float)).sum())
    certified_instruments = int((summary["certified_level_count"] > 0).sum())
    actionable_count = int((summary["actionable_level_count"] > 0).sum())
    explanation = (
        "Veri bulundu; ancak tüm kanıt kapılarını geçen seviye yok. "
        "Yakın aday ve MA davranış tabloları mevcutsa aşağıda gösterilir."
        if certified_levels == 0
        else "Her varlık tabloda yalnızca bir kez gösterilir."
    )
    header = (
        f"📊 *{label}*\n\n"
        f"Benzersiz varlık: *{unique_count}*\n"
        f"Test edilen aktif seviye: *{tested_levels}*\n"
        f"Discovery geçen seviye: *{discovery_levels}*\n"
        f"Sertifikali seviye/varlik: *{certified_levels}/{certified_instruments}*\n"
        f"Dusuk-guven seviye: *{low_confidence_levels}*\n"
        f"Yakin/aksiyon alinabilir varlik: *{actionable_count}*\n\n"
        f"_{explanation}_"
    )
    ok = send_telegram(token, chat_id, header, parse_mode="Markdown")
    headers, rows, colors, title = build_guarded_table(
        summary, top_n=top_n, min_touches=behavior_min_touches
    )
    if not rows:
        if behavior_dir is not None:
            ok = send_behavior_tables(
                token,
                chat_id,
                behavior_dir,
                top_n=top_n,
                min_touches=behavior_min_touches,
                notify_empty=True,
            ) and ok
        return ok
    images = render_table_image(headers, rows, title=title, col_colors=colors)
    if images:
        for index, image in enumerate(images, 1):
            caption = (
                title
                if len(images) == 1
                else f"{title} — Sayfa {index}/{len(images)}"
            )
            ok = send_photo(token, chat_id, image, caption=caption) and ok
        if behavior_dir is not None:
            ok = send_behavior_tables(
                token,
                chat_id,
                behavior_dir,
                top_n=top_n,
                min_touches=behavior_min_touches,
            ) and ok
        return ok
    lines = [title, "```", " | ".join(headers), "-" * 60]
    lines.extend(" | ".join(row) for row in rows)
    lines.append("```")
    ok = send_telegram(token, chat_id, "\n".join(lines), parse_mode="Markdown") and ok
    if behavior_dir is not None:
        ok = send_behavior_tables(
            token,
            chat_id,
            behavior_dir,
            top_n=top_n,
            min_touches=behavior_min_touches,
        ) and ok
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="instrument_summary.csv path")
    parser.add_argument("--label", default="Guarded MA Research")
    parser.add_argument("--top", type=int, default=20, help="0 shows every ranked instrument")
    parser.add_argument(
        "--behavior-dir",
        default=None,
        help="Directory containing ma_behavior_*.csv files; defaults to summary directory",
    )
    parser.add_argument(
        "--behavior-min-touches",
        type=int,
        default=_MIN_DISPLAY_TOUCHES,
        help="Minimum raw MA touches required for Telegram behavior tables",
    )
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID secrets are required",
            file=sys.stderr,
        )
        return 2
    if not os.path.exists(args.summary):
        print(f"summary not found: {args.summary}", file=sys.stderr)
        return 2
    summary = pd.read_csv(args.summary)
    behavior_dir = args.behavior_dir or str(Path(args.summary).parent)
    if summary.empty:
        send_telegram(token, chat_id, f"⚠️ {args.label}: özet boş", parse_mode=None)
        return 1
    return (
        0
        if send_guarded_summary(
            token,
            chat_id,
            summary,
            label=args.label,
            top_n=args.top,
            behavior_dir=behavior_dir,
            behavior_min_touches=args.behavior_min_touches,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
