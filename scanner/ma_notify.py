#!/usr/bin/env python3
"""Send the deduplicated market summary and complete CSV to Telegram."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

import httpx
import pandas as pd

try:
    from .ma_watchlist import format_watchlist_text
    from .telegram_table_image import TableColumn, render_table_png
except ImportError:  # direct script execution
    from ma_watchlist import format_watchlist_text
    from telegram_table_image import TableColumn, render_table_png


def _number(value: object, digits: int = 2) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(number) else f"{float(number):.{digits}f}"


_TELEGRAM_TEXT_BUDGET = 4_000
_IMAGE_ROW_LIMIT = 30
_MIN_DISPLAY_TOUCHES = 5
_FOOTER = [
    "Skor = 0-100 gozlemsel seviye gucu; Uyum = eski trade-simulasyon skoru.",
    "Telegram tablolarinda varsayilan olarak 5+ temasli satirlar one cikarilir.",
    "Temas ham bagimsiz ziyaret sayisidir; 1-2 temas guclu seviye sayilmaz.",
    "Tam rapor CSV/HTML eki ve GitHub artifact icindedir.",
]


def _fits(lines: list[str], additions: list[str]) -> bool:
    return len("\n".join([*lines, *additions, *_FOOTER])) <= _TELEGRAM_TEXT_BUDGET


def _displayable_by_touches(
    frame: pd.DataFrame,
    *touch_columns: str,
    min_touches: int = _MIN_DISPLAY_TOUCHES,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    touches, found = _touch_values(frame, *touch_columns)
    if not found:
        return frame
    return frame[touches >= min_touches].copy()


def _touch_values(frame: pd.DataFrame, *touch_columns: str) -> tuple[pd.Series, bool]:
    touches = pd.Series(0.0, index=frame.index)
    found = False
    for column in touch_columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
            touches = touches.where(touches > 0, values)
            found = True
    return touches, found


def _display_frame_by_touch_strength(
    frame: pd.DataFrame,
    *touch_columns: str,
    min_touches: int = _MIN_DISPLAY_TOUCHES,
) -> tuple[pd.DataFrame, bool]:
    """Return only rows with enough repeated touches for user-facing reports.

    The full CSV/HTML artifacts still preserve every raw row. Telegram images and
    short text must not promote 1-4 touch rows as actionable MA DNA.
    """

    if frame is None or frame.empty:
        return frame, False
    touches, found = _touch_values(frame, *touch_columns)
    if not found:
        return frame, False
    strong = frame[touches >= min_touches].copy()
    return strong, strong.empty and len(frame) > 0


def format_summary(frame: pd.DataFrame, label: str, top: int = 25) -> str:
    display_frame, weak_fallback = _display_frame_by_touch_strength(
        frame, "best_level_touches", "best_touches"
    )
    selected = display_frame.head(max(1, top))
    safe_label = str(label)[:200]
    lines = [
        f"<b>{safe_label}</b>",
        f"Toplam: <b>{frame['symbol'].nunique()}</b> varlik - her varlik tek satir",
    ]
    if weak_fallback:
        lines.append(
            "5+ temasli guclu satir yok; 1-4 temasli ham adaylar sadece CSV/HTML ekinde tutuldu."
        )
    lines.extend([
        "",
        "<pre>",
        f"{'Varlik':<7} {'MA':<8} {'TF':<4} {'Tms':>3} {'Skor':>5} "
        f"{'Tut%':>5} {'Bnc':>5} {'Uzk%':>6} {'Uyum':>5} {'Dur':<4}",
        "-" * 68,
    ])
    has_exclusions = (
        "filter_status" in selected.columns
        and bool((selected["filter_status"] != "Uygun").any())
    )
    reason_reserve = ["Filtre disi nedenleri tam CSV'de.", ""] if has_exclusions else []
    displayed_indices: list[object] = []
    if selected.empty and weak_fallback:
        lines.append("5+ temasli satir yok; gosterilecek guvenilir aday bulunmadi.")
    for index, row in selected.iterrows():
        eligible = str(row.get("filter_status", "Uygun")) == "Uygun"
        touches = row.get("best_level_touches", row.get("best_touches"))
        score = row.get("best_level_score", row.get("best_compatibility_score"))
        table_row = (
            f"{str(row['symbol']):<7} {str(row.get('best_ma','-')):<8} "
            f"{str(row.get('best_timeframe','-')):<4} "
            f"{_number(touches, 0):>3} "
            f"{_number(score):>5} "
            f"{_number(row.get('best_hold_rate_pct', row.get('best_side_adherence_pct'))):>5} "
            f"{_number(row.get('best_median_bounce_atr')):>5} "
            f"{_number(row.get('best_distance_pct')):>6} "
            f"{_number(row.get('best_compatibility_score')):>5} "
            f"{'Ham' if weak_fallback else 'OK' if eligible else 'Disi':<4}"
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
        f"Seçilen sonuçlar: <b>{len(frame)}</b> analiz satırı",
        "En güçlü satırlar aşağıdadır; tam tablo CSV ve HTML ekindedir.",
        "",
        "<pre>",
        f"{'TF':<4} {'MA':<8} {'Taraf':<7} {'Tms':>3} {'Skor':>5} "
        f"{'Tut%':>5} {'Bnc':>5} {'Uzk%':>6} {'Uyum':>5}",
        "-" * 62,
    ]
    displayed = 0
    footer = [
        "</pre>",
        "",
        "Tut% (eski Kor%) = kirilmadan tutma; Tms = ham bagimsiz temas.",
        "Bnc = medyan sicrama ATR. Uyum = eski trade-simulasyon skoru.",
    ]
    display_frame, weak_fallback = _display_frame_by_touch_strength(frame, "Temas", "level_touches")
    if weak_fallback:
        lines[2] = (
            "5+ temasli guvenilir MA DNA satiri yok; 1-4 temasli ham satirlar "
            "yalnizca CSV/HTML ekinde tutuldu."
        )
    for _, row in display_frame.head(max(1, top)).iterrows():
        touches = row.get("Temas")
        score = row.get("Seviye Skoru", row.get("Uyum Skoru"))
        hold = row.get("Tutma %", row.get("Taraf Koruma %"))
        bounce = row.get("Sıçrama ATR", row.get("Sicrama ATR"))
        table_row = (
            f"{str(row.get('Zaman Dilimi', '-')):<4} "
            f"{str(row.get('MA', '-')):<8} "
            f"{str(row.get('Taraf', '-')):<7} "
            f"{_number(touches, 0):>3} "
            f"{_number(score):>5} "
            f"{_number(hold):>5} "
            f"{_number(bounce):>5} "
            f"{_number(row.get('Uzaklık %', row.get('Uzaklik %'))):>6} "
            f"{_number(row.get('Uyum Skoru')):>5}"
        )
        if len("\n".join([*lines, table_row, *footer])) > _TELEGRAM_TEXT_BUDGET:
            break
        lines.append(table_row)
        displayed += 1
    lines.extend(footer)
    if display_frame.empty and len(frame) > 0:
        lines.append(
            "5+ temasli satir yok; bu kosuda tekrarlayan guclu MA seviyesi bulunmadi."
        )
    elif weak_fallback:
        lines.append(
            "Bu tablo zayif/ham adaydir: 5+ temas yok, tek temasli satirlar saklandi."
        )
    elif displayed < min(len(display_frame), max(1, top)):
        lines.append("Diger 5+ temasli satirlar tam CSV/HTML tablosundadir.")
    return "\n".join(lines)


def _watchlist_detail_message(label: str, block: str, *, omitted: int = 0) -> str:
    lines = [
        f"<b>{str(label)[:200]}</b>",
        "Izleme seti: fiyatin once temas edebilecegi, yeterli gecmis temasi olan MA bolgeleri.",
        "Bu hedef fiyat veya al/sat sinyali degildir; siralama fiyata yakinliga goredir.",
        "",
        "<pre>",
        block,
        "</pre>",
    ]
    footer = "Tam detay CSV/HTML eki ve GitHub artifact icindedir."
    if omitted > 0:
        note = (
            f"Telegram limiti nedeniyle {omitted} bolge daha mesajda kisaltildi; "
            "tam liste ekte."
        )
        if len("\n".join([*lines, note, footer])) <= _TELEGRAM_TEXT_BUDGET:
            lines.append(note)
    lines.append(footer)
    return "\n".join(lines)


def _watchlist_block(
    records: list[dict[str, object]], max_ma_list_chars: int | None
) -> str:
    return html.escape(
        format_watchlist_text(
            pd.DataFrame.from_records(records),
            max_ma_list_chars=max_ma_list_chars,
        )
    )


def format_watchlist_detail(frame: pd.DataFrame, label: str, top: int = 20) -> str:
    selected = frame.head(max(1, top)) if top > 0 else frame
    if selected is None or selected.empty:
        block = html.escape(format_watchlist_text(pd.DataFrame()))
        return _watchlist_detail_message(label, block)

    best_records: list[dict[str, object]] = []
    best_limit: int | None = 180
    for ma_list_limit in (180, 120, 80, 40, 0):
        records: list[dict[str, object]] = []
        for _, row in selected.iterrows():
            candidate = [*records, row.to_dict()]
            omitted = len(selected) - len(candidate)
            block = _watchlist_block(candidate, ma_list_limit)
            message = _watchlist_detail_message(label, block, omitted=omitted)
            if len(message) > _TELEGRAM_TEXT_BUDGET:
                break
            if omitted > 0 and "Telegram limiti" not in message:
                break
            records = candidate
        if len(records) > len(best_records) or (
            len(records) == len(best_records)
            and records
            and ma_list_limit > (best_limit if best_limit is not None else -1)
        ):
            best_records = records
            best_limit = ma_list_limit

    if best_records:
        omitted = len(selected) - len(best_records)
        block = _watchlist_block(best_records, best_limit)
        message = _watchlist_detail_message(label, block, omitted=omitted)
        if len(message) <= _TELEGRAM_TEXT_BUDGET:
            return message

    fallback = html.escape(
        "Izleme seti uretildi; Telegram metin limiti nedeniyle tablo kisaltildi.\n"
        "Tam liste CSV/HTML eki ve GitHub artifact icindedir."
    )
    return _watchlist_detail_message(label, fallback, omitted=len(selected))



def _image_limit(top: int) -> int:
    return min(max(1, top), _IMAGE_ROW_LIMIT) if top > 0 else _IMAGE_ROW_LIMIT


def _safe_number(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(number) else float(number)


def _fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    number = _safe_number(value)
    return "-" if number is None else f"{number:.{digits}f}{suffix}"


def _fmt_price_like(value: object) -> str:
    number = _safe_number(value)
    if number is None:
        return "-"
    digits = 4 if abs(number) < 10 else 3 if abs(number) < 100 else 2
    return f"{number:.{digits}f}"


def _fmt_int(value: object) -> str:
    number = _safe_number(value)
    return "-" if number is None else str(int(round(number)))


def _zone_band(row: pd.Series) -> str:
    low = _safe_number(row.get("zone_low"))
    high = _safe_number(row.get("zone_high"))
    if low is None or high is None:
        return "-"
    if abs(high - low) < 0.005:
        return f"{low:.2f}"
    return f"{low:.2f}-{high:.2f}"


def _short_ma_list(value: object, max_items: int = 4) -> str:
    parts = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not parts:
        return "-"
    if len(parts) <= max_items:
        return ", ".join(parts)
    return ", ".join(parts[:max_items]) + f", +{len(parts) - max_items}"


def _first_value(row: pd.Series, *keys: str, default: object = "-") -> object:
    for key in keys:
        if key in row:
            return row.get(key)
    return default


_DETAIL_DISTANCE_HEADERS = (
    "Uzaklık %",
    "Uzaklik %",
    "Uzakl?k %",
    "Uzakl??k %",
)
_DETAIL_ROLE_HEADERS = (
    "Güncel Rol",
    "Guncel Rol",
    "G?ncel Rol",
    "G??ncel Rol",
)


_DETAIL_PRICE_HEADERS = (
    "Fiyat",
    "current_price",
    "price",
)
_DETAIL_LEVEL_HEADERS = (
    "MA Değeri",
    "MA Deđeri",
    "MA Degeri",
    "MA De?eri",
    "MA DeÄŸeri",
    "current_ma",
    "current_ma_value",
    "ma_value",
)


def _current_side_label(row: pd.Series) -> str:
    price = _safe_number(_first_value(row, *_DETAIL_PRICE_HEADERS, default=None))
    level = _safe_number(_first_value(row, *_DETAIL_LEVEL_HEADERS, default=None))
    if price is None or level is None:
        return str(row.get("Taraf", row.get("side", "-")))
    if abs(price - level) < 1e-12:
        return "Temas"
    return "Destek" if level < price else "Direnc"


def _watchlist_image_payload(frame: pd.DataFrame, label: str, top: int) -> tuple[bytes, str]:
    selected = frame.head(_image_limit(top)).copy()
    include_symbol = "symbol" in selected.columns and selected["symbol"].nunique(dropna=True) > 1
    rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        payload = {
            "timeframe": row.get("timeframe", "-"),
            "side": row.get("side", "-"),
            "price": _fmt_price_like(_first_value(row, "current_price", "Fiyat", default=None)),
            "zone": _zone_band(row),
            "distance": _fmt(row.get("distance_pct"), 1, "%"),
            "touches": _fmt_int(row.get("level_touches")),
            "hold": _fmt(row.get("hold_rate_pct"), 0, "%"),
            "zone_quality": row.get("zone_quality", row.get("confidence", "-")),
            "width": _fmt(row.get("zone_width_atr"), 2),
            "ma_list": _short_ma_list(row.get("ma_list")),
        }
        if include_symbol:
            payload["symbol"] = row.get("symbol", "-")
        rows.append(payload)

    columns = []
    if include_symbol:
        columns.append(TableColumn("symbol", "Varlik", 105))
    columns.extend([
        TableColumn("timeframe", "TF", 62, "center"),
        TableColumn("side", "Taraf", 88),
        TableColumn("price", "Fiyat", 96, "right"),
        TableColumn("zone", "Bolge", 205),
        TableColumn("distance", "Uzak", 78, "right"),
        TableColumn("touches", "Temas", 78, "right"),
        TableColumn("hold", "Tut", 70, "right"),
        TableColumn("zone_quality", "Kalite", 80),
        TableColumn("width", "Gen ATR", 82, "right"),
        TableColumn("ma_list", "Ortalamalar", 348 if include_symbol else 438),
    ])
    omitted = max(0, len(frame) - len(selected))
    badge = f"{len(selected)} satir" + (f" | +{omitted}" if omitted else "")
    subtitle = "Fiyata yakin izlenebilir MA bolgeleri - tam liste CSV/HTML ekte"
    return render_table_png(rows, columns, title=str(label)[:80], subtitle=subtitle, badge=badge), "watchlist"


def _single_image_selection(frame: pd.DataFrame, top: int) -> tuple[pd.DataFrame, bool]:
    selected, weak_fallback = _display_frame_by_touch_strength(frame, "Temas", "level_touches")
    return selected.head(_image_limit(top)).copy(), weak_fallback


def _single_rows_from_frame(
    selected: pd.DataFrame, *, weak_fallback: bool = False
) -> list[dict[str, object]]:
    rows = []
    for _, row in selected.iterrows():
        role = _first_value(row, *_DETAIL_ROLE_HEADERS)
        if weak_fallback:
            role = "Ham<5"
        rows.append({
            "timeframe": row.get("Zaman Dilimi", row.get("timeframe", "-")),
            "ma": row.get("MA", "-"),
            "price": _fmt_price_like(_first_value(row, *_DETAIL_PRICE_HEADERS, default=None)),
            "level": _fmt_price_like(_first_value(row, *_DETAIL_LEVEL_HEADERS, default=None)),
            "side": _current_side_label(row),
            "touches": _fmt_int(row.get("Temas", row.get("level_touches"))),
            "score": _fmt(row.get("Seviye Skoru", row.get("Uyum Skoru")), 1),
            "hold": _fmt(row.get("Tutma %", row.get("Taraf Koruma %")), 0, "%"),
            "distance": _fmt(
                _first_value(row, *_DETAIL_DISTANCE_HEADERS, default=None),
                1,
                "%",
            ),
            "role": role,
        })
    return rows


def _single_image_rows(frame: pd.DataFrame, top: int) -> list[dict[str, object]]:
    selected, weak_fallback = _single_image_selection(frame, top)
    return _single_rows_from_frame(selected, weak_fallback=weak_fallback)


def _single_image_payload(frame: pd.DataFrame, label: str, top: int) -> tuple[bytes, str]:
    selected, weak_fallback = _single_image_selection(frame, top)
    rows = _single_rows_from_frame(selected, weak_fallback=weak_fallback)
    columns = [
        TableColumn("timeframe", "TF", 54, "center"),
        TableColumn("ma", "MA", 130),
        TableColumn("price", "Fiyat", 96, "right"),
        TableColumn("level", "Seviye", 108, "right"),
        TableColumn("side", "Bugun", 92),
        TableColumn("touches", "Temas", 72, "right"),
        TableColumn("score", "Skor", 76, "right"),
        TableColumn("hold", "Tut", 68, "right"),
        TableColumn("distance", "Uzak", 76, "right"),
        TableColumn("role", "Rol", 118),
    ]
    omitted = max(0, len(frame) - len(rows))
    if weak_fallback and not rows:
        badge = f"Kanit yok | {len(frame)} ham"
    else:
        badge = f"{len(rows)} satir" + (f" | +{omitted}" if omitted else "")
    subtitle = (
        "5+ temas yok - guvenilir MA DNA bulunmadi; ham satirlar CSV/HTML ekte"
        if weak_fallback
        else "En guclu MA satirlari - izleme seti varsa once o gosterilir"
    )
    return render_table_png(rows, columns, title=str(label)[:80], subtitle=subtitle, badge=badge), "detail"


def _summary_image_payload(frame: pd.DataFrame, label: str, top: int) -> tuple[bytes, str]:
    selected, weak_fallback = _display_frame_by_touch_strength(
        frame, "best_level_touches", "best_touches"
    )
    selected = selected.head(_image_limit(top)).copy()
    rows = []
    for _, row in selected.iterrows():
        rows.append({
            "symbol": row.get("symbol", "-"),
            "ma": row.get("best_ma", "-"),
            "timeframe": row.get("best_timeframe", "-"),
            "touches": _fmt_int(row.get("best_level_touches", row.get("best_touches"))),
            "score": _fmt(row.get("best_level_score", row.get("best_compatibility_score")), 1),
            "hold": _fmt(row.get("best_hold_rate_pct", row.get("best_side_adherence_pct")), 0, "%"),
            "distance": _fmt(row.get("best_distance_pct"), 1, "%"),
            "status": "Ham<5" if weak_fallback else row.get("filter_status", "Uygun"),
        })
    columns = [
        TableColumn("symbol", "Varlik", 105),
        TableColumn("ma", "MA", 115),
        TableColumn("timeframe", "TF", 62, "center"),
        TableColumn("touches", "Temas", 82, "right"),
        TableColumn("score", "Skor", 82, "right"),
        TableColumn("hold", "Tut", 78, "right"),
        TableColumn("distance", "Uzak", 82, "right"),
        TableColumn("status", "Durum", 140),
    ]
    omitted = max(0, len(frame) - len(selected))
    if weak_fallback and not rows:
        badge = f"Kanit yok | {len(frame)} ham"
    else:
        badge = f"{len(selected)} satir" + (f" | +{omitted}" if omitted else "")
    subtitle = (
        "5+ temas yok - guvenilir MA DNA bulunmadi; ham satirlar CSV/HTML ekte"
        if weak_fallback
        else "Her varlik icin en iyi MA DNA satiri - tam liste CSV/HTML ekte"
    )
    return render_table_png(rows, columns, title=str(label)[:80], subtitle=subtitle, badge=badge), "summary"


def build_notification_image(
    frame: pd.DataFrame,
    label: str,
    top: int,
    *,
    detail_path: Path | None = None,
    watch_frame: pd.DataFrame | None = None,
) -> tuple[bytes, str] | None:
    if watch_frame is not None and not watch_frame.empty:
        return _watchlist_image_payload(watch_frame, label, top)
    if detail_path is not None and frame is not None and not frame.empty:
        return _single_image_payload(frame, label, top)
    if frame is not None and not frame.empty:
        return _summary_image_payload(frame, label, top)
    return None


def _photo_caption(label: str, kind: str) -> str:
    names = {
        "watchlist": "Izleme seti tablo gorseli",
        "detail": "Tek hisse MA tablo gorseli",
        "summary": "MA tarama ozet gorseli",
    }
    return (
        f"<b>{str(label)[:120]}</b>\n"
        f"{names.get(kind, 'MA tablo gorseli')} gonderildi. "
        "Tam CSV/HTML eklerde."
    )

def send(
    summary_path: Path,
    label: str,
    top: int,
    detail_path: Path | None = None,
    report_path: Path | None = None,
    watchlist_path: Path | None = None,
    image: bool = True,
) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    message_path = detail_path or summary_path
    frame = pd.read_csv(message_path)
    watch_frame = (
        pd.read_csv(watchlist_path)
        if watchlist_path is not None and watchlist_path.exists()
        else None
    )
    if detail_path is not None and watch_frame is not None and not watch_frame.empty:
        message = format_watchlist_detail(watch_frame, label, top)
    else:
        message = (
            format_single_detail(frame, label, top)
            if detail_path is not None
            else format_summary(frame, label, top)
        )
    image_payload = (
        build_notification_image(
            frame,
            label,
            top,
            detail_path=detail_path,
            watch_frame=watch_frame,
        )
        if image
        else None
    )
    base = f"https://api.telegram.org/bot{token}"
    with httpx.Client(timeout=60.0) as client:
        photo_sent = False
        if image_payload is not None:
            image_bytes, kind = image_payload
            response = client.post(
                f"{base}/sendPhoto",
                data={
                    "chat_id": chat_id,
                    "caption": _photo_caption(label, kind),
                    "parse_mode": "HTML",
                },
                files={"photo": ("ma_table.png", image_bytes, "image/png")},
            )
            try:
                response.raise_for_status()
                photo_sent = True
            except httpx.HTTPStatusError:
                photo_sent = False
        if not photo_sent:
            response = client.post(
                f"{base}/sendMessage",
                data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            )
            response.raise_for_status()
        attachments = [message_path]
        if watchlist_path is not None and watchlist_path.exists():
            attachments.append(watchlist_path)
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
    parser.add_argument("--watchlist")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Telegram tablo gorseli gondermeden eski metin modunu kullan",
    )
    args = parser.parse_args()
    send(
        Path(args.summary),
        args.label,
        args.top,
        detail_path=Path(args.detail) if args.detail else None,
        report_path=Path(args.report) if args.report else None,
        watchlist_path=Path(args.watchlist) if args.watchlist else None,
        image=not args.text_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
