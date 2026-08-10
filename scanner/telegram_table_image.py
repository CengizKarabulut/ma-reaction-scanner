"""Render compact Telegram-friendly table images."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class TableColumn:
    key: str
    title: str
    width: int
    align: str = "left"


_BG = (235, 241, 250)
_CARD = (255, 255, 255)
_HEAD = (224, 236, 251)
_GRID = (214, 222, 232)
_ROW_ALT = (247, 249, 252)
_TEXT = (22, 34, 51)
_MUTED = (96, 108, 126)
_BLUE = (28, 91, 204)
_GREEN = (20, 132, 72)
_RED = (178, 42, 36)
_AMBER = (158, 108, 0)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    paths = [
        Path("/usr/share/fonts/truetype/dejavu") / names[0],
        Path("/usr/share/fonts/truetype/liberation2") / ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"),
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
    ]
    for path in paths:
        try:
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return int(right - left)


def _shorten(
    draw: ImageDraw.ImageDraw, text: object, font: ImageFont.ImageFont, width: int
) -> str:
    value = "" if text is None else str(text)
    if _text_width(draw, value, font) <= width:
        return value
    if width <= _text_width(draw, "...", font):
        return "..."
    trimmed = value
    while trimmed and _text_width(draw, trimmed + "...", font) > width:
        trimmed = trimmed[:-1]
    return trimmed.rstrip() + "..."


def _color_for(key: str, value: object) -> tuple[int, int, int]:
    text = str(value).lower()
    if key in {"side", "taraf"}:
        if "destek" in text:
            return _GREEN
        if "diren" in text:
            return _RED
    if key in {"confidence", "guven", "class", "zone_quality"}:
        if "guclu" in text or "g??" in text:
            return _GREEN
        if "zay" in text:
            return _RED
        if "orta" in text or "izleme" in text:
            return _AMBER
    return _TEXT


def render_table_png(
    rows: Iterable[Mapping[str, object]],
    columns: list[TableColumn],
    *,
    title: str,
    subtitle: str = "",
    badge: str = "",
    footer: str = "Tam liste CSV/HTML artifact icinde.",
) -> bytes:
    """Return a clean PNG table suitable for Telegram ``sendPhoto``."""

    rows = list(rows)
    title_font = _font(40, bold=True)
    subtitle_font = _font(22)
    badge_font = _font(22, bold=True)
    header_font = _font(19, bold=True)
    body_font = _font(19, bold=True)
    small_font = _font(18)

    margin = 34
    card_pad_x = 38
    card_pad_y = 30
    row_h = 42
    header_h = 38
    table_gap = 24
    footer_gap = 28
    table_w = sum(col.width for col in columns)
    width = max(920, table_w + margin * 2 + card_pad_x * 2)
    title_h = 54
    subtitle_h = 32 if subtitle else 0
    table_h = header_h + max(1, len(rows)) * row_h
    height = (
        margin * 2
        + card_pad_y * 2
        + title_h
        + subtitle_h
        + table_gap
        + table_h
        + footer_gap
        + 30
    )

    image = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(image)
    card = (margin, margin, width - margin, height - margin)
    draw.rounded_rectangle(card, radius=28, fill=_CARD)

    x = margin + card_pad_x
    y = margin + card_pad_y
    draw.text((x, y), title, fill=_TEXT, font=title_font)
    if badge:
        badge_pad_x = 20
        badge_w = _text_width(draw, badge, badge_font) + badge_pad_x * 2
        badge_h = 42
        bx = width - margin - card_pad_x - badge_w
        by = y + 4
        draw.rounded_rectangle((bx, by, bx + badge_w, by + badge_h), radius=18, fill=_HEAD)
        draw.text((bx + badge_pad_x, by + 8), badge, fill=_BLUE, font=badge_font)
    y += title_h
    if subtitle:
        draw.text((x, y), subtitle, fill=_MUTED, font=subtitle_font)
        y += subtitle_h
    y += table_gap

    table_x = x
    draw.rounded_rectangle((table_x, y, table_x + table_w, y + header_h), radius=8, fill=_HEAD)
    col_x = table_x
    for col in columns:
        draw.text((col_x + 10, y + 8), col.title, fill=_TEXT, font=header_font)
        col_x += col.width

    y += header_h
    if not rows:
        draw.rectangle((table_x, y, table_x + table_w, y + row_h), fill=_ROW_ALT)
        draw.text((table_x + 14, y + 11), "Gosterilecek satir yok", fill=_MUTED, font=body_font)
        y += row_h
    for index, row in enumerate(rows):
        fill = _CARD if index % 2 == 0 else _ROW_ALT
        draw.rectangle((table_x, y, table_x + table_w, y + row_h), fill=fill)
        draw.line((table_x, y, table_x + table_w, y), fill=_GRID, width=1)
        col_x = table_x
        for col in columns:
            raw = row.get(col.key, "")
            value = _shorten(draw, raw, body_font, col.width - 20)
            color = _color_for(col.key, raw)
            tx = col_x + 10
            if col.align == "right":
                tx = col_x + col.width - 10 - _text_width(draw, value, body_font)
            elif col.align == "center":
                tx = col_x + max(10, (col.width - _text_width(draw, value, body_font)) // 2)
            draw.text((tx, y + 10), value, fill=color, font=body_font)
            col_x += col.width
        y += row_h
    draw.line((table_x, y, table_x + table_w, y), fill=_GRID, width=1)

    if footer:
        draw.text((x, y + footer_gap), footer, fill=_MUTED, font=small_font)

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
