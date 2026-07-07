#!/usr/bin/env python3
"""Friendly BIST index and sector catalogue.

Users select Turkish names; provider/index symbols remain an implementation
detail.  The symbols below are the sector indices exposed by borsapy's Index
API, plus XINSA which is already maintained in the repository's verified
component cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


MAIN_INDEX_NAMES: dict[str, str] = {
    "XU030": "BIST 30",
    "XU050": "BIST 50",
    "XU100": "BIST 100",
    "XUTUM": "BIST Tüm",
}

SECTOR_INDEX_NAMES: dict[str, str] = {
    "XBANK": "BIST Banka",
    "XUTEK": "BIST Teknoloji",
    "XGIDA": "BIST Gıda İçecek",
    "XKMYA": "BIST Kimya Petrol Plastik",
    "XHOLD": "BIST Holding ve Yatırım",
    "XSGRT": "BIST Sigorta",
    "XTRZM": "BIST Turizm",
    "XELKT": "BIST Elektrik",
    "XILTM": "BIST İletişim",
    "XINSA": "BIST İnşaat",
    "XMADN": "BIST Madencilik",
    "XMANA": "BIST Metal Ana",
    "XSPOR": "BIST Spor",
    "XTEKS": "BIST Tekstil Deri",
    "XULAS": "BIST Ulaştırma",
    "XUSIN": "BIST Sınai",
    "XUMAL": "BIST Mali",
}


@dataclass(frozen=True)
class SectorChoice:
    key: str
    label: str
    index_symbol: str
    aliases: tuple[str, ...] = ()


SECTOR_CHOICES: tuple[SectorChoice, ...] = (
    SectorChoice("bankacilik", "Bankacılık", "XBANK", ("banka", "bank")),
    SectorChoice("teknoloji", "Teknoloji", "XUTEK", ("tech",)),
    SectorChoice("gida_icecek", "Gıda ve İçecek", "XGIDA", ("gıda", "gida")),
    SectorChoice(
        "kimya_petrol_plastik",
        "Kimya, Petrol ve Plastik",
        "XKMYA",
        ("kimya", "petrol", "plastik"),
    ),
    SectorChoice(
        "holding_yatirim", "Holding ve Yatırım", "XHOLD", ("holding", "yatırım")
    ),
    SectorChoice("sigorta", "Sigorta", "XSGRT"),
    SectorChoice("turizm", "Turizm", "XTRZM"),
    SectorChoice("elektrik", "Elektrik", "XELKT", ("enerji",)),
    SectorChoice("iletisim", "İletişim", "XILTM"),
    SectorChoice("insaat", "İnşaat", "XINSA"),
    SectorChoice("madencilik", "Madencilik", "XMADN", ("maden",)),
    SectorChoice("metal_ana", "Metal Ana", "XMANA", ("metal",)),
    SectorChoice("spor", "Spor", "XSPOR"),
    SectorChoice("tekstil_deri", "Tekstil ve Deri", "XTEKS", ("tekstil",)),
    SectorChoice("ulastirma", "Ulaştırma", "XULAS", ("ulaşım",)),
    SectorChoice("sinai", "Sınai", "XUSIN", ("sanayi",)),
    SectorChoice("mali", "Mali", "XUMAL", ("finans",)),
)


def normalize_choice(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.strip().casefold())
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    ascii_text = ascii_text.translate(str.maketrans({"ı": "i", "ş": "s"}))
    return re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")


def resolve_sector_choice(value: str) -> SectorChoice:
    wanted = normalize_choice(value)
    for choice in SECTOR_CHOICES:
        candidates = {
            normalize_choice(choice.key),
            normalize_choice(choice.label),
            normalize_choice(choice.index_symbol),
            *(normalize_choice(alias) for alias in choice.aliases),
        }
        if wanted in candidates:
            return choice
    available = ", ".join(choice.label for choice in SECTOR_CHOICES)
    raise ValueError(
        f"bilinmeyen sektör '{value}'. Kullanılabilir sektörler: {available}"
    )


def list_sector_choices() -> list[dict[str, str]]:
    return [
        {
            "key": choice.key,
            "sector": choice.label,
            "index": choice.index_symbol,
            "index_name": SECTOR_INDEX_NAMES[choice.index_symbol],
        }
        for choice in SECTOR_CHOICES
    ]


def sector_label_for_index(index_symbol: str) -> str:
    symbol = index_symbol.upper()
    for choice in SECTOR_CHOICES:
        if choice.index_symbol == symbol:
            return choice.label
    return SECTOR_INDEX_NAMES.get(symbol, symbol).removeprefix("BIST ")
