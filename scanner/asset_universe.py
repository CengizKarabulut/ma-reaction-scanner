#!/usr/bin/env python3
"""Typed asset catalog and automatic universe resolution.

The research pipeline must never infer that every symbol is a BIST stock.
This module keeps stocks, indices, sector indices, crypto assets and
commodities separate from selection through reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable

try:
    from .bist_classification import (
        MAIN_INDEX_NAMES,
        SECTOR_INDEX_NAMES,
        resolve_sector_choice,
        sector_label_for_index,
    )
except ImportError:  # direct script execution
    from bist_classification import (
        MAIN_INDEX_NAMES,
        SECTOR_INDEX_NAMES,
        resolve_sector_choice,
        sector_label_for_index,
    )


ASSET_LABELS: dict[str, str] = {
    "stock": "Hisse",
    "index": "Endeks",
    "sector_index": "Sektör Endeksi",
    "crypto": "Kripto",
    "commodity": "Emtia",
}
ASSET_CLASSES: tuple[str, ...] = tuple(ASSET_LABELS)


@dataclass(frozen=True)
class AssetInstrument:
    symbol: str
    display_name: str
    asset_class: str
    universe: str
    market: str
    provider_symbol: str
    sector: str = ""
    industry: str = ""
    index_memberships: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.asset_class not in ASSET_CLASSES:
            raise ValueError(f"unsupported asset class: {self.asset_class}")

    @property
    def asset_label(self) -> str:
        return ASSET_LABELS[self.asset_class]

    @property
    def instrument_id(self) -> str:
        return f"{self.asset_class}:{self.symbol}"


@dataclass(frozen=True)
class UniverseDefinition:
    key: str
    label: str
    asset_class: str
    description: str


UNIVERSES: tuple[UniverseDefinition, ...] = (
    UniverseDefinition(
        "bist30_stocks", "BIST 30 Hisseleri", "stock", "BIST 30 bileşen hisseleri"
    ),
    UniverseDefinition(
        "bist50_stocks", "BIST 50 Hisseleri", "stock", "BIST 50 bileşen hisseleri"
    ),
    UniverseDefinition(
        "bist100_stocks", "BIST 100 Hisseleri", "stock", "BIST 100 bileşen hisseleri"
    ),
    UniverseDefinition(
        "bist_all_stocks", "Tüm BIST Hisseleri", "stock", "BIST Tüm bileşenleri"
    ),
    UniverseDefinition(
        "bist_bank_stocks", "Banka Hisseleri", "stock", "XBANK bileşen hisseleri"
    ),
    UniverseDefinition(
        "bist_technology_stocks",
        "Teknoloji Hisseleri",
        "stock",
        "XUTEK bileşen hisseleri",
    ),
    UniverseDefinition(
        "bist_food_stocks", "Gıda Hisseleri", "stock", "XGIDA bileşen hisseleri"
    ),
    UniverseDefinition(
        "bist_chemistry_stocks", "Kimya Hisseleri", "stock", "XKMYA bileşen hisseleri"
    ),
    UniverseDefinition(
        "bist_sector_stocks",
        "Seçilen Sektörün Hisseleri",
        "stock",
        "Türkçe sektör adıyla seçilen BIST sektör endeksinin bileşenleri",
    ),
    UniverseDefinition(
        "bist_main_indices",
        "Ana BIST Endeksleri",
        "index",
        "XU030, XU050, XU100 ve XUTUM",
    ),
    UniverseDefinition(
        "bist_sector_indices",
        "BIST Sektör Endeksleri",
        "sector_index",
        "Sektör endekslerinin kendileri",
    ),
    UniverseDefinition(
        "bist_all_indices",
        "Tüm BIST Endeksleri",
        "index",
        "Sağlayıcının bildirdiği benzersiz BIST endeksleri",
    ),
    UniverseDefinition(
        "crypto_majors",
        "Majör Kripto Paralar",
        "crypto",
        "USD kotasyonlu majör kripto varlıklar",
    ),
    UniverseDefinition(
        "commodities_majors",
        "Majör Emtialar",
        "commodity",
        "Vadeli altın, gümüş, petrol, gaz ve bakır",
    ),
    UniverseDefinition(
        "custom",
        "Özel Sembol Listesi",
        "stock",
        "Kullanıcının açıkça verdiği tek varlık sınıfı",
    ),
)


_DEFINITION_BY_KEY = {item.key: item for item in UNIVERSES}
_ALIASES = {
    "bist_30": "bist30_stocks",
    "bist_50": "bist50_stocks",
    "bist_100": "bist100_stocks",
    "bist_tum": "bist_all_stocks",
    "bist_endeksler": "bist_main_indices",
    "kripto": "crypto_majors",
    "emtia": "commodities_majors",
}

_STOCK_RESOLVERS = {
    "bist30_stocks": "BIST_30",
    "bist50_stocks": "BIST_50",
    "bist100_stocks": "BIST_100",
    "bist_all_stocks": "BIST_TUM",
    "bist_bank_stocks": "BIST_BILES:XBANK",
    "bist_technology_stocks": "BIST_BILES:XUTEK",
    "bist_food_stocks": "BIST_BILES:XGIDA",
    "bist_chemistry_stocks": "BIST_BILES:XKMYA",
}

_FIXED_SECTOR_UNIVERSES = {
    "bist_bank_stocks": "XBANK",
    "bist_technology_stocks": "XUTEK",
    "bist_food_stocks": "XGIDA",
    "bist_chemistry_stocks": "XKMYA",
}
_CRYPTO = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "BNB",
    "XRP-USD": "XRP",
    "SOL-USD": "Solana",
    "ADA-USD": "Cardano",
    "DOGE-USD": "Dogecoin",
    "AVAX-USD": "Avalanche",
    "LINK-USD": "Chainlink",
    "DOT-USD": "Polkadot",
}
_COMMODITIES = {
    "GC=F": "Altın",
    "SI=F": "Gümüş",
    "CL=F": "WTI Petrol",
    "BZ=F": "Brent Petrol",
    "NG=F": "Doğal Gaz",
    "HG=F": "Bakır",
}


def _normalize_universe_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return _ALIASES.get(key, key)


def list_universes() -> list[dict[str, str]]:
    return [
        {
            "key": item.key,
            "label": item.label,
            "asset_class": item.asset_class,
            "asset_label": ASSET_LABELS[item.asset_class],
            "description": item.description,
        }
        for item in UNIVERSES
    ]


def _unique(instruments: Iterable[AssetInstrument]) -> list[AssetInstrument]:
    seen: set[tuple[str, str]] = set()
    unique: list[AssetInstrument] = []
    for instrument in instruments:
        key = (instrument.asset_class, instrument.symbol.upper())
        if key in seen:
            continue
        seen.add(key)
        unique.append(instrument)
    return unique


def _make_instruments(
    members: Iterable[tuple[str, str]],
    asset_class: str,
    universe: str,
    market: str,
    sector: str = "",
    index_memberships: tuple[str, ...] = (),
) -> list[AssetInstrument]:
    return _unique(
        AssetInstrument(
            symbol=symbol.upper(),
            display_name=name,
            asset_class=asset_class,
            universe=universe,
            market=market,
            provider_symbol=symbol.upper(),
            sector=sector,
            index_memberships=index_memberships,
        )
        for symbol, name in members
    )


def _load_stock_members(resolver_name: str) -> list[str]:
    try:
        from .tickers import get_list
    except ImportError:  # direct script execution
        from tickers import get_list
    return list(get_list(resolver_name))


def _load_all_indices() -> list[tuple[str, str]]:
    try:
        from .sector_resolver import list_all_indices
    except ImportError:  # direct script execution
        from sector_resolver import list_all_indices

    members: list[tuple[str, str]] = []
    try:
        for item in list_all_indices(detailed=True):
            if isinstance(item, dict):
                symbol = str(item.get("symbol", "")).upper()
                name = str(item.get("name") or symbol)
            else:
                symbol = str(item).upper()
                name = symbol
            if symbol:
                members.append((symbol, name))
    except Exception:
        members = []
    return members or list({**MAIN_INDEX_NAMES, **SECTOR_INDEX_NAMES}.items())


def resolve_universe(
    name: str,
    *,
    stock_loader: Callable[[str], list[str]] | None = None,
    sector: str | None = None,
) -> list[AssetInstrument]:
    """Resolve a friendly universe into unique typed instruments."""

    key = _normalize_universe_key(name)
    if key not in _DEFINITION_BY_KEY:
        choices = ", ".join(item.key for item in UNIVERSES)
        raise ValueError(f"unknown universe '{name}'. Available: {choices}")
    if key == "custom":
        raise ValueError("custom universe requires explicit symbols")

    if key == "bist_sector_stocks":
        if not sector:
            raise ValueError("bist_sector_stocks için --sector seçimi zorunludur")
        choice = resolve_sector_choice(sector)
        loader = stock_loader or _load_stock_members
        symbols = loader(f"BIST_BILES:{choice.index_symbol}")
        if not symbols:
            raise RuntimeError(
                f"'{choice.label}' sektörü ({choice.index_symbol}) için hisse bulunamadı"
            )
        return _make_instruments(
            ((symbol, symbol) for symbol in symbols),
            "stock",
            key,
            "BIST",
            sector=choice.label,
            index_memberships=(choice.index_symbol,),
        )

    if key in _STOCK_RESOLVERS:
        loader = stock_loader or _load_stock_members
        symbols = loader(_STOCK_RESOLVERS[key])
        if not symbols:
            raise RuntimeError(f"universe '{key}' resolved to no stocks")
        sector_index = _FIXED_SECTOR_UNIVERSES.get(key)
        membership = (sector_index,) if sector_index else ()
        sector_label = sector_label_for_index(sector_index) if sector_index else ""
        return _make_instruments(
            ((symbol, symbol) for symbol in symbols),
            "stock",
            key,
            "BIST",
            sector=sector_label,
            index_memberships=membership,
        )
    if key == "bist_main_indices":
        return _make_instruments(MAIN_INDEX_NAMES.items(), "index", key, "BIST")
    if key == "bist_sector_indices":
        return _make_instruments(
            SECTOR_INDEX_NAMES.items(), "sector_index", key, "BIST"
        )
    if key == "bist_all_indices":
        return _make_instruments(_load_all_indices(), "index", key, "BIST")
    if key == "crypto_majors":
        return _make_instruments(_CRYPTO.items(), "crypto", key, "GLOBAL")
    if key == "commodities_majors":
        return _make_instruments(_COMMODITIES.items(), "commodity", key, "GLOBAL")
    raise AssertionError(f"unhandled universe: {key}")


def build_custom_instruments(
    symbols: Iterable[str],
    *,
    asset_class: str,
    market: str | None = None,
    universe: str = "custom",
) -> list[AssetInstrument]:
    if asset_class not in ASSET_CLASSES:
        raise ValueError(f"unsupported asset class: {asset_class}")
    resolved_market = market or (
        "BIST" if asset_class in {"stock", "index", "sector_index"} else "GLOBAL"
    )
    cleaned = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    if not cleaned:
        raise ValueError("at least one custom symbol is required")
    return _make_instruments(
        ((symbol, symbol) for symbol in cleaned), asset_class, universe, resolved_market
    )
