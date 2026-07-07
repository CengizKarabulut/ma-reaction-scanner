#!/usr/bin/env python3
"""Attach BIST sector and index memberships to typed stock instruments."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Iterable, Mapping

try:
    from .asset_universe import AssetInstrument
    from .bist_classification import (
        MAIN_INDEX_NAMES,
        SECTOR_INDEX_NAMES,
        sector_label_for_index,
    )
except ImportError:  # direct script execution
    from asset_universe import AssetInstrument
    from bist_classification import (
        MAIN_INDEX_NAMES,
        SECTOR_INDEX_NAMES,
        sector_label_for_index,
    )


DEFAULT_INDEX_CACHE = Path(__file__).parent / "data" / "bist_indices.json"
DEFAULT_SECTOR_CACHE = Path(__file__).parent / ".cache" / "sectors_cache.json"
_MAIN_PRIORITY = tuple(MAIN_INDEX_NAMES)
_SECTOR_PRIORITY = tuple(SECTOR_INDEX_NAMES)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def load_index_components(path: str | Path | None = None) -> dict[str, set[str]]:
    """Load verified borsapy-derived index components from the repository cache."""

    payload = _read_json(Path(path) if path else DEFAULT_INDEX_CACHE)
    result: dict[str, set[str]] = {}
    for symbol, entry in payload.get("indices", {}).items():
        if not isinstance(entry, dict) or entry.get("verified") is False:
            continue
        members = {
            str(member).replace(".IS", "").upper()
            for member in entry.get("symbols", [])
            if str(member).strip()
        }
        if members:
            result[str(symbol).upper()] = members
    return result


def load_stock_details(path: str | Path | None = None) -> dict[str, dict]:
    payload = _read_json(Path(path) if path else DEFAULT_SECTOR_CACHE)
    details = payload.get("sectors", {})
    if not isinstance(details, dict):
        return {}
    return {str(symbol).upper(): value for symbol, value in details.items()}


def invert_index_components(
    components: Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    memberships: dict[str, set[str]] = {}
    for index_symbol, members in components.items():
        index_code = str(index_symbol).upper()
        for member in members:
            ticker = str(member).replace(".IS", "").upper()
            memberships.setdefault(ticker, set()).add(index_code)
    return {
        ticker: tuple(sorted(codes, key=_membership_sort_key))
        for ticker, codes in memberships.items()
    }


def _membership_sort_key(symbol: str) -> tuple[int, int | str]:
    if symbol in _MAIN_PRIORITY:
        return (0, _MAIN_PRIORITY.index(symbol))
    if symbol in _SECTOR_PRIORITY:
        return (1, _SECTOR_PRIORITY.index(symbol))
    return (2, symbol)


def _sector_text(memberships: Iterable[str]) -> str:
    labels: list[str] = []
    for symbol in memberships:
        if symbol not in SECTOR_INDEX_NAMES:
            continue
        label = sector_label_for_index(symbol)
        if label not in labels:
            labels.append(label)
    return " / ".join(labels)


def enrich_stock_instruments(
    instruments: Iterable[AssetInstrument],
    *,
    index_components: Mapping[str, Iterable[str]] | None = None,
    stock_details: Mapping[str, Mapping[str, object]] | None = None,
) -> list[AssetInstrument]:
    """Enrich BIST stocks without making the scan depend on live metadata calls."""

    items = list(instruments)
    components = (
        {key: set(value) for key, value in index_components.items()}
        if index_components is not None
        else load_index_components()
    )
    memberships_by_ticker = invert_index_components(components)
    details = dict(stock_details) if stock_details is not None else load_stock_details()

    enriched: list[AssetInstrument] = []
    for instrument in items:
        if instrument.asset_class != "stock" or instrument.market != "BIST":
            enriched.append(instrument)
            continue
        symbol = instrument.symbol.replace(".IS", "").upper()
        combined = set(instrument.index_memberships)
        combined.update(memberships_by_ticker.get(symbol, ()))
        memberships = tuple(sorted(combined, key=_membership_sort_key))
        detail = details.get(symbol, {})
        cached_sector = str(detail.get("sector") or "").strip()
        derived_sector = _sector_text(memberships)
        industry = instrument.industry or str(detail.get("industry") or "").strip()
        cached_name = str(detail.get("name") or "").strip()
        display_name = instrument.display_name
        if display_name == instrument.symbol and cached_name:
            display_name = cached_name
        enriched.append(
            replace(
                instrument,
                display_name=display_name,
                sector=instrument.sector or derived_sector or cached_sector,
                industry=industry,
                index_memberships=memberships,
            )
        )
    return enriched


def format_index_memberships(memberships: Iterable[str]) -> str:
    """Human-readable code + name representation for CSV/text reports."""

    labels = {**MAIN_INDEX_NAMES, **SECTOR_INDEX_NAMES}
    return "; ".join(
        f"{symbol} ({labels[symbol]})" if symbol in labels else symbol
        for symbol in memberships
    )
