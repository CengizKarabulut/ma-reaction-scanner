#!/usr/bin/env python3
"""Run descriptive MA-respect scans for instruments highlighted by guarded scans.

The guarded pipeline is deliberately strict.  Its behavior tables are more
descriptive: they show the instruments that are often near historically visited
MAs, react well, or simply have many raw MA visits.  This bridge turns those
highlighted instruments into a follow-up MA-respect report without adding a new
workflow to the Actions menu.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .asset_universe import ASSET_CLASSES
    from .guarded_notifier import select_top_instruments
    from .ma_descriptive_cli import DEFAULT_DESC_PERIODS, MA_TYPES, main as descriptive_main
except ImportError:  # direct script execution
    from asset_universe import ASSET_CLASSES
    from guarded_notifier import select_top_instruments
    from ma_descriptive_cli import DEFAULT_DESC_PERIODS, MA_TYPES, main as descriptive_main


_BEHAVIOR_FILES: tuple[tuple[str, str], ...] = (
    ("ma_behavior_near_price.csv", "near_price"),
    ("ma_behavior_best_reactions.csv", "best_reactions"),
    ("ma_behavior_most_visited.csv", "most_visited"),
)
_BIST_ASSET_CLASSES = {"stock", "index", "sector_index"}


@dataclass(frozen=True)
class HighlightInstrument:
    asset_class: str
    symbol: str
    market: str
    source: str


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _asset_class(value: object) -> str:
    text = str(value or "stock").strip().lower()
    return text if text in ASSET_CLASSES else "stock"


def _market_for_asset(asset_class: str) -> str:
    return "BIST" if asset_class in _BIST_ASSET_CLASSES else "GLOBAL"


def _append_unique(
    records: list[HighlightInstrument],
    seen: set[tuple[str, str]],
    frame: pd.DataFrame,
    *,
    source: str,
    limit: int,
) -> None:
    if frame is None or frame.empty or "symbol" not in frame:
        return
    remaining = max(0, int(limit))
    if remaining == 0:
        return
    for _, row in frame.head(remaining).iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol or symbol == "NAN":
            continue
        asset_class = _asset_class(row.get("asset_class"))
        key = (asset_class, symbol)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            HighlightInstrument(
                asset_class=asset_class,
                symbol=symbol,
                market=_market_for_asset(asset_class),
                source=source,
            )
        )


def _touch_filtered(frame: pd.DataFrame, min_touches: int) -> pd.DataFrame:
    if frame.empty or "total_touch_events" not in frame:
        return frame
    touches = pd.to_numeric(frame["total_touch_events"], errors="coerce").fillna(0)
    return frame[touches >= max(1, int(min_touches))].copy()


def collect_highlight_instruments(
    summary_path: str | Path,
    *,
    behavior_dir: str | Path | None = None,
    guarded_top: int = 20,
    behavior_top: int = 20,
    max_symbols: int = 12,
    behavior_min_touches: int = 10,
) -> list[HighlightInstrument]:
    """Collect unique instruments worth a descriptive follow-up scan.

    Behavior tables are prioritized because they are the "extra listed" rows the
    user sees after the strict guarded summary.
    """

    summary_path = Path(summary_path)
    folder = Path(behavior_dir) if behavior_dir is not None else summary_path.parent
    records: list[HighlightInstrument] = []
    seen: set[tuple[str, str]] = set()
    cap = max(0, int(max_symbols))

    def room() -> int:
        return 1_000_000 if cap <= 0 else max(0, cap - len(records))

    for filename, source in _BEHAVIOR_FILES:
        if room() <= 0:
            break
        table = _touch_filtered(_read_csv(folder / filename), behavior_min_touches)
        _append_unique(
            records,
            seen,
            table,
            source=source,
            limit=min(max(1, int(behavior_top)), room()),
        )

    if room() > 0:
        summary = _read_csv(summary_path)
        if not summary.empty:
            try:
                guarded = select_top_instruments(
                    summary,
                    top_n=max(1, int(guarded_top)),
                    min_touches=behavior_min_touches,
                )
            except Exception:
                guarded = summary.head(max(1, int(guarded_top)))
            _append_unique(
                records,
                seen,
                guarded,
                source="guarded_summary",
                limit=room(),
            )
    return records


def write_manifest(output_dir: str | Path, instruments: Iterable[HighlightInstrument]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "asset_class": item.asset_class,
            "market": item.market,
            "symbol": item.symbol,
            "source": item.source,
        }
        for item in instruments
    ]
    pd.DataFrame(rows, columns=["asset_class", "market", "symbol", "source"]).to_csv(
        output / "highlighted_instruments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output / "highlighted_symbols.txt").write_text(
        ",".join(row["symbol"] for row in rows),
        encoding="utf-8",
    )


def _grouped(instruments: Iterable[HighlightInstrument]) -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], list[str]] = {}
    for item in instruments:
        groups.setdefault((item.asset_class, item.market), []).append(item.symbol)
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="guarded instrument_summary.csv path")
    parser.add_argument("--behavior-dir", default=None, help="directory containing ma_behavior_*.csv")
    parser.add_argument("--output-dir", default="reports/descriptive_highlights")
    parser.add_argument("--label", default="Guarded Ek Liste MA Saygi")
    parser.add_argument("--timeframe", default="1d", choices=["1h", "4h", "1d", "1wk", "1mo"])
    parser.add_argument("--periods", default=",".join(map(str, DEFAULT_DESC_PERIODS)))
    parser.add_argument("--ma-types", default=",".join(MA_TYPES))
    parser.add_argument("--source", default="auto", choices=["auto", "borsapy", "yfinance"])
    parser.add_argument("--lookback", type=int, default=750)
    parser.add_argument("--guarded-top", type=int, default=20)
    parser.add_argument("--behavior-top", type=int, default=20)
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--behavior-min-touches", type=int, default=10)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--detail-top", type=int, default=10)
    parser.add_argument("--per-symbol-top", type=int, default=2)
    parser.add_argument("--min-visits", type=int, default=1)
    parser.add_argument("--sort-by", default="visits", choices=["visits", "score"])
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="write manifest only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    instruments = collect_highlight_instruments(
        args.summary,
        behavior_dir=args.behavior_dir,
        guarded_top=args.guarded_top,
        behavior_top=args.behavior_top,
        max_symbols=args.max_symbols,
        behavior_min_touches=args.behavior_min_touches,
    )
    write_manifest(output_dir, instruments)
    if not instruments:
        print("No highlighted instruments found for descriptive MA follow-up.")
        return 0

    print(
        "Descriptive MA follow-up symbols: "
        + ", ".join(f"{item.asset_class}:{item.symbol}" for item in instruments)
    )
    if args.dry_run:
        return 0

    exit_codes: list[int] = []
    groups = _grouped(instruments)
    for (asset_class, market), symbols in groups.items():
        group_label = args.label
        if len(groups) > 1:
            group_label = f"{args.label} ({asset_class})"
        group_dir = output_dir / f"{asset_class}_{market.lower()}"
        cli_args = [
            "--universe",
            "custom",
            "--label",
            group_label,
            "--tickers",
            ",".join(symbols),
            "--asset-class",
            asset_class,
            "--market",
            market,
            "--timeframe",
            args.timeframe,
            "--periods",
            args.periods,
            "--ma-types",
            args.ma_types,
            "--source",
            args.source,
            "--side",
            "auto",
            "--lookback",
            str(args.lookback),
            "--top",
            str(args.top),
            "--detail-top",
            str(args.detail_top),
            "--per-symbol-top",
            str(args.per_symbol_top),
            "--min-visits",
            str(args.min_visits),
            "--sort-by",
            args.sort_by,
            "--output-dir",
            str(group_dir),
        ]
        if args.telegram:
            cli_args.append("--telegram")
        exit_codes.append(descriptive_main(cli_args))
    return 0 if any(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
