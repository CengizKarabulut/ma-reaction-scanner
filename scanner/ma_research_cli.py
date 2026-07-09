#!/usr/bin/env python3
"""Run the guarded multi-timeframe MA support/resistance research panel."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

import pandas as pd

try:
    from .asset_reporting import build_instrument_summary, format_instrument_summary
    from .asset_universe import (
        ASSET_CLASSES,
        build_custom_instruments,
        list_universes,
        resolve_universe,
    )
    from .bist_classification import list_sector_choices
    from .stock_metadata import enrich_stock_instruments, format_index_memberships
except ImportError:  # direct script execution
    from asset_reporting import build_instrument_summary, format_instrument_summary
    from asset_universe import (
        ASSET_CLASSES,
        build_custom_instruments,
        list_universes,
        resolve_universe,
    )
    from bist_classification import list_sector_choices
    from stock_metadata import enrich_stock_instruments, format_index_memberships

try:
    from .ma_core import (
        DEFAULT_PERIODS,
        MA_TYPES,
        TIMEFRAME_CONFIGS,
        analyze_ma_universe,
        build_confluence,
        format_panel,
        select_panel_levels,
    )
    from .ma_data import MarketDataProvider, TIMEFRAMES
except ImportError:  # direct script execution
    from ma_core import (
        DEFAULT_PERIODS,
        MA_TYPES,
        TIMEFRAME_CONFIGS,
        analyze_ma_universe,
        build_confluence,
        format_panel,
        select_panel_levels,
    )
    from ma_data import MarketDataProvider, TIMEFRAMES


LOG = logging.getLogger("ma_research")


def _csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _resolve_instruments(args: argparse.Namespace):
    if args.universe == "custom":
        instruments = build_custom_instruments(
            _csv_list(args.tickers),
            asset_class=args.asset_class,
            market=args.market,
        )
    else:
        if args.universe == "bist_sector_stocks" and str(args.sector).casefold().startswith("tüm"):
            raise ValueError(
                "bist_sector_stocks için belirli bir sektör seçin; "
                "tüm sektörleri taramak için bist30/50/100/all_stocks evrenlerinden birini kullanın"
            )
        instruments = resolve_universe(args.universe, sector=args.sector)
    return enrich_stock_instruments(instruments)


def _write_markdown(panel: pd.DataFrame, confluence: pd.DataFrame, path: Path) -> None:
    lines = ["# MA Support/Resistance Research Panel", ""]
    if panel.empty:
        lines.append("No candidate levels were available.")
    else:
        columns = [
            "ticker",
            "timeframe",
            "side",
            "ma_type",
            "period",
            "current_ma",
            "distance_pct",
            "distance_atr",
            "discovery_events",
            "q_value",
            "status",
        ]
        lines.append(panel[columns].to_markdown(index=False))
    lines.extend(["", "## Confluence context", ""])
    if confluence.empty:
        lines.append("No cross-timeframe clusters.")
    else:
        lines.append(confluence.to_markdown(index=False))
    lines.extend(
        [
            "",
            "> CANDIDATE_ONLY rows are location information, not validated trade signals.",
            "> Confluence is contextual clustering; its timeframes are not independent evidence.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe", default="custom", help="Friendly automatic universe key"
    )
    parser.add_argument("--asset-class", choices=ASSET_CLASSES, default="stock")
    parser.add_argument("--market", choices=["BIST", "GLOBAL"], default=None)
    parser.add_argument("--list-universes", action="store_true")
    parser.add_argument(
        "--show-detail", action="store_true", help="Print detailed MA rows too"
    )
    parser.add_argument(
        "--sector", default=None, help="Türkçe sektör adı veya menü anahtarı"
    )
    parser.add_argument("--list-sectors", action="store_true")
    parser.add_argument(
        "--tickers", default="GARAN,THYAO,AKBNK", help="Custom symbols only"
    )
    parser.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    parser.add_argument(
        "--source", choices=["auto", "borsapy", "yfinance"], default="auto"
    )
    parser.add_argument(
        "--top", type=int, default=5, help="Candidates shown per side/timeframe"
    )
    parser.add_argument("--periods", default=",".join(map(str, DEFAULT_PERIODS)))
    parser.add_argument("--ma-types", default=",".join(MA_TYPES))
    parser.add_argument("--null-iterations", type=int, default=None)
    parser.add_argument("--fdr-q", type=float, default=None)
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Evaluate only the currently active support/resistance side of each MA",
    )
    parser.add_argument(
        "--max-evaluated-distance-atr",
        type=float,
        default=None,
        help=(
            "Skip expensive evidence tests for active MA levels farther than this "
            "ATR distance; a current-location row is still emitted"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Diagnostic run: fewer null draws, no secondary controls",
    )
    parser.add_argument(
        "--probe", action="store_true", help="Only test provider/interval availability"
    )
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--prefer-cache", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.list_universes:
        print(pd.DataFrame(list_universes()).to_string(index=False))
        return 0
    if args.list_sectors:
        print(pd.DataFrame(list_sector_choices()).to_string(index=False))
        return 0
    try:
        instruments = _resolve_instruments(args)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    preview = ", ".join(item.symbol for item in instruments[:12])
    if len(instruments) > 12:
        preview += f", ... (+{len(instruments) - 12})"
    LOG.info(
        "Etkin seçim: evren=%s, varlık_sayısı=%d, semboller=%s",
        args.universe,
        len(instruments),
        preview,
    )

    timeframes = _csv_list(args.timeframes)
    invalid_timeframes = sorted(set(timeframes) - set(TIMEFRAMES))
    if invalid_timeframes:
        raise SystemExit(f"Unsupported timeframes: {invalid_timeframes}")
    periods = [int(x) for x in _csv_list(args.periods)]
    ma_types = [x.upper() for x in _csv_list(args.ma_types)]
    invalid_mas = sorted(set(ma_types) - set(MA_TYPES))
    if invalid_mas:
        raise SystemExit(f"Unsupported MA types: {invalid_mas}")

    side_factor = 1 if args.active_only else 2
    estimated_hypotheses = (
        len(instruments) * len(timeframes) * len(periods) * len(ma_types) * side_factor
    )
    LOG.info(
        "Analiz butcesi: timeframe=%d, periyot=%d, MA=%d, taraf=%d, hipotez~%d",
        len(timeframes),
        len(periods),
        len(ma_types),
        side_factor,
        estimated_hypotheses,
    )
    if args.fast:
        LOG.info(
            "Hizli tarama modu: null_iterations en fazla 29, shift/horizontal kontroller kapali"
        )
    if args.max_evaluated_distance_atr is not None:
        LOG.info(
            "Uzaklik freni: |distance_atr| > %.2f olan aktif seviyeler konum satiri olarak birakilir",
            args.max_evaluated_distance_atr,
        )

    provider = MarketDataProvider(
        source=args.source,
        cache_dir=args.cache_dir,
        snapshot=not args.no_snapshot,
    )
    if args.probe:
        rows = []
        for instrument in instruments:
            for row in provider.probe(
                instrument.provider_symbol,
                asset_class=instrument.asset_class,
                market=instrument.market,
            ):
                row.update(
                    {
                        "symbol": instrument.symbol,
                        "display_name": instrument.display_name,
                        "asset_class": instrument.asset_class,
                        "asset_label": instrument.asset_label,
                        "sector": instrument.sector,
                        "industry": instrument.industry,
                        "index_memberships": format_index_memberships(
                            instrument.index_memberships
                        ),
                    }
                )
                rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
        return 0

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir or f"reports/ma_research_{run_stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    panel_results = []
    provenance = []
    failures = []

    for instrument in instruments:
        ticker = instrument.symbol
        provider_ticker = instrument.provider_symbol
        for timeframe in timeframes:
            cfg = TIMEFRAME_CONFIGS[timeframe]
            if args.fast:
                fast_null_iterations = (
                    29 if args.null_iterations is None else min(args.null_iterations, 29)
                )
                cfg = replace(
                    cfg,
                    null_iterations=fast_null_iterations,
                    use_shift_control=False,
                    use_horizontal_control=False,
                )
            elif args.null_iterations is not None:
                cfg = replace(cfg, null_iterations=args.null_iterations)
            if args.fdr_q is not None:
                cfg = replace(cfg, fdr_q=args.fdr_q)
            try:
                fetched = provider.fetch(
                    provider_ticker,
                    timeframe,
                    prefer_cache=args.prefer_cache,
                    asset_class=instrument.asset_class,
                    market=instrument.market,
                )
                LOG.info(
                    "%s %s: %d bars via %s (%s)",
                    ticker,
                    timeframe,
                    len(fetched.frame),
                    fetched.source,
                    fetched.fingerprint[:12],
                )
                result = analyze_ma_universe(
                    fetched.frame,
                    ticker,
                    timeframe,
                    cfg,
                    ma_types=ma_types,
                    periods=periods,
                    active_only=args.active_only,
                    max_evaluated_distance_atr=args.max_evaluated_distance_atr,
                )
                if result.empty:
                    raise RuntimeError("no MA candidate had enough calculable history")
                all_results.append(result)
                result["asset_class"] = instrument.asset_class
                result["asset_label"] = instrument.asset_label
                result["universe"] = instrument.universe
                result["display_name"] = instrument.display_name
                result["instrument_id"] = instrument.instrument_id
                result["market"] = instrument.market
                result["provider_symbol"] = provider_ticker
                result["sector"] = instrument.sector
                result["industry"] = instrument.industry
                result["index_memberships"] = format_index_memberships(
                    instrument.index_memberships
                )

                panel_results.append(select_panel_levels(result, args.top))
                provenance.append(
                    {
                        "ticker": ticker,
                        "timeframe": timeframe,
                        "source": fetched.source,
                        "provider_symbol": provider_ticker,
                        "display_name": instrument.display_name,
                        "asset_class": instrument.asset_class,
                        "asset_label": instrument.asset_label,
                        "universe": instrument.universe,
                        "market": instrument.market,
                        "sector": instrument.sector,
                        "industry": instrument.industry,
                        "index_memberships": list(instrument.index_memberships),
                        "base_interval": fetched.base_interval,
                        "bars": len(fetched.frame),
                        "first_bar": str(fetched.frame.index[0]),
                        "last_bar": str(fetched.frame.index[-1]),
                        "fingerprint": fetched.fingerprint,
                        "snapshot_path": fetched.snapshot_path,
                        "config": asdict(cfg),
                    }
                )
            except Exception as exc:
                LOG.exception("%s %s failed", ticker, timeframe)
                failures.append(
                    {
                        "ticker": ticker,
                        "provider_symbol": provider_ticker,
                        "timeframe": timeframe,
                        "asset_class": instrument.asset_class,
                        "asset_label": instrument.asset_label,
                        "universe": instrument.universe,
                        "sector": instrument.sector,
                        "index_memberships": list(instrument.index_memberships),
                        "error": str(exc),
                    }
                )
    candidates = (
        pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    )
    panel = (
        pd.concat(panel_results, ignore_index=True) if panel_results else pd.DataFrame()
    )
    summary = build_instrument_summary(candidates)
    confluence = build_confluence(panel)
    if not candidates.empty:
        candidates.to_csv(output_dir / "all_candidates.csv", index=False)
    if not summary.empty:
        summary.to_csv(output_dir / "instrument_summary.csv", index=False)
        summary.to_csv(output_dir / "panel.csv", index=False)
    if not panel.empty:
        panel.to_csv(output_dir / "panel_detail.csv", index=False)
    if not confluence.empty:
        confluence.to_csv(output_dir / "confluence.csv", index=False)
    summary_text = format_instrument_summary(summary)
    (output_dir / "instrument_summary.txt").write_text(summary_text, encoding="utf-8")
    (output_dir / "panel.txt").write_text(summary_text, encoding="utf-8")
    (output_dir / "panel_detail.txt").write_text(format_panel(panel), encoding="utf-8")
    try:
        summary_markdown = "# Tekilleştirilmiş Varlık Özeti\n\n" + summary.to_markdown(
            index=False
        )
        summary_markdown += "\n\n> Her varlık bu özette yalnızca bir kez gösterilir."
        (output_dir / "instrument_summary.md").write_text(
            summary_markdown, encoding="utf-8"
        )
        (output_dir / "panel.md").write_text(summary_markdown, encoding="utf-8")
        _write_markdown(panel, confluence, output_dir / "panel_detail.md")
    except ImportError:
        LOG.warning("tabulate is unavailable; Markdown table skipped")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "universe": args.universe,
        "instrument_count": len(instruments),
        "instruments": [asdict(instrument) for instrument in instruments],
        "provenance": provenance,
        "failures": failures,
        "sector": args.sector,
        "interpretation": {
            "certified": "Passed discovery null/FDR, validation, and untouched holdout gates.",
            "candidate_only": "Location information only; not validated evidence.",
            "confluence": "Context cluster; timeframes are correlated, not independent votes.",
            "instrument_summary": "Exactly one row per typed instrument; MA details are separate.",
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary_text)
    if args.show_detail:
        print("\nMA DETAIL\n" + format_panel(panel))
    print(f"\nOutputs: {output_dir.resolve()}")
    if failures:
        print(
            f"Warnings: {len(failures)} ticker/timeframe failures; see run_metadata.json"
        )
    return 0 if not summary.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
