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
    from .ma_core import (
        DEFAULT_PERIODS, MA_TYPES, TIMEFRAME_CONFIGS, analyze_ma_universe,
        build_confluence, format_panel, select_panel_levels,
    )
    from .ma_data import MarketDataProvider, TIMEFRAMES
except ImportError:  # direct script execution
    from ma_core import (
        DEFAULT_PERIODS, MA_TYPES, TIMEFRAME_CONFIGS, analyze_ma_universe,
        build_confluence, format_panel, select_panel_levels,
    )
    from ma_data import MarketDataProvider, TIMEFRAMES


LOG = logging.getLogger("ma_research")


def _csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _write_markdown(panel: pd.DataFrame, confluence: pd.DataFrame, path: Path) -> None:
    lines = ["# MA Support/Resistance Research Panel", ""]
    if panel.empty:
        lines.append("No candidate levels were available.")
    else:
        columns = [
            "ticker", "timeframe", "side", "ma_type", "period", "current_ma",
            "distance_pct", "distance_atr", "discovery_events", "q_value", "status",
        ]
        lines.append(panel[columns].to_markdown(index=False))
    lines.extend(["", "## Confluence context", ""])
    if confluence.empty:
        lines.append("No cross-timeframe clusters.")
    else:
        lines.append(confluence.to_markdown(index=False))
    lines.extend([
        "", "> CANDIDATE_ONLY rows are location information, not validated trade signals.",
        "> Confluence is contextual clustering; its timeframes are not independent evidence.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="GARAN,THYAO,AKBNK", help="Comma-separated symbols")
    parser.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    parser.add_argument("--source", choices=["auto", "borsapy", "yfinance"], default="auto")
    parser.add_argument("--top", type=int, default=5, help="Candidates shown per side/timeframe")
    parser.add_argument("--periods", default=",".join(map(str, DEFAULT_PERIODS)))
    parser.add_argument("--ma-types", default=",".join(MA_TYPES))
    parser.add_argument("--null-iterations", type=int, default=None)
    parser.add_argument("--fdr-q", type=float, default=None)
    parser.add_argument("--fast", action="store_true", help="Diagnostic run: fewer null draws, no secondary controls")
    parser.add_argument("--probe", action="store_true", help="Only test provider/interval availability")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--prefer-cache", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    tickers = [x.upper() for x in _csv_list(args.tickers)]
    timeframes = _csv_list(args.timeframes)
    invalid_timeframes = sorted(set(timeframes) - set(TIMEFRAMES))
    if invalid_timeframes:
        raise SystemExit(f"Unsupported timeframes: {invalid_timeframes}")
    periods = [int(x) for x in _csv_list(args.periods)]
    ma_types = [x.upper() for x in _csv_list(args.ma_types)]
    invalid_mas = sorted(set(ma_types) - set(MA_TYPES))
    if invalid_mas:
        raise SystemExit(f"Unsupported MA types: {invalid_mas}")

    provider = MarketDataProvider(
        source=args.source,
        cache_dir=args.cache_dir,
        snapshot=not args.no_snapshot,
    )
    if args.probe:
        rows = []
        for ticker in tickers:
            for row in provider.probe(ticker):
                row["ticker"] = ticker
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

    for ticker in tickers:
        for timeframe in timeframes:
            cfg = TIMEFRAME_CONFIGS[timeframe]
            if args.fast:
                cfg = replace(cfg, null_iterations=29, use_shift_control=False, use_horizontal_control=False)
            if args.null_iterations is not None:
                cfg = replace(cfg, null_iterations=args.null_iterations)
            if args.fdr_q is not None:
                cfg = replace(cfg, fdr_q=args.fdr_q)
            try:
                fetched = provider.fetch(ticker, timeframe, prefer_cache=args.prefer_cache)
                LOG.info(
                    "%s %s: %d bars via %s (%s)",
                    ticker, timeframe, len(fetched.frame), fetched.source, fetched.fingerprint[:12],
                )
                result = analyze_ma_universe(
                    fetched.frame, ticker, timeframe, cfg, ma_types=ma_types, periods=periods
                )
                if result.empty:
                    raise RuntimeError("no MA candidate had enough calculable history")
                all_results.append(result)
                panel_results.append(select_panel_levels(result, args.top))
                provenance.append({
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "source": fetched.source,
                    "base_interval": fetched.base_interval,
                    "bars": len(fetched.frame),
                    "first_bar": str(fetched.frame.index[0]),
                    "last_bar": str(fetched.frame.index[-1]),
                    "fingerprint": fetched.fingerprint,
                    "snapshot_path": fetched.snapshot_path,
                    "config": asdict(cfg),
                })
            except Exception as exc:
                LOG.exception("%s %s failed", ticker, timeframe)
                failures.append({"ticker": ticker, "timeframe": timeframe, "error": str(exc)})

    candidates = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    panel = pd.concat(panel_results, ignore_index=True) if panel_results else pd.DataFrame()
    confluence = build_confluence(panel)
    if not candidates.empty:
        candidates.to_csv(output_dir / "all_candidates.csv", index=False)
    if not panel.empty:
        panel.to_csv(output_dir / "panel.csv", index=False)
    if not confluence.empty:
        confluence.to_csv(output_dir / "confluence.csv", index=False)
    (output_dir / "panel.txt").write_text(format_panel(panel), encoding="utf-8")
    try:
        _write_markdown(panel, confluence, output_dir / "panel.md")
    except ImportError:
        LOG.warning("tabulate is unavailable; Markdown table skipped")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "provenance": provenance,
        "failures": failures,
        "interpretation": {
            "certified": "Passed discovery null/FDR, validation, and untouched holdout gates.",
            "candidate_only": "Location information only; not validated evidence.",
            "confluence": "Context cluster; timeframes are correlated, not independent votes.",
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(format_panel(panel))
    print(f"\nOutputs: {output_dir.resolve()}")
    if failures:
        print(f"Warnings: {len(failures)} ticker/timeframe failures; see run_metadata.json")
    return 0 if not panel.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())

