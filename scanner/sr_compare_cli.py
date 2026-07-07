#!/usr/bin/env python3
"""Compare MA levels with causal pivot and rolling-VWAP alternatives."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path


try:
    from .ma_core import TIMEFRAME_CONFIGS, compute_ma, prepare_frame
    from .ma_data import MarketDataProvider
    from .sr_baselines import causal_baseline_levels, compare_level_families
except ImportError:
    from ma_core import TIMEFRAME_CONFIGS, compute_ma, prepare_frame
    from ma_data import MarketDataProvider
    from sr_baselines import causal_baseline_levels, compare_level_families


def _ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _strings(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--timeframe", choices=list(TIMEFRAME_CONFIGS), default="1d")
    parser.add_argument("--source", choices=["auto", "borsapy", "yfinance"], default="auto")
    parser.add_argument("--windows", default="20,50,100")
    parser.add_argument("--ma-types", default="SMA,EMA,VWMA")
    parser.add_argument("--null-iterations", type=int, default=499)
    parser.add_argument("--output", default="reports/sr_family_comparison.csv")
    args = parser.parse_args(argv)

    windows = _ints(args.windows)
    provider = MarketDataProvider(source=args.source)
    fetched = provider.fetch(args.ticker, args.timeframe)
    config = replace(TIMEFRAME_CONFIGS[args.timeframe], null_iterations=args.null_iterations)
    prepared = prepare_frame(fetched.frame, config)
    levels = causal_baseline_levels(fetched.frame, windows)
    for ma_type in _strings(args.ma_types):
        for period in windows:
            if period >= len(prepared) - config.horizon:
                continue
            label = f"{ma_type}_{period}"
            levels[label] = (
                "moving_average", period,
                compute_ma(ma_type, prepared["Close"], prepared["Volume"], period),
            )
    result = compare_level_families(
        prepared, levels, config, args.ticker.upper(), args.timeframe
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    shown = [
        "level_label", "family", "side", "current_level", "distance_atr",
        "discovery_events", "q_value", "validation_pass", "holdout_pass", "certified",
    ]
    print(result[shown].to_string(index=False))
    print(f"\nOutput: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

