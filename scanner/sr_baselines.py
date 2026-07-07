#!/usr/bin/env python3
"""Causal non-MA support/resistance baselines for honest comparison."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .ma_core import AnalysisConfig, adjust_fdr, evaluate_candidate, normalize_ohlcv
except ImportError:
    from ma_core import AnalysisConfig, adjust_fdr, evaluate_candidate, normalize_ohlcv


def causal_baseline_levels(
    frame: pd.DataFrame,
    windows: Sequence[int] = (20, 50, 100),
) -> dict[str, tuple[str, int, pd.Series]]:
    """Create levels using only information available before each bar.

    Returned mapping values are ``(family, window, series)``.  Rolling pivots
    use robust quantiles rather than a single extreme; rolling VWAP supplies a
    volume-based alternative to moving averages.
    """

    df = normalize_ohlcv(frame)
    previous_low = df["Low"].shift(1)
    previous_high = df["High"].shift(1)
    typical = ((df["High"] + df["Low"] + df["Close"]) / 3.0).shift(1)
    previous_volume = df["Volume"].shift(1)
    levels: dict[str, tuple[str, int, pd.Series]] = {}
    for window in windows:
        if window < 5:
            raise ValueError("baseline windows must be at least 5")
        levels[f"PIVOT_SUPPORT_{window}"] = (
            "pivot", window, previous_low.rolling(window, min_periods=window).quantile(0.10)
        )
        levels[f"PIVOT_RESISTANCE_{window}"] = (
            "pivot", window, previous_high.rolling(window, min_periods=window).quantile(0.90)
        )
        denominator = previous_volume.rolling(window, min_periods=window).sum().replace(0, np.nan)
        rolling_vwap = (typical * previous_volume).rolling(window, min_periods=window).sum() / denominator
        levels[f"VWAP_{window}"] = ("volume_profile_proxy", window, rolling_vwap)
    return levels


def compare_level_families(
    prepared_frame: pd.DataFrame,
    levels: Mapping[str, tuple[str, int, pd.Series]],
    config: AnalysisConfig,
    ticker: str = "",
    timeframe: str = "",
) -> pd.DataFrame:
    """Run the same null/validation/holdout gates across level families."""

    rows = []
    price = float(prepared_frame["Close"].iloc[-1])
    atr_now = float(prepared_frame["ATR"].iloc[-1])
    for index, (label, (family, window, series)) in enumerate(levels.items()):
        current = float(series.iloc[-1]) if np.isfinite(series.iloc[-1]) else np.nan
        if not np.isfinite(current):
            continue
        side = 1 if current <= price else -1
        row = evaluate_candidate(
            prepared_frame, series, label, window, side, config,
            seed=config.random_seed + index * 10_007,
        )
        row.update({
            "ticker": ticker,
            "timeframe": timeframe,
            "family": family,
            "level_label": label,
            "current_level": current,
            "distance_atr": (current - price) / atr_now if atr_now > 0 else np.nan,
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value"] = adjust_fdr(result["p_value"].to_numpy(dtype=float), config.fdr_method)
    result["discovery_pass"] = (
        (result["discovery_events"] >= config.min_events)
        & (result["discovery_score"] > 0)
        & (result["q_value"] <= config.fdr_q)
    )
    result["certified"] = result["discovery_pass"] & result["validation_pass"] & result["holdout_pass"]
    result["comparison_rank"] = (
        100 * result["certified"].astype(int)
        + result["holdout_score"].fillna(-10)
        + result["validation_score"].fillna(-10)
    )
    return result.sort_values("comparison_rank", ascending=False).reset_index(drop=True)

