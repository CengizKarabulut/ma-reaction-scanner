#!/usr/bin/env python3
"""Benchmark-relative price series for BIST moving-average research.

Nominal TRY series carry a structural uptrend that has nothing to do with any
moving average: with high inflation, almost every long-side "support held"
statistic looks good simply because the number went up.  ``auto_adjust`` in the
data layer corrects dividends and splits, not the unit of account.

Dividing the series by a benchmark (XU100 by default) removes the common
market and currency drift and leaves the part that is specific to the
instrument.  The result is then *rebased* to the benchmark's most recent
value, which keeps the analysis in readable, present-day lira: the last bar of
the relative series equals the last nominal close exactly, so a moving-average
level computed on the deflated series is still a usable price level on today's
chart.
"""

from __future__ import annotations

import pandas as pd

try:
    from .ma_engine import normalize_ohlcv
except ImportError:  # direct script execution
    from ma_engine import normalize_ohlcv


_PRICE_COLUMNS = ("Open", "High", "Low", "Close")


def to_relative(
    frame: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    rebase: bool = True,
    allow_non_positive: bool = False,
) -> pd.DataFrame:
    """Divide an OHLCV frame by a benchmark close, optionally rebased.

    Dividing all four price columns by the same strictly positive number keeps
    the OHLC ordering intact, so the result still passes ``normalize_ohlcv``.
    Volume is left untouched: it is a share count, not a price.
    """

    df = normalize_ohlcv(frame, allow_non_positive=allow_non_positive)
    reference = normalize_ohlcv(benchmark)["Close"]
    aligned = reference.reindex(df.index).ffill()
    valid = aligned.notna() & (aligned > 0)
    if not bool(valid.any()):
        raise ValueError("Endeks serisi ile ortak bar bulunamadi")
    df = df.loc[valid]
    aligned = aligned.loc[valid]

    scale = float(aligned.iloc[-1]) if rebase else 1.0
    if not scale > 0:
        raise ValueError("Endeks son degeri pozitif olmalidir")

    relative = df.copy()
    for column in _PRICE_COLUMNS:
        relative[column] = df[column] / aligned * scale
    return normalize_ohlcv(relative, allow_non_positive=allow_non_positive)


class BenchmarkCache:
    """Fetch each benchmark series once per timeframe, not once per symbol.

    A full-BIST scan touches hundreds of symbols across eight timeframes.
    Without memoisation the benchmark would be downloaded once per symbol,
    which multiplies provider load and run time by the size of the universe.
    """

    def __init__(
        self,
        provider,
        symbol: str,
        *,
        asset_class: str = "index",
        market: str = "BIST",
        prefer_cache: bool = False,
    ) -> None:
        self.provider = provider
        self.symbol = str(symbol).upper()
        self.asset_class = asset_class
        self.market = market
        self.prefer_cache = prefer_cache
        self._frames: dict[str, pd.DataFrame] = {}
        self._failures: dict[str, str] = {}

    def get(self, timeframe: str) -> pd.DataFrame:
        if timeframe in self._frames:
            return self._frames[timeframe]
        if timeframe in self._failures:
            raise RuntimeError(self._failures[timeframe])
        try:
            fetched = self.provider.fetch(
                self.symbol,
                timeframe,
                prefer_cache=self.prefer_cache,
                asset_class=self.asset_class,
                market=self.market,
            )
        except Exception as exc:  # cache the failure so we retry only once
            message = f"Endeks verisi alinamadi ({self.symbol} {timeframe}): {exc}"
            self._failures[timeframe] = message
            raise RuntimeError(message) from exc
        self._frames[timeframe] = fetched.frame
        return fetched.frame
