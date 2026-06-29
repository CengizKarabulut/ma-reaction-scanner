#!/usr/bin/env python3
"""Statistically guarded moving-average support/resistance research core.

This module deliberately separates three concepts that the legacy scanner mixed:

1. A *candidate level*: a moving average currently below/above price.
2. Historical evidence: independent touches followed by measurable reactions.
3. A *certified level*: evidence survives null controls, multiple-testing
   correction, validation and an untouched holdout segment.

Certification is evidence about historical conditional behaviour.  It is not a
promise that price will reverse at the next touch.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MA_TYPES: tuple[str, ...] = ("SMA", "EMA", "WMA", "VWMA", "KAMA", "ALMA", "HMA")
DEFAULT_PERIODS: tuple[int, ...] = (
    3, 5, 8, 10, 13, 20, 21, 22, 34, 50,
    55, 89, 100, 144, 200, 233, 250, 377, 610, 987,
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters fixed before a scan is evaluated."""

    atr_period: int = 14
    adx_period: int = 14
    adx_threshold: float = 25.0
    zone_atr: float = 0.20
    separation_atr: float = 2.0
    horizon: int = 10
    target_atr: float = 1.0
    stop_atr: float = 1.0
    min_events: int = 10
    min_segment_events: int = 3
    discovery_fraction: float = 0.60
    validation_fraction: float = 0.20
    null_iterations: int = 499
    null_quantile: float = 0.90
    fdr_q: float = 0.10
    # BH is used for the positively dependent MA family. BY remains available
    # for deliberately conservative audits but requires far more null draws.
    fdr_method: str = "bh"
    min_wilson: float = 0.35
    max_actionable_distance_atr: float = 4.0
    random_seed: int = 1729
    use_shift_control: bool = True
    use_horizontal_control: bool = True

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.min_events < 1 or self.min_segment_events < 1:
            raise ValueError("event thresholds must be positive")
        if not 0.40 <= self.discovery_fraction <= 0.80:
            raise ValueError("discovery_fraction must be between 0.40 and 0.80")
        if not 0.10 <= self.validation_fraction <= 0.40:
            raise ValueError("validation_fraction must be between 0.10 and 0.40")
        if self.discovery_fraction + self.validation_fraction >= 0.95:
            raise ValueError("a non-trivial holdout segment is required")
        if self.fdr_method not in {"bh", "by"}:
            raise ValueError("fdr_method must be 'bh' or 'by'")


TIMEFRAME_CONFIGS: Mapping[str, AnalysisConfig] = {
    "5m": AnalysisConfig(horizon=12, min_events=18, min_segment_events=5, zone_atr=0.18),
    "15m": AnalysisConfig(horizon=8, min_events=16, min_segment_events=4, zone_atr=0.18),
    "30m": AnalysisConfig(horizon=8, min_events=14, min_segment_events=4),
    "1h": AnalysisConfig(horizon=8, min_events=12, min_segment_events=3),
    "4h": AnalysisConfig(horizon=6, min_events=10, min_segment_events=3),
    "1d": AnalysisConfig(horizon=10, min_events=10, min_segment_events=3),
    "1wk": AnalysisConfig(horizon=6, min_events=7, min_segment_events=2),
    "1mo": AnalysisConfig(horizon=4, min_events=5, min_segment_events=2),
}


@dataclass(frozen=True)
class TouchEvent:
    position: int
    timestamp: object
    direction: int  # +1: support reaction, -1: resistance rejection
    regime: str
    atr: float
    entry: float
    ma_value: float
    volatility_bin: int
    session_bin: int
    approach_bin: int = 0


@dataclass(frozen=True)
class EventMeasurement:
    position: int
    direction: int
    first_hit: int  # +1 target, -1 stop/ambiguous, 0 neither
    fixed_return_atr: float
    favorable_atr: float
    adverse_atr: float
    bars_to_target: float
    retested: bool
    ambiguous_bar: bool


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted numeric OHLCV data with canonical column names."""

    if frame is None or frame.empty:
        raise ValueError("OHLCV data is empty")
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for col in df.columns:
            parts = [str(x) for x in col if str(x) and str(x) != "None"]
            canonical = next((x for x in parts if x.lower() in {
                "open", "high", "low", "close", "adj close", "volume"
            }), parts[0])
            flattened.append(canonical)
        df.columns = flattened

    aliases = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    rename: dict[object, str] = {}
    for name in ("Open", "High", "Low", "Close", "Volume"):
        key = name.lower()
        if key in aliases:
            rename[aliases[key]] = name
    if "Close" not in rename.values() and "adj close" in aliases:
        rename[aliases["adj close"]] = "Close"
    df = df.rename(columns=rename)
    missing = [c for c in ("Open", "High", "Low", "Close") if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLC columns: {missing}")
    if "Volume" not in df.columns:
        df["Volume"] = 1.0
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0)
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    upper = df[["Open", "Close", "Low"]].max(axis=1)
    lower = df[["Open", "Close", "High"]].min(axis=1)
    scale = df[["Open", "High", "Low", "Close"]].abs().max(axis=1)
    tolerance = np.maximum(scale * 1e-12, 1e-12)
    invalid = (df["High"] + tolerance < upper) | (
        df["Low"] - tolerance > lower
    )
    if invalid.any():
        raise ValueError(f"invalid OHLC ordering in {int(invalid.sum())} rows")
    # Data vendors occasionally differ by one floating-point ULP after price
    # adjustment. Clamp only rows already proven to be inside the tolerance.
    df["High"] = df[["High", "Open", "Close", "Low"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close", "High"]].min(axis=1)
    return df


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period, min_periods=period).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
    )


def vwma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    denom = volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    return (close * volume).rolling(period, min_periods=period).sum() / denom


def kama(series: pd.Series, period: int, fast: int = 2, slow: int = 30) -> pd.Series:
    change = (series - series.shift(period)).abs()
    volatility = series.diff().abs().rolling(period, min_periods=period).sum()
    efficiency = (change / volatility.replace(0, np.nan)).fillna(0.0)
    fastest = 2.0 / (fast + 1)
    slowest = 2.0 / (slow + 1)
    smoothing = ((efficiency * (fastest - slowest)) + slowest) ** 2
    out = np.full(len(series), np.nan)
    values = series.to_numpy(dtype=float)
    smooth = smoothing.to_numpy(dtype=float)
    started = False
    for i in range(len(values)):
        if not started and i >= period and np.isfinite(values[i]):
            out[i] = values[i]
            started = True
        elif started:
            out[i] = out[i - 1] + smooth[i] * (values[i] - out[i - 1])
    return pd.Series(out, index=series.index)


def alma(series: pd.Series, period: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    center = offset * (period - 1)
    width = period / sigma
    x = np.arange(period)
    weights = np.exp(-((x - center) ** 2) / (2 * width * width))
    weights /= weights.sum()
    return series.rolling(period, min_periods=period).apply(
        lambda values: float(np.dot(values, weights)), raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    half = max(1, period // 2)
    root = max(1, int(sqrt(period)))
    return wma(2.0 * wma(series, half) - wma(series, period), root)


def compute_ma(ma_type: str, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """The single canonical MA implementation for the repository."""

    kind = ma_type.upper()
    if period < 2:
        raise ValueError("MA period must be at least 2")
    if kind == "SMA":
        return sma(close, period)
    if kind == "EMA":
        return ema(close, period)
    if kind == "WMA":
        return wma(close, period)
    if kind == "VWMA":
        return vwma(close, volume, period)
    if kind == "KAMA":
        return kama(close, period)
    if kind == "ALMA":
        return alma(close, period)
    if kind == "HMA":
        return hma(close, period)
    raise ValueError(f"unknown MA type: {ma_type}")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous = close.shift(1)
    return pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_values = true_range(high, low, close).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_values
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_values
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def prepare_frame(frame: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    df = normalize_ohlcv(frame)
    df["ATR"] = atr(df["High"], df["Low"], df["Close"], config.atr_period)
    df["ADX"] = adx(df["High"], df["Low"], df["Close"], config.adx_period)
    volatility = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
    ranked = volatility.rank(method="first", pct=True)
    df["VOL_BIN"] = np.minimum((ranked.fillna(0.5) * 4).astype(int), 3)
    if isinstance(df.index, pd.DatetimeIndex) and any(df.index.hour != 0):
        df["SESSION_BIN"] = np.where(df.index.hour < 14, 0, 1)
    else:
        df["SESSION_BIN"] = 0
    return df


def detect_independent_touches(
    df: pd.DataFrame,
    ma_series: pd.Series,
    config: AnalysisConfig,
) -> list[TouchEvent]:
    """Detect non-overlapping touches approached from the same side."""

    ma = ma_series.reindex(df.index).to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    atr_values = df["ATR"].to_numpy(dtype=float)
    adx_values = df["ADX"].to_numpy(dtype=float)
    vol_bins = df["VOL_BIN"].to_numpy(dtype=int)
    session_bins = df["SESSION_BIN"].to_numpy(dtype=int)
    events: list[TouchEvent] = []
    far_support = False
    far_resistance = False
    previous_in_zone = False
    last_event = -config.horizon - 1

    for i in range(1, len(df) - config.horizon):
        if not (np.isfinite(ma[i]) and np.isfinite(atr_values[i]) and atr_values[i] > 0):
            previous_in_zone = False
            far_support = far_resistance = False
            continue
        signed_distance = (close[i] - ma[i]) / atr_values[i]
        zone = config.zone_atr * atr_values[i]
        in_zone = low[i] <= ma[i] + zone and high[i] >= ma[i] - zone
        if not in_zone:
            if signed_distance >= config.separation_atr:
                far_support, far_resistance = True, False
            elif signed_distance <= -config.separation_atr:
                far_support, far_resistance = False, True

        new_zone_entry = in_zone and not previous_in_zone
        if new_zone_entry:
            prev_signed = (
                (close[i - 1] - ma[i - 1]) / atr_values[i - 1]
                if np.isfinite(ma[i - 1]) and np.isfinite(atr_values[i - 1]) and atr_values[i - 1] > 0
                else 0.0
            )
            direction = 1 if prev_signed > 0 else -1
            approached_correctly = far_support if direction == 1 else far_resistance
            independent = i - last_event > config.horizon
            if approached_correctly and independent:
                regime = "trend" if np.isfinite(adx_values[i]) and adx_values[i] >= config.adx_threshold else "range"
                lookback = max(0, i - config.horizon)
                approach_strength = max(0.0, -direction * (close[i - 1] - close[lookback]) / atr_values[i])
                approach_bin = int(min(3, np.floor(approach_strength)))
                events.append(TouchEvent(
                    position=i,
                    timestamp=df.index[i],
                    direction=direction,
                    regime=regime,
                    atr=float(atr_values[i]),
                    entry=float(close[i]),
                    ma_value=float(ma[i]),
                    volatility_bin=int(vol_bins[i]),
                    session_bin=int(session_bins[i]),
                    approach_bin=approach_bin,
                ))
                last_event = i
            far_support = far_resistance = False
        previous_in_zone = in_zone
    return events


def measure_event(
    df: pd.DataFrame,
    position: int,
    direction: int,
    config: AnalysisConfig,
) -> EventMeasurement | None:
    """Measure first-hit and fixed-horizon outcomes without peak-exit hindsight."""

    end = position + config.horizon
    if position < 0 or end >= len(df):
        return None
    entry = float(df["Close"].iloc[position])
    atr_value = float(df["ATR"].iloc[position])
    if not np.isfinite(atr_value) or atr_value <= 0:
        return None
    target = entry + direction * config.target_atr * atr_value
    stop = entry - direction * config.stop_atr * atr_value
    window = df.iloc[position + 1:end + 1]
    first_hit = 0
    bars_to_target = np.nan
    ambiguous = False
    for offset, (_, bar) in enumerate(window.iterrows(), 1):
        if direction == 1:
            target_hit = float(bar["High"]) >= target
            stop_hit = float(bar["Low"]) <= stop
        else:
            target_hit = float(bar["Low"]) <= target
            stop_hit = float(bar["High"]) >= stop
        if target_hit and stop_hit:
            first_hit = -1  # conservative: intrabar ordering is unknowable
            ambiguous = True
            break
        if target_hit:
            first_hit = 1
            bars_to_target = float(offset)
            break
        if stop_hit:
            first_hit = -1
            break

    highs = window["High"].to_numpy(dtype=float)
    lows = window["Low"].to_numpy(dtype=float)
    if direction == 1:
        favorable = max(0.0, (float(np.max(highs)) - entry) / atr_value)
        adverse = max(0.0, (entry - float(np.min(lows))) / atr_value)
    else:
        favorable = max(0.0, (entry - float(np.min(lows))) / atr_value)
        adverse = max(0.0, (float(np.max(highs)) - entry) / atr_value)
    fixed_return = direction * (float(df["Close"].iloc[end]) - entry) / atr_value
    after_first = window.iloc[1:] if len(window) > 1 else window.iloc[0:0]
    retested = bool(
        not after_first.empty
        and ((after_first["Low"] <= entry) & (after_first["High"] >= entry)).any()
    )
    return EventMeasurement(
        position=position,
        direction=direction,
        first_hit=first_hit,
        fixed_return_atr=float(fixed_return),
        favorable_atr=float(favorable),
        adverse_atr=float(adverse),
        bars_to_target=float(bars_to_target),
        retested=retested,
        ambiguous_bar=ambiguous,
    )


def _measure_events(
    df: pd.DataFrame,
    events: Iterable[TouchEvent],
    config: AnalysisConfig,
    start: int,
    end: int,
    side: int,
) -> list[EventMeasurement]:
    measured = []
    for event in events:
        if event.direction != side:
            continue
        if event.position < start or event.position + config.horizon >= end:
            continue
        outcome = measure_event(df, event.position, event.direction, config)
        if outcome is not None:
            measured.append(outcome)
    return measured


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * sqrt((p * (1.0 - p) / total) + z * z / (4.0 * total * total))
    return max(0.0, (center - margin) / denominator)


def summarize_measurements(measurements: Sequence[EventMeasurement]) -> dict[str, float]:
    n = len(measurements)
    if not n:
        return {
            "events": 0, "target_hits": 0, "stop_hits": 0, "timeouts": 0,
            "hit_rate": np.nan, "wilson_lower": 0.0, "median_fixed_atr": np.nan,
            "median_favorable_atr": np.nan, "median_adverse_atr": np.nan,
            "median_bars_to_target": np.nan, "retest_rate": np.nan,
            "ambiguous_rate": np.nan, "score": np.nan,
        }
    hits = sum(m.first_hit == 1 for m in measurements)
    stops = sum(m.first_hit == -1 for m in measurements)
    timeouts = n - hits - stops
    hit_rate = hits / n
    fixed = np.array([m.fixed_return_atr for m in measurements], dtype=float)
    favorable = np.array([m.favorable_atr for m in measurements], dtype=float)
    adverse = np.array([m.adverse_atr for m in measurements], dtype=float)
    times = np.array([m.bars_to_target for m in measurements], dtype=float)
    median_fixed = float(np.median(fixed))
    median_favorable = float(np.median(favorable))
    median_adverse = float(np.median(adverse))
    score = median_fixed + 0.50 * (2.0 * hit_rate - 1.0) - 0.15 * median_adverse
    return {
        "events": n,
        "target_hits": hits,
        "stop_hits": stops,
        "timeouts": timeouts,
        "hit_rate": hit_rate,
        "wilson_lower": wilson_lower(hits, n),
        "median_fixed_atr": median_fixed,
        "median_favorable_atr": median_favorable,
        "median_adverse_atr": median_adverse,
        "median_bars_to_target": float(np.nanmedian(times)) if np.isfinite(times).any() else np.nan,
        "retest_rate": float(np.mean([m.retested for m in measurements])),
        "ambiguous_rate": float(np.mean([m.ambiguous_bar for m in measurements])),
        "score": float(score),
    }


def _segment_bounds(length: int, config: AnalysisConfig) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    discovery_end = int(length * config.discovery_fraction)
    validation_end = int(length * (config.discovery_fraction + config.validation_fraction))
    return (0, discovery_end), (discovery_end, validation_end), (validation_end, length)


def _normal_survival(z_value: float) -> float:
    return 0.5 * (1.0 - erf(z_value / sqrt(2.0)))


def _empirical_pvalue(observed: float, controls: Sequence[float]) -> float:
    valid = np.asarray([x for x in controls if np.isfinite(x)], dtype=float)
    if not np.isfinite(observed) or len(valid) == 0:
        return np.nan
    return float((1 + np.sum(valid >= observed)) / (len(valid) + 1))


def _matched_random_scores(
    df: pd.DataFrame,
    ma_series: pd.Series,
    actual_events: Sequence[TouchEvent],
    side: int,
    start: int,
    end: int,
    config: AnalysisConfig,
    rng: np.random.Generator,
) -> list[float]:
    events = [e for e in actual_events if e.direction == side and start <= e.position < end - config.horizon]
    if not events:
        return []
    ma = ma_series.reindex(df.index).to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    adx_values = df["ADX"].to_numpy(dtype=float)
    vol_bins = df["VOL_BIN"].to_numpy(dtype=int)
    session_bins = df["SESSION_BIN"].to_numpy(dtype=int)
    excluded: set[int] = set()
    for event in actual_events:
        excluded.update(range(max(start, event.position - config.horizon), min(end, event.position + config.horizon + 1)))
    pools: dict[tuple[str, int, int, int], list[int]] = {}
    fallback: dict[tuple[str, int], list[int]] = {}
    all_pool: list[int] = []
    for i in range(max(start + 1, config.atr_period), end - config.horizon):
        if i in excluded or not (np.isfinite(ma[i]) and np.isfinite(df["ATR"].iloc[i])):
            continue
        prev_side = 1 if close[i - 1] > ma[i - 1] else -1
        if prev_side != side:
            continue
        lookback = max(start, i - config.horizon)
        approach_strength = max(0.0, -side * (close[i - 1] - close[lookback]) / float(df["ATR"].iloc[i]))
        if approach_strength <= 0:
            continue
        approach_bin = int(min(3, np.floor(approach_strength)))
        regime = "trend" if np.isfinite(adx_values[i]) and adx_values[i] >= config.adx_threshold else "range"
        key = (regime, int(vol_bins[i]), int(session_bins[i]), approach_bin)
        pools.setdefault(key, []).append(i)
        fallback.setdefault((regime, approach_bin), []).append(i)
        all_pool.append(i)
    if len(all_pool) < max(3, len(events)):
        return []

    scores: list[float] = []
    for _ in range(config.null_iterations):
        sample_positions: list[int] = []
        for event in events:
            key = (event.regime, event.volatility_bin, event.session_bin, event.approach_bin)
            pool = pools.get(key) or fallback.get((event.regime, event.approach_bin))
            if not pool:
                pool = all_pool
            sample_positions.append(int(rng.choice(pool)))
        outcomes = [measure_event(df, pos, side, config) for pos in sample_positions]
        metrics = summarize_measurements([x for x in outcomes if x is not None])
        if np.isfinite(metrics["score"]):
            scores.append(float(metrics["score"]))
    return scores


def _shift_control_scores(
    df: pd.DataFrame,
    ma_series: pd.Series,
    side: int,
    start: int,
    end: int,
    period: int,
    config: AnalysisConfig,
) -> list[float]:
    length = end - start
    if length < max(80, period * 2):
        return []
    min_lag = max(config.horizon * 2, min(period, max(5, length // 10)))
    max_lag = max(min_lag + 1, min(length // 2, min_lag * 6))
    lags = sorted(set(int(x) for x in np.linspace(min_lag, max_lag, 7)))
    scores = []
    for lag in lags:
        shifted = ma_series.shift(lag)
        events = detect_independent_touches(df, shifted, config)
        metrics = summarize_measurements(_measure_events(df, events, config, start, end, side))
        if np.isfinite(metrics["score"]):
            scores.append(float(metrics["score"]))
    return scores


def _piecewise_horizontal_level(close: pd.Series, block: int, offset: int) -> pd.Series:
    values = np.full(len(close), np.nan)
    first = max(0, offset)
    for start in range(first, len(close), block):
        previous_start = start - block
        if previous_start < 0:
            continue
        previous = close.iloc[previous_start:start]
        if previous.notna().any():
            values[start:min(start + block, len(close))] = float(previous.median())
    return pd.Series(values, index=close.index)


def _horizontal_control_scores(
    df: pd.DataFrame,
    side: int,
    start: int,
    end: int,
    period: int,
    config: AnalysisConfig,
) -> list[float]:
    block = max(config.horizon * 3, min(max(period, 20), max(20, (end - start) // 4)))
    scores = []
    for offset in sorted(set((0, block // 3, (2 * block) // 3))):
        level = _piecewise_horizontal_level(df["Close"], block, offset)
        events = detect_independent_touches(df, level, config)
        metrics = summarize_measurements(_measure_events(df, events, config, start, end, side))
        if np.isfinite(metrics["score"]):
            scores.append(float(metrics["score"]))
    return scores


def evaluate_candidate(
    df: pd.DataFrame,
    ma_series: pd.Series,
    ma_type: str,
    period: int,
    side: int,
    config: AnalysisConfig,
    seed: int,
) -> dict[str, object]:
    events = detect_independent_touches(df, ma_series, config)
    discovery, validation, holdout = _segment_bounds(len(df), config)
    d_metrics = summarize_measurements(_measure_events(df, events, config, *discovery, side))
    v_metrics = summarize_measurements(_measure_events(df, events, config, *validation, side))
    h_metrics = summarize_measurements(_measure_events(df, events, config, *holdout, side))
    result: dict[str, object] = {
        "ma_type": ma_type,
        "period": int(period),
        "side": "support" if side == 1 else "resistance",
    }
    for prefix, metrics in (("discovery", d_metrics), ("validation", v_metrics), ("holdout", h_metrics)):
        for key, value in metrics.items():
            result[f"{prefix}_{key}"] = value

    p_random = p_shift = p_horizontal = np.nan
    random_threshold = np.nan
    shift_scores: list[float] = []
    horizontal_scores: list[float] = []
    if d_metrics["events"] >= config.min_events and np.isfinite(d_metrics["score"]):
        rng = np.random.default_rng(seed)
        random_scores = _matched_random_scores(
            df, ma_series, events, side, *discovery, config, rng
        )
        p_random = _empirical_pvalue(float(d_metrics["score"]), random_scores)
        if random_scores:
            random_threshold = float(np.quantile(random_scores, config.null_quantile))
        # Expensive controls are only useful for candidates that are not clearly noise.
        if config.use_shift_control and (not np.isfinite(p_random) or p_random <= 0.25):
            shift_scores = _shift_control_scores(
                df, ma_series, side, *discovery, period, config
            )
            p_shift = _empirical_pvalue(float(d_metrics["score"]), shift_scores)
        if config.use_horizontal_control and (not np.isfinite(p_random) or p_random <= 0.25):
            horizontal_scores = _horizontal_control_scores(
                df, side, *discovery, period, config
            )
            p_horizontal = _empirical_pvalue(float(d_metrics["score"]), horizontal_scores)

    shift_threshold = float(np.quantile(shift_scores, config.null_quantile)) if shift_scores else np.nan
    horizontal_threshold = float(np.quantile(horizontal_scores, config.null_quantile)) if horizontal_scores else np.nan
    shift_pass = (not config.use_shift_control) or (
        len(shift_scores) > 0 and float(d_metrics["score"]) > shift_threshold
    )
    horizontal_pass = (not config.use_horizontal_control) or (
        len(horizontal_scores) > 0 and float(d_metrics["score"]) > horizontal_threshold
    )
    secondary_controls_pass = bool(shift_pass and horizontal_pass)
    # The small secondary-control ensembles are gates, not p-value estimators;
    # taking their coarse empirical p-values as a maximum would make FDR
    # mathematically impossible. The matched null supplies the calibrated p.
    combined_p = p_random if secondary_controls_pass and np.isfinite(p_random) else (
        1.0 if np.isfinite(p_random) else np.nan
    )
    result.update({
        "p_random": p_random,
        "p_shift": p_shift,
        "p_horizontal": p_horizontal,
        "p_value": combined_p,
        "shift_control_pass": bool(shift_pass),
        "horizontal_control_pass": bool(horizontal_pass),
        "secondary_controls_pass": secondary_controls_pass,
        "shift_score_threshold": shift_threshold,
        "horizontal_score_threshold": horizontal_threshold,
        "random_score_threshold": random_threshold,
        "validation_pass": bool(
            v_metrics["events"] >= config.min_segment_events
            and np.isfinite(v_metrics["score"])
            and v_metrics["score"] > 0
            and v_metrics["median_fixed_atr"] > 0
        ),
        "holdout_pass": bool(
            h_metrics["events"] >= config.min_segment_events
            and np.isfinite(h_metrics["score"])
            and h_metrics["score"] > 0
            and h_metrics["median_fixed_atr"] > 0
        ),
    })
    return result


def adjust_fdr(p_values: Sequence[float], method: str = "by") -> np.ndarray:
    """Benjamini-Hochberg or dependence-robust Benjamini-Yekutieli q-values."""

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if not len(valid_positions):
        return adjusted
    valid = values[valid_positions]
    order = np.argsort(valid)
    ranked = valid[order]
    m = len(ranked)
    dependency = sum(1.0 / i for i in range(1, m + 1)) if method == "by" else 1.0
    raw = ranked * m * dependency / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    local = np.empty(m)
    local[order] = monotone
    adjusted[valid_positions] = local
    return adjusted


def analyze_ma_universe(
    frame: pd.DataFrame,
    ticker: str,
    timeframe: str,
    config: AnalysisConfig | None = None,
    ma_types: Sequence[str] = MA_TYPES,
    periods: Sequence[int] = DEFAULT_PERIODS,
) -> pd.DataFrame:
    """Evaluate every MA/side, correct the family, and attach current levels."""

    cfg = config or TIMEFRAME_CONFIGS.get(timeframe, AnalysisConfig())
    df = prepare_frame(frame, cfg)
    rows: list[dict[str, object]] = []
    current_price = float(df["Close"].iloc[-1])
    current_atr = float(df["ATR"].iloc[-1])
    for type_index, ma_type in enumerate(ma_types):
        for period in periods:
            if period >= len(df) - cfg.horizon - 5:
                continue
            if ma_type.upper() == "HMA" and period < 13:
                continue
            ma_series = compute_ma(ma_type, df["Close"], df["Volume"], int(period))
            current_ma = float(ma_series.iloc[-1]) if np.isfinite(ma_series.iloc[-1]) else np.nan
            if not np.isfinite(current_ma):
                continue
            current_side = "support" if current_ma <= current_price else "resistance"
            for side in (1, -1):
                seed = cfg.random_seed + type_index * 100_003 + int(period) * 101 + (1 if side == 1 else 2)
                row = evaluate_candidate(df, ma_series, ma_type, int(period), side, cfg, seed)
                row.update({
                    "ticker": ticker.upper(),
                    "timeframe": timeframe,
                    "current_price": current_price,
                    "current_ma": current_ma,
                    "distance_pct": 100.0 * (current_ma - current_price) / current_price,
                    "distance_atr": (current_ma - current_price) / current_atr if current_atr > 0 else np.nan,
                    "current_side": current_side,
                    "active_side": row["side"] == current_side,
                })
                rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Only the currently relevant side of each MA belongs to the live decision
    # family. The opposite-side history is retained for diagnostics but is not
    # silently counted as another present-day hypothesis.
    result["q_value"] = np.nan
    active_mask = result["active_side"] & result["p_value"].notna()
    result.loc[active_mask, "q_value"] = adjust_fdr(
        result.loc[active_mask, "p_value"].to_numpy(dtype=float), cfg.fdr_method
    )
    result["discovery_pass"] = (
        (result["discovery_events"] >= cfg.min_events)
        & (result["discovery_score"] > 0)
        & (result["discovery_wilson_lower"] >= cfg.min_wilson)
        & (result["q_value"] <= cfg.fdr_q)
    )
    result["certified"] = (
        result["discovery_pass"] & result["validation_pass"] & result["holdout_pass"]
    )
    result["actionable"] = result["certified"] & (
        result["distance_atr"].abs() <= cfg.max_actionable_distance_atr
    )
    result["status"] = "unverified_candidate"
    result.loc[~result["active_side"], "status"] = "inactive_side_history"
    result.loc[result["discovery_events"] < cfg.min_events, "status"] = "insufficient_history"
    result.loc[result["discovery_pass"] & ~result["validation_pass"], "status"] = "validation_failed"
    result.loc[
        result["discovery_pass"] & result["validation_pass"] & ~result["holdout_pass"], "status"
    ] = "holdout_failed"
    result.loc[result["certified"], "status"] = "certified"
    result.loc[result["certified"] & ~result["actionable"], "status"] = "certified_but_far"
    quality = (
        result["discovery_score"].fillna(-10)
        + result["validation_score"].fillna(-10)
        + result["holdout_score"].fillna(-10)
        - 0.05 * result["distance_atr"].abs().fillna(100)
    )
    result["rank_score"] = (
        100.0 * result["certified"].astype(float)
        + 20.0 * result["discovery_pass"].astype(float)
        + quality
    )
    return result


def select_panel_levels(results: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return both sides; unverified rows remain explicitly labelled candidates."""

    if results.empty:
        return results.copy()
    active = results[results["active_side"]].copy()
    selected = []
    for side in ("support", "resistance"):
        group = active[active["side"] == side].copy()
        group = group.sort_values(
            ["certified", "actionable", "rank_score", "distance_atr"],
            ascending=[False, False, False, side == "support"],
        )
        # If evidence is weak, proximity is more useful than a noisy historical score.
        certified = group[group["certified"]]
        candidates = group[~group["certified"]].sort_values("distance_atr", key=lambda s: s.abs())
        chosen = pd.concat([certified, candidates]).drop_duplicates(["ma_type", "period"]).head(top_n)
        selected.append(chosen)
    out = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if not out.empty:
        out["evidence_label"] = np.where(
            out["certified"], "CERTIFIED", "CANDIDATE_ONLY"
        )
    return out


def build_confluence(
    panel_rows: pd.DataFrame,
    tolerance_pct: float = 0.75,
    min_timeframes: int = 2,
) -> pd.DataFrame:
    """Cluster nearby levels as context, never as independent statistical proof."""

    if panel_rows.empty:
        return pd.DataFrame()
    clusters: list[dict[str, object]] = []
    for (ticker, side), group in panel_rows.groupby(["ticker", "side"]):
        levels = group.sort_values("current_ma").to_dict("records")
        current: list[dict[str, object]] = []
        for row in levels:
            if not current:
                current = [row]
                continue
            center = float(np.median([float(x["current_ma"]) for x in current]))
            distance = abs(float(row["current_ma"]) - center) / center * 100.0
            if distance <= tolerance_pct:
                current.append(row)
            else:
                clusters.append(_summarize_cluster(ticker, side, current, min_timeframes))
                current = [row]
        if current:
            clusters.append(_summarize_cluster(ticker, side, current, min_timeframes))
    result = pd.DataFrame(clusters)
    if result.empty:
        return result
    return result.sort_values(
        ["qualified", "timeframe_count", "certified_count", "context_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _summarize_cluster(
    ticker: str,
    side: str,
    rows: Sequence[dict[str, object]],
    min_timeframes: int,
) -> dict[str, object]:
    timeframes = sorted({str(x["timeframe"]) for x in rows})
    certified_count = sum(bool(x["certified"]) for x in rows)
    levels = [float(x["current_ma"]) for x in rows]
    labels = [f"{x['timeframe']}:{x['ma_type']}{int(x['period'])}" for x in rows]
    context_score = len(timeframes) + 0.5 * certified_count
    return {
        "ticker": ticker,
        "side": side,
        "level": float(np.median(levels)),
        "timeframe_count": len(timeframes),
        "certified_count": certified_count,
        "timeframes": ",".join(timeframes),
        "members": ", ".join(labels),
        "context_score": context_score,
        "qualified": len(timeframes) >= min_timeframes and certified_count >= 1,
        "interpretation": "context_cluster_not_independent_confirmation",
    }


def format_panel(panel: pd.DataFrame) -> str:
    if panel.empty:
        return "No levels could be calculated."
    lines = []
    for (ticker, timeframe), group in panel.groupby(["ticker", "timeframe"], sort=False):
        price = float(group["current_price"].iloc[0])
        lines.append(f"\n{ticker} | {timeframe} | price={price:.2f}")
        for side in ("support", "resistance"):
            lines.append(f"  {side.upper()}")
            subset = group[group["side"] == side]
            if subset.empty:
                lines.append("    none")
                continue
            for _, row in subset.iterrows():
                marker = "✓" if bool(row["certified"]) else "·"
                q_text = f"{float(row['q_value']):.3f}" if np.isfinite(row["q_value"]) else "n/a"
                lines.append(
                    f"    {marker} {row['ma_type']}{int(row['period'])} level={row['current_ma']:.2f} "
                    f"dist={row['distance_pct']:+.2f}%/{row['distance_atr']:+.2f}ATR "
                    f"n={int(row['discovery_events'])} q={q_text} status={row['status']}"
                )
    return "\n".join(lines).lstrip()

