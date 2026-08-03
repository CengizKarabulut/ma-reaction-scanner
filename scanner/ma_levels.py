#!/usr/bin/env python3
"""Simulation-free moving-average level quality metrics.

The trade layer in :mod:`scanner.ma_engine` answers "would this have been
profitable?".  That question needs stop, cost and trailing assumptions, and it
compares touch trades against a random baseline whose risk unit is structurally
different: an MA touch is stopped a few tenths of an ATR away while the baseline
is stopped more than a full ATR away.  The two legs therefore face different
R scaling and different cost burdens, so ``edge_r`` is not a clean measurement
of "does this average matter".

This module answers a narrower, far more robust question:

    *Does price actually react to this moving average, and does the average
    behave like a boundary rather than the middle of the noise?*

Every metric here is a direct observation of the price series.  No stop, no
round-trip cost, no trailing rule, no baseline, no position sizing.  That makes
the output insensitive to the assumptions that dominate the trade layer, which
is what you want when a moving average is used as a support/resistance
reference rather than as an entry signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .ma_engine import Touch
except ImportError:  # direct script execution
    from ma_engine import Touch


@dataclass(frozen=True)
class LevelConfig:
    """Thresholds for the observation-only level metrics.

    ``reaction_bars`` and ``hold_bars`` are deliberately separate: how far price
    travels away from the average (reaction) and whether it stayed on the right
    side (hold) are different questions, usually answered over different
    horizons.
    """

    reaction_bars: int = 10
    hold_bars: int = 5
    break_atr: float = 0.50
    bounce_cap_atr: float = 2.0
    cross_cap_per_100: float = 4.0
    evidence_target_touches: int = 20
    neighbor_ratio: float = 0.25

    def __post_init__(self) -> None:
        positive_integers = (
            self.reaction_bars,
            self.hold_bars,
            self.evidence_target_touches,
        )
        if any(int(value) < 1 for value in positive_integers):
            raise ValueError("Tepki, tutma ve kanit esikleri pozitif olmalidir")
        non_negative = (
            self.break_atr,
            self.bounce_cap_atr,
            self.cross_cap_per_100,
            self.neighbor_ratio,
        )
        if any(float(value) < 0 for value in non_negative):
            raise ValueError("ATR ve oran esikleri negatif olamaz")
        if self.bounce_cap_atr <= 0 or self.cross_cap_per_100 <= 0:
            raise ValueError("Sicrama ve kesisim tavanlari sifirdan buyuk olmalidir")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TouchOutcome:
    """What happened after a single independent touch, in ATR units."""

    position: int
    bounce_atr: float
    penetration_atr: float
    held: bool


def _finite_window(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    window = values[start:stop]
    return window[np.isfinite(window)]


def touch_outcomes(
    df: pd.DataFrame,
    touches: Sequence[Touch],
    side: int,
    config: LevelConfig,
) -> list[TouchOutcome]:
    """Measure the reaction after each touch without simulating a trade.

    For a support touch the reaction is how far *above* the average price
    travelled, and the level "held" when no close fell more than ``break_atr``
    below the average within ``hold_bars``.  Resistance is the mirror image.
    Both are expressed in ATR at the touch bar, so the numbers are comparable
    across symbols and across price levels.
    """

    if side not in {1, -1}:
        raise ValueError("side +1 veya -1 olmalidir")
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    total = len(df)
    outcomes: list[TouchOutcome] = []
    for touch in touches:
        atr = float(touch.atr)
        if not np.isfinite(atr) or atr <= 0:
            continue
        start = touch.position + 1
        if start >= total:
            continue
        reaction_stop = min(total, start + config.reaction_bars)
        hold_stop = min(total, start + config.hold_bars)

        favourable = _finite_window(high if side == 1 else low, start, reaction_stop)
        if favourable.size == 0:
            continue
        extreme = float(favourable.max() if side == 1 else favourable.min())
        bounce_atr = side * (extreme - touch.ma_value) / atr

        adverse = _finite_window(low if side == 1 else high, start, hold_stop)
        penetration_atr = 0.0
        if adverse.size:
            worst = float(adverse.min() if side == 1 else adverse.max())
            penetration_atr = max(0.0, side * (touch.ma_value - worst) / atr)

        closes = _finite_window(close, start, hold_stop)
        if closes.size:
            breach = side * (touch.ma_value - closes) / atr
            held = bool(float(np.max(breach)) <= config.break_atr)
        else:
            held = True

        outcomes.append(
            TouchOutcome(
                position=touch.position,
                bounce_atr=float(bounce_atr),
                penetration_atr=float(penetration_atr),
                held=held,
            )
        )
    return outcomes


def summarize_outcomes(
    outcomes: Sequence[TouchOutcome],
    *,
    valid_bars: int,
    cross_count: int,
    config: LevelConfig,
) -> dict[str, float]:
    """Aggregate touch outcomes into the observation-only level metrics.

    ``touch_density_per_100`` exists because a raw touch count structurally
    favours short averages: an EMA20 hugs price and is simply touched more
    often than an EMA200.  Normalising by the number of bars on which the
    average is defined makes short and long periods comparable.
    """

    count = len(outcomes)
    bars = max(int(valid_bars), 1)
    touch_density = 100.0 * count / bars
    cross_density = 100.0 * max(int(cross_count), 0) / bars
    if count == 0:
        return {
            "level_touches": 0,
            "touch_density_per_100": float(touch_density),
            "cross_per_100": float(cross_density),
            "median_bounce_atr": float("nan"),
            "median_penetration_atr": float("nan"),
            "hold_rate_pct": float("nan"),
            "break_rate_pct": float("nan"),
        }
    bounces = np.asarray([item.bounce_atr for item in outcomes], dtype=float)
    penetrations = np.asarray([item.penetration_atr for item in outcomes], dtype=float)
    hold_rate = 100.0 * float(np.mean([item.held for item in outcomes]))
    return {
        "level_touches": count,
        "touch_density_per_100": float(touch_density),
        "cross_per_100": float(cross_density),
        "median_bounce_atr": float(np.median(bounces)),
        "median_penetration_atr": float(np.median(penetrations)),
        "hold_rate_pct": hold_rate,
        "break_rate_pct": 100.0 - hold_rate,
    }


def level_score(metrics: dict[str, float], config: LevelConfig) -> float:
    """Return a transparent 0-100 level-quality score.

    Weights, and why:

    * ``30`` **hold rate** - the direct answer to "is this a boundary".
    * ``25`` **median bounce** - an average nobody reacts to is not a level.
    * ``25`` **evidence** - how many independent tests there were.
    * ``20`` **cleanliness** - price slicing through the average again and
      again means the average sits inside the noise, not at its edge.

    None of these can be improved by changing a stop, a cost assumption or a
    holding period, which is the whole point.
    """

    def bounded(value: object, low: float, high: float) -> float:
        number = float(value) if value is not None else float("nan")
        if not np.isfinite(number):
            return 0.0
        return float(np.clip(number, low, high))

    touches = int(metrics.get("level_touches", 0) or 0)
    if touches <= 0:
        # An average price never came back to is not a clean level, it is an
        # untested one.  Awarding the cleanliness points here would let
        # never-touched averages outrank genuinely tested ones.
        return 0.0
    evidence = min(touches / max(config.evidence_target_touches, 1), 1.0) * 25.0
    hold = bounded(metrics.get("hold_rate_pct"), 0.0, 100.0) / 100.0 * 30.0
    bounce = (
        bounded(metrics.get("median_bounce_atr"), 0.0, config.bounce_cap_atr)
        / config.bounce_cap_atr
        * 25.0
    )
    cross = bounded(metrics.get("cross_per_100"), 0.0, config.cross_cap_per_100)
    cleanliness = (1.0 - cross / config.cross_cap_per_100) * 20.0
    return round(evidence + hold + bounce + cleanliness, 2)


def level_class(metrics: dict[str, float], score: float, config: LevelConfig) -> str:
    """Coarse, human-readable bucket for the level score."""

    touches = int(metrics.get("level_touches", 0) or 0)
    if touches < max(3, config.evidence_target_touches // 4):
        return "Yetersiz temas"
    if score >= 70.0:
        return "Guclu seviye"
    if score >= 55.0:
        return "Seviye"
    if score >= 40.0:
        return "Zayif seviye"
    return "Seviye degil"


# ---------------------------------------------------------------------------
# Cross-row post-processing: plateau robustness and relative adherence.
# ---------------------------------------------------------------------------


_GROUP_KEYS = ("asset_class", "symbol", "timeframe", "ma_type", "side")


def _present_keys(frame: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key in frame.columns]


def add_plateau_scores(
    frame: pd.DataFrame,
    config: LevelConfig,
    *,
    score_column: str = "level_score",
) -> pd.DataFrame:
    """Flag level scores that are isolated peaks rather than robust plateaus.

    A moving average that genuinely matters is not fussy about two or three
    periods either way: EMA47, EMA50 and EMA53 should tell a similar story.  If
    one period scores far above its immediate neighbours, the most likely
    explanation is that the scan tried many combinations and this one got lucky
    - not that the market cares about that exact number.

    ``plateau_ratio`` is the mean neighbour score divided by the row's own
    score, clipped at 1.0.  Values near 1 mean the result survives small
    parameter changes; low values mean it does not.  ``plateau_neighbors``
    reports how many neighbours were available, because the test is only
    meaningful when the scanned period grid actually contains close periods.
    """

    if frame is None or len(frame) == 0 or score_column not in getattr(frame, "columns", []):
        result = frame.copy() if frame is not None else pd.DataFrame()
        result["plateau_ratio"] = pd.Series(dtype=float)
        result["plateau_neighbors"] = pd.Series(dtype=int)
        return result

    result = frame.copy()
    result["plateau_ratio"] = np.nan
    result["plateau_neighbors"] = 0
    keys = _present_keys(result, _GROUP_KEYS)
    if not keys or "period" not in result.columns:
        return result

    periods = pd.to_numeric(result["period"], errors="coerce")
    scores = pd.to_numeric(result[score_column], errors="coerce")
    for _, group in result.groupby(list(keys), sort=False, dropna=False):
        indexes = list(group.index)
        group_periods = periods.loc[indexes].to_numpy(dtype=float)
        group_scores = scores.loc[indexes].to_numpy(dtype=float)
        for offset, index in enumerate(indexes):
            own_period = group_periods[offset]
            own_score = group_scores[offset]
            if not np.isfinite(own_period) or own_period <= 0:
                continue
            span = config.neighbor_ratio * own_period
            distance = np.abs(group_periods - own_period)
            mask = (distance <= span) & (distance > 0) & np.isfinite(group_scores)
            neighbours = group_scores[mask]
            result.at[index, "plateau_neighbors"] = int(neighbours.size)
            if neighbours.size == 0 or not np.isfinite(own_score):
                continue
            if own_score <= 0:
                result.at[index, "plateau_ratio"] = 1.0
                continue
            result.at[index, "plateau_ratio"] = float(
                np.clip(float(np.mean(neighbours)) / own_score, 0.0, 1.0)
            )
    return result


def add_adherence_excess(
    frame: pd.DataFrame,
    *,
    column: str = "side_adherence_pct",
) -> pd.DataFrame:
    """Express side adherence relative to the symbol's own average.

    Raw adherence is misleading on a series that simply went up: in a stock
    that trended for five years *every* moving average shows a high share of
    time spent above it.  That is a fact about the stock, not about the
    average.  The excess over the symbol's median adherence at the same
    timeframe and side isolates the part actually attributable to the chosen
    average - which matters especially on nominal TRY series, where inflation
    alone produces a structural uptrend.
    """

    result = frame.copy() if frame is not None else pd.DataFrame()
    if len(result) == 0 or column not in getattr(result, "columns", []):
        result["adherence_excess_pct"] = pd.Series(dtype=float)
        return result
    keys = _present_keys(result, ("asset_class", "symbol", "timeframe", "side"))
    values = pd.to_numeric(result[column], errors="coerce")
    if not keys:
        result["adherence_excess_pct"] = values - values.median()
        return result
    baseline = values.groupby([result[key] for key in keys]).transform("median")
    result["adherence_excess_pct"] = values - baseline
    return result


def finalize_level_frame(frame: pd.DataFrame, config: LevelConfig) -> pd.DataFrame:
    """Apply every cross-row adjustment in the order they depend on."""

    enriched = add_adherence_excess(frame)
    return add_plateau_scores(enriched, config)
