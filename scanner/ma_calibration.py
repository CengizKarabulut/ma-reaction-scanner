#!/usr/bin/env python3
"""Diagnostic calibration summaries for MA level scores.

This module intentionally does not change the score formula.  It only groups
already-produced scan rows into score buckets so the project can inspect whether
higher ``level_score`` buckets also show better observed behaviour.

The first report is marked ``in_sample_aggregate`` because it summarizes the
same aggregate rows that produced the scores.  It is a diagnostic dashboard, not
out-of-sample evidence.  Walk-forward validation can build on the same output
shape later without changing downstream CSV consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SCORE_BUCKETS: tuple[float, ...] = (0.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, np.inf)
_CALIBRATION_SCOPE = "in_sample_aggregate"


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for diagnostic score-bucket summaries."""

    score_buckets: tuple[float, ...] = DEFAULT_SCORE_BUCKETS
    score_column: str = "level_score"

    def __post_init__(self) -> None:
        if len(self.score_buckets) < 2:
            raise ValueError("En az iki skor bucket siniri gerekir")
        if any(
            float(left) >= float(right)
            for left, right in zip(self.score_buckets, self.score_buckets[1:])
        ):
            raise ValueError("Skor bucket sinirlari artan sirada olmalidir")


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _bucket_labels(edges: Iterable[float]) -> list[str]:
    values = list(edges)
    labels: list[str] = []
    for left, right in zip(values, values[1:]):
        left_text = f"{left:g}"
        right_text = "+" if np.isinf(right) else f"{right:g}"
        labels.append(f"{left_text}-{right_text}" if not np.isinf(right) else f"{left_text}+")
    return labels


def score_bucket_summary(
    frame: pd.DataFrame,
    config: CalibrationConfig | None = None,
) -> pd.DataFrame:
    """Summarize observed level behaviour by score bucket and side.

    The result is deliberately explicit about scope.  Rows are aggregate MA
    combinations, not individual future observations, so this report should be
    read as a calibration diagnostic until a walk-forward evaluator is added.
    """

    config = config or CalibrationConfig()
    columns = [
        "analysis_scope",
        "side",
        "score_bucket",
        "row_count",
        "median_level_score",
        "mean_level_score",
        "median_level_touches",
        "median_hold_rate_pct",
        "median_break_rate_pct",
        "median_bounce_atr",
        "median_bounce_p75_atr",
        "median_reaction_1atr_rate_pct",
        "median_reaction_2atr_rate_pct",
        "median_penetration_atr",
        "median_penetration_p75_atr",
        "row_share_median_bounce_ge_1atr_pct",
        "row_share_median_bounce_ge_2atr_pct",
    ]
    if frame is None or frame.empty or config.score_column not in getattr(frame, "columns", []):
        return pd.DataFrame(columns=columns)

    prepared = frame.copy()
    scores = _numeric(prepared, config.score_column)
    labels = _bucket_labels(config.score_buckets)
    prepared["_score_bucket"] = pd.cut(
        scores,
        bins=list(config.score_buckets),
        labels=labels,
        right=False,
        include_lowest=True,
    )
    prepared = prepared[prepared["_score_bucket"].notna()].copy()
    if prepared.empty:
        return pd.DataFrame(columns=columns)

    if "side" not in prepared.columns:
        prepared["side"] = "ALL"

    records: list[dict[str, object]] = []
    for (side, bucket), group in prepared.groupby(
        ["side", "_score_bucket"], sort=True, dropna=False, observed=True
    ):
        bounces = _numeric(group, "median_bounce_atr")
        records.append(
            {
                "analysis_scope": _CALIBRATION_SCOPE,
                "side": str(side),
                "score_bucket": str(bucket),
                "row_count": int(len(group)),
                "median_level_score": float(_numeric(group, config.score_column).median()),
                "mean_level_score": float(_numeric(group, config.score_column).mean()),
                "median_level_touches": float(_numeric(group, "level_touches").median()),
                "median_hold_rate_pct": float(_numeric(group, "hold_rate_pct").median()),
                "median_break_rate_pct": float(_numeric(group, "break_rate_pct").median()),
                "median_bounce_atr": float(bounces.median()),
                "median_bounce_p75_atr": float(_numeric(group, "bounce_p75_atr").median()),
                "median_reaction_1atr_rate_pct": float(_numeric(group, "reaction_1atr_rate_pct").median()),
                "median_reaction_2atr_rate_pct": float(_numeric(group, "reaction_2atr_rate_pct").median()),
                "median_penetration_atr": float(_numeric(group, "median_penetration_atr").median()),
                "median_penetration_p75_atr": float(_numeric(group, "penetration_p75_atr").median()),
                "row_share_median_bounce_ge_1atr_pct": float(100.0 * (bounces >= 1.0).mean()),
                "row_share_median_bounce_ge_2atr_pct": float(100.0 * (bounces >= 2.0).mean()),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)
