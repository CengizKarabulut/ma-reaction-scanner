#!/usr/bin/env python3
"""One-instrument-one-row summaries for guarded MA research outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


_PRICE_TIMEFRAME_PRIORITY = {
    "1d": 0,
    "4h": 1,
    "1h": 2,
    "30m": 3,
    "15m": 4,
    "5m": 5,
    "1wk": 6,
    "1mo": 7,
}


def _is_true(value: object) -> bool:
    """Return True only for explicit boolean true values."""

    return isinstance(value, (bool, np.bool_)) and bool(value)


def _true_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a boolean mask without treating NaN/object values as true."""

    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].eq(True).fillna(False).astype(bool)


def _best_level(group: pd.DataFrame, side: str) -> pd.Series | None:
    rows = group[_true_mask(group, "active_side") & (group["side"] == side)].copy()
    if rows.empty:
        return None
    rows["absolute_distance_atr"] = rows["distance_atr"].abs()
    if "sr_strength_score" not in rows:
        rows["sr_strength_score"] = np.nan
    certified = rows[_true_mask(rows, "certified")].copy()
    if not certified.empty:
        return certified.sort_values(
            ["actionable", "sr_strength_score", "rank_score", "q_value"],
            ascending=[False, False, False, True],
            na_position="last",
        ).iloc[0]
    return rows.sort_values(
        ["sr_strength_score", "absolute_distance_atr", "rank_score"],
        ascending=[False, True, False],
        na_position="last",
    ).iloc[0]


def _price_row(group: pd.DataFrame) -> pd.Series:
    ranked = group.copy()
    ranked["timeframe_priority"] = (
        ranked["timeframe"].map(_PRICE_TIMEFRAME_PRIORITY).fillna(99)
    )
    return ranked.sort_values("timeframe_priority").iloc[0]


def _level_fields(prefix: str, row: pd.Series | None) -> dict[str, object]:
    if row is None:
        return {
            f"{prefix}_timeframe": "",
            f"{prefix}_ma": "",
            f"{prefix}_period": np.nan,
            f"{prefix}_level": np.nan,
            f"{prefix}_distance_pct": np.nan,
            f"{prefix}_distance_atr": np.nan,
            f"{prefix}_q_value": np.nan,
            f"{prefix}_status": "none",
            f"{prefix}_discovery_events": 0,
            f"{prefix}_discovery_pass": False,
            f"{prefix}_validation_pass": False,
            f"{prefix}_holdout_pass": False,
            f"{prefix}_low_confidence": False,
            f"{prefix}_sr_strength_score": np.nan,
            f"{prefix}_holdout_median_fixed_atr_ci_low": np.nan,
            f"{prefix}_holdout_median_fixed_atr_ci_high": np.nan,
            f"{prefix}_holdout_net_median_fixed_atr": np.nan,
            f"{prefix}_evidence": "NONE",
        }
    q_value = row.get("q_value", np.nan)
    low_confidence = _is_true(row.get("low_confidence", False))
    evidence = "CERTIFIED" if bool(row["certified"]) else (
        "LOW_CONFIDENCE" if low_confidence else "CANDIDATE_ONLY"
    )
    return {
        f"{prefix}_timeframe": str(row["timeframe"]),
        f"{prefix}_ma": str(row["ma_type"]),
        f"{prefix}_period": int(row["period"]),
        f"{prefix}_level": float(row["current_ma"]),
        f"{prefix}_distance_pct": float(row.get("distance_pct", np.nan)),
        f"{prefix}_distance_atr": float(row["distance_atr"]),
        f"{prefix}_q_value": float(q_value) if np.isfinite(q_value) else np.nan,
        f"{prefix}_status": str(row.get("status", "unverified_candidate")),
        f"{prefix}_discovery_events": int(row.get("discovery_events", 0) or 0),
        f"{prefix}_discovery_pass": bool(row.get("discovery_pass", False)),
        f"{prefix}_validation_pass": bool(row.get("validation_pass", False)),
        f"{prefix}_holdout_pass": bool(row.get("holdout_pass", False)),
        f"{prefix}_low_confidence": low_confidence,
        f"{prefix}_sr_strength_score": float(row.get("sr_strength_score", np.nan)),
        f"{prefix}_holdout_median_fixed_atr_ci_low": float(
            row.get("holdout_median_fixed_atr_ci_low", np.nan)
        ),
        f"{prefix}_holdout_median_fixed_atr_ci_high": float(
            row.get("holdout_median_fixed_atr_ci_high", np.nan)
        ),
        f"{prefix}_holdout_net_median_fixed_atr": float(
            row.get("holdout_net_median_fixed_atr", np.nan)
        ),
        f"{prefix}_evidence": evidence,
    }


def _strongest_fields(active: pd.DataFrame) -> dict[str, object]:
    empty = {
        "strongest_timeframe": "",
        "strongest_side": "",
        "strongest_ma": "",
        "strongest_period": np.nan,
        "strongest_level": np.nan,
        "strongest_sr_strength_score": np.nan,
        "strongest_evidence": "NONE",
    }
    if active.empty or "sr_strength_score" not in active:
        return empty
    rows = active.copy()
    rows["sr_strength_score"] = pd.to_numeric(
        rows["sr_strength_score"], errors="coerce"
    )
    rows = rows[np.isfinite(rows["sr_strength_score"])]
    if rows.empty:
        return empty
    row = rows.sort_values("sr_strength_score", ascending=False).iloc[0]
    low_confidence = _is_true(row.get("low_confidence", False))
    evidence = "CERTIFIED" if _is_true(row.get("certified", False)) else (
        "LOW_CONFIDENCE" if low_confidence else "CANDIDATE_ONLY"
    )
    return {
        "strongest_timeframe": str(row["timeframe"]),
        "strongest_side": str(row["side"]),
        "strongest_ma": str(row["ma_type"]),
        "strongest_period": int(row["period"]),
        "strongest_level": float(row["current_ma"]),
        "strongest_sr_strength_score": float(row["sr_strength_score"]),
        "strongest_evidence": evidence,
    }


def _nearest_fields(
    support: pd.Series | None, resistance: pd.Series | None
) -> dict[str, object]:
    candidates = [
        row
        for row in (support, resistance)
        if row is not None and np.isfinite(row.get("distance_atr", np.nan))
    ]
    if not candidates:
        return {
            "nearest_timeframe": "",
            "nearest_side": "",
            "nearest_ma": "",
            "nearest_period": np.nan,
            "nearest_level": np.nan,
            "nearest_distance_pct": np.nan,
            "nearest_distance_atr": np.nan,
            "nearest_abs_distance_atr": np.nan,
            "nearest_status": "none",
            "nearest_discovery_events": 0,
            "nearest_sr_strength_score": np.nan,
            "nearest_discovery_pass": False,
            "nearest_validation_pass": False,
            "nearest_holdout_pass": False,
        }
    row = min(candidates, key=lambda item: abs(float(item["distance_atr"])))
    distance_atr = float(row["distance_atr"])
    return {
        "nearest_timeframe": str(row["timeframe"]),
        "nearest_side": str(row["side"]),
        "nearest_ma": str(row["ma_type"]),
        "nearest_period": int(row["period"]),
        "nearest_level": float(row["current_ma"]),
        "nearest_distance_pct": float(row.get("distance_pct", np.nan)),
        "nearest_distance_atr": distance_atr,
        "nearest_abs_distance_atr": abs(distance_atr),
        "nearest_status": str(row.get("status", "unverified_candidate")),
        "nearest_discovery_events": int(row.get("discovery_events", 0) or 0),
        "nearest_sr_strength_score": float(row.get("sr_strength_score", np.nan)),
        "nearest_discovery_pass": bool(row.get("discovery_pass", False)),
        "nearest_validation_pass": bool(row.get("validation_pass", False)),
        "nearest_holdout_pass": bool(row.get("holdout_pass", False)),
    }


def _metadata_text(group: pd.DataFrame, column: str) -> str:
    if column not in group or group.empty:
        return ""
    value = group[column].iloc[0]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _mean_metric(group: pd.DataFrame, column: str, scale: float = 1.0) -> float:
    """Return a finite numeric mean without inventing missing evidence."""

    if column not in group or group.empty:
        return np.nan
    values = pd.to_numeric(group[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean() * scale) if not values.empty else np.nan


def _max_metric(group: pd.DataFrame, column: str) -> float:
    """Return a finite numeric maximum without inventing missing evidence."""

    if column not in group or group.empty:
        return np.nan
    values = pd.to_numeric(group[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.max()) if not values.empty else np.nan


def build_instrument_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return exactly one row per typed instrument across all MAs/timeframes."""

    if candidates is None or candidates.empty:
        return pd.DataFrame()
    required = {"ticker", "asset_class", "asset_label", "universe", "display_name"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"instrument metadata missing from candidates: {missing}")

    records: list[dict[str, object]] = []
    keys = ["asset_class", "ticker"]
    for (asset_class, ticker), group in candidates.groupby(keys, sort=False):
        price = _price_row(group)
        support = _best_level(group, "support")
        resistance = _best_level(group, "resistance")
        active = group[_true_mask(group, "active_side")].copy()
        discovery = active[_true_mask(active, "discovery_pass")]
        certified = active[_true_mask(active, "certified")]
        low_confidence = active[_true_mask(active, "low_confidence")]
        thin_holdout = active[_true_mask(active, "certified_thin_holdout")]
        actionable = certified[_true_mask(certified, "actionable")]
        screen_skipped = active[_true_mask(active, "screen_skipped")]
        evaluated = active[~_true_mask(active, "screen_skipped")]
        active_count = len(active)
        evaluated_count = len(evaluated)
        certified_count = len(certified)
        record: dict[str, object] = {
            "asset_class": asset_class,
            "asset_label": str(group["asset_label"].iloc[0]),
            "universe": str(group["universe"].iloc[0]),
            "symbol": ticker,
            "display_name": str(group["display_name"].iloc[0]),
            "sector": _metadata_text(group, "sector"),
            "industry": _metadata_text(group, "industry"),
            "index_memberships": _metadata_text(group, "index_memberships"),
            "current_price": float(price["current_price"]),
            "active_level_count": active_count,
            "tested_level_count": evaluated_count,
            "screen_skipped_level_count": len(screen_skipped),
            "discovery_pass_count": len(discovery),
            "certified_level_count": certified_count,
            "low_confidence_level_count": len(low_confidence),
            "thin_holdout_level_count": len(thin_holdout),
            "actionable_level_count": len(actionable),
            "certification_rate_pct": (
                100.0 * certified_count / evaluated_count if evaluated_count else 0.0
            ),
            "avg_holdout_hit_rate_pct": _mean_metric(
                certified, "holdout_hit_rate", scale=100.0
            ),
            "avg_holdout_return_atr": _mean_metric(
                certified, "holdout_median_fixed_atr"
            ),
            "avg_holdout_net_return_atr": _mean_metric(
                certified, "holdout_net_median_fixed_atr"
            ),
            "avg_q_value": _mean_metric(certified, "q_value"),
            "max_sr_strength_score": _max_metric(active, "sr_strength_score"),
            "overall_evidence": "CERTIFIED" if certified_count else (
                "LOW_CONFIDENCE" if len(low_confidence) else "CANDIDATE_ONLY"
            ),
        }
        record.update(_level_fields("support", support))
        record.update(_level_fields("resistance", resistance))
        record.update(_strongest_fields(active))
        record.update(_nearest_fields(support, resistance))
        records.append(record)
    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "asset_class",
                "certified_level_count",
                "max_sr_strength_score",
                "nearest_abs_distance_atr",
                "symbol",
            ],
            ascending=[True, False, False, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def format_instrument_summary(summary: pd.DataFrame) -> str:
    if summary is None or summary.empty:
        return "No instruments could be summarized."
    lines: list[str] = []
    for asset_label, group in summary.groupby("asset_label", sort=False):
        lines.append(f"\n{asset_label.upper()} ÖZETİ — {len(group)} benzersiz varlık")
        for _, row in group.iterrows():
            support = (
                f"{row['support_timeframe']}:{row['support_ma']}{int(row['support_period'])} "
                f"@{row['support_level']:.4f} [{row['support_evidence']}]"
                if row["support_evidence"] != "NONE"
                else "yok"
            )
            resistance = (
                f"{row['resistance_timeframe']}:{row['resistance_ma']}{int(row['resistance_period'])} "
                f"@{row['resistance_level']:.4f} [{row['resistance_evidence']}]"
                if row["resistance_evidence"] != "NONE"
                else "yok"
            )
            classification = ""
            if row.get("sector"):
                classification += f" | sektör={row['sector']}"
            if row.get("index_memberships"):
                classification += f" | endeksler={row['index_memberships']}"
            strongest = (
                f"{row['strongest_side']} {row['strongest_timeframe']}:"
                f"{row['strongest_ma']}{int(row['strongest_period'])} "
                f"guc={row['strongest_sr_strength_score']:.1f} [{row['strongest_evidence']}]"
                if row["strongest_evidence"] != "NONE"
                else "yok"
            )
            lines.append(
                f"  {row['symbol']} ({row['display_name']}){classification} | fiyat={row['current_price']:.4f} "
                f"| guclu_sr={strongest} | destek={support} | direnc={resistance}"
            )
    lines.append("\nHer varlık bu özette yalnızca bir kez gösterilir.")
    return "\n".join(lines)
