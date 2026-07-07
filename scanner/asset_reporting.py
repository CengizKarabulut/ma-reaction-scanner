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


def _best_level(group: pd.DataFrame, side: str) -> pd.Series | None:
    rows = group[(group["active_side"]) & (group["side"] == side)].copy()
    if rows.empty:
        return None
    certified = rows[rows["certified"]].copy()
    if not certified.empty:
        return certified.sort_values(
            ["actionable", "rank_score", "q_value"],
            ascending=[False, False, True],
            na_position="last",
        ).iloc[0]
    rows["absolute_distance_atr"] = rows["distance_atr"].abs()
    return rows.sort_values(
        ["absolute_distance_atr", "rank_score"],
        ascending=[True, False],
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
            f"{prefix}_distance_atr": np.nan,
            f"{prefix}_q_value": np.nan,
            f"{prefix}_evidence": "NONE",
        }
    return {
        f"{prefix}_timeframe": str(row["timeframe"]),
        f"{prefix}_ma": str(row["ma_type"]),
        f"{prefix}_period": int(row["period"]),
        f"{prefix}_level": float(row["current_ma"]),
        f"{prefix}_distance_atr": float(row["distance_atr"]),
        f"{prefix}_q_value": (
            float(row["q_value"]) if np.isfinite(row["q_value"]) else np.nan
        ),
        f"{prefix}_evidence": "CERTIFIED"
        if bool(row["certified"])
        else "CANDIDATE_ONLY",
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
        active = group[group["active_side"].fillna(False).astype(bool)].copy()
        discovery = (
            active[active["discovery_pass"].fillna(False).astype(bool)]
            if "discovery_pass" in active
            else active.iloc[0:0]
        )
        certified = active[active["certified"].fillna(False).astype(bool)]
        actionable = (
            certified[certified["actionable"].fillna(False).astype(bool)]
            if "actionable" in certified
            else certified.iloc[0:0]
        )
        active_count = len(active)
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
            "tested_level_count": active_count,
            "discovery_pass_count": len(discovery),
            "certified_level_count": certified_count,
            "actionable_level_count": len(actionable),
            "certification_rate_pct": (
                100.0 * certified_count / active_count if active_count else 0.0
            ),
            "avg_holdout_hit_rate_pct": _mean_metric(
                certified, "holdout_hit_rate", scale=100.0
            ),
            "avg_holdout_return_atr": _mean_metric(
                certified, "holdout_median_fixed_atr"
            ),
            "avg_q_value": _mean_metric(certified, "q_value"),
            "overall_evidence": "CERTIFIED" if certified_count else "CANDIDATE_ONLY",
        }
        record.update(_level_fields("support", support))
        record.update(_level_fields("resistance", resistance))
        records.append(record)
    return (
        pd.DataFrame(records)
        .sort_values(
            ["asset_class", "certified_level_count", "symbol"],
            ascending=[True, False, True],
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
            lines.append(
                f"  {row['symbol']} ({row['display_name']}){classification} | fiyat={row['current_price']:.4f} "
                f"| destek={support} | direnç={resistance}"
            )
    lines.append("\nHer varlık bu özette yalnızca bir kez gösterilir.")
    return "\n".join(lines)
