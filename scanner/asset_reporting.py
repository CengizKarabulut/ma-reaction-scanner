#!/usr/bin/env python3
"""One-instrument-one-row summaries for guarded MA research outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_BEHAVIOR_MIN_TOUCHES = 10

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
    rows = rows[_total_touch_events(rows) >= DEFAULT_BEHAVIOR_MIN_TOUCHES].copy()
    if rows.empty:
        return None
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
            "nearest_total_touch_events": 0,
            "nearest_sr_strength_score": np.nan,
            "nearest_discovery_pass": False,
            "nearest_validation_pass": False,
            "nearest_holdout_pass": False,
        }
    row = min(candidates, key=lambda item: abs(float(item["distance_atr"])))
    distance_atr = float(row["distance_atr"])
    total_touch_events = _row_total_touch_events(row)
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
        "nearest_total_touch_events": total_touch_events,
        "nearest_sr_strength_score": float(row.get("sr_strength_score", np.nan)),
        "nearest_discovery_pass": bool(row.get("discovery_pass", False)),
        "nearest_validation_pass": bool(row.get("validation_pass", False)),
        "nearest_holdout_pass": bool(row.get("holdout_pass", False)),
    }


def _numeric_value(value: object, default: float = 0.0) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if np.isfinite(number):
        return float(number)
    return default


def _row_total_touch_events(row: pd.Series) -> int:
    if "behavior_events" in row.index:
        behavior_events = _numeric_value(row.get("behavior_events"), np.nan)
        if np.isfinite(behavior_events):
            return int(max(0.0, behavior_events))
    total = sum(
        _numeric_value(row.get(f"{prefix}_events", 0.0))
        for prefix in ("discovery", "validation", "holdout")
    )
    return int(max(0.0, total))


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


_BEHAVIOR_OUTPUT_COLUMNS = [
    "asset_class",
    "asset_label",
    "symbol",
    "display_name",
    "sector",
    "industry",
    "index_memberships",
    "timeframe",
    "side",
    "ma_type",
    "period",
    "ma_label",
    "current_price",
    "current_ma",
    "distance_pct",
    "distance_atr",
    "abs_distance_atr",
    "total_touch_events",
    "recent_touch_events",
    "reaction_hit_rate_pct",
    "reaction_median_fixed_atr",
    "reaction_quality_score",
    "near_action_score",
    "sr_strength_score",
    "discovery_pass",
    "certified",
    "low_confidence",
    "evidence_label",
    "status",
]


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _total_touch_events(frame: pd.DataFrame) -> pd.Series:
    if "behavior_events" in frame:
        return _numeric(frame, "behavior_events")
    return (
        _numeric(frame, "discovery_events")
        + _numeric(frame, "validation_events")
        + _numeric(frame, "holdout_events")
    )


def _weighted_metric(frame: pd.DataFrame, suffix: str) -> pd.Series:
    behavior_column = f"behavior_{suffix}"
    if behavior_column in frame:
        return _numeric(frame, behavior_column, np.nan)
    total = pd.Series(0.0, index=frame.index, dtype=float)
    weighted = pd.Series(0.0, index=frame.index, dtype=float)
    for prefix in ("discovery", "validation", "holdout"):
        events = _numeric(frame, f"{prefix}_events")
        values = _numeric(frame, f"{prefix}_{suffix}")
        total = total + events
        weighted = weighted + events * values
    return (weighted / total.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _evidence_label(row: pd.Series) -> str:
    if _is_true(row.get("certified", False)):
        return "CERTIFIED"
    if _is_true(row.get("low_confidence", False)):
        return "LOW_CONFIDENCE"
    if _is_true(row.get("discovery_pass", False)):
        return "DISCOVERY_ONLY"
    return "CANDIDATE_ONLY"


def _prepare_behavior_rows(candidates: pd.DataFrame, min_touches: int = DEFAULT_BEHAVIOR_MIN_TOUCHES) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=_BEHAVIOR_OUTPUT_COLUMNS)
    rows = candidates[_true_mask(candidates, "active_side")].copy()
    if rows.empty:
        return pd.DataFrame(columns=_BEHAVIOR_OUTPUT_COLUMNS)
    if "ticker" not in rows:
        raise ValueError("behavior profile requires ticker column")

    for column in (
        "asset_class",
        "asset_label",
        "display_name",
        "sector",
        "industry",
        "index_memberships",
        "status",
    ):
        if column not in rows:
            rows[column] = ""

    rows["total_touch_events"] = _total_touch_events(rows).astype(int)
    rows = rows[rows["total_touch_events"] >= max(1, int(min_touches))].copy()
    if rows.empty:
        return pd.DataFrame(columns=_BEHAVIOR_OUTPUT_COLUMNS)
    validation_events = _numeric(rows, "validation_events")
    holdout_events = _numeric(rows, "holdout_events")
    rows["recent_touch_events"] = (validation_events + holdout_events).astype(int)
    rows["abs_distance_atr"] = _numeric(rows, "distance_atr", np.nan).abs()
    rows["reaction_hit_rate_pct"] = _weighted_metric(rows, "hit_rate") * 100.0
    rows["reaction_median_fixed_atr"] = _weighted_metric(rows, "median_fixed_atr")
    rows["reaction_median_fixed_atr"] = rows["reaction_median_fixed_atr"].replace(
        [np.inf, -np.inf], np.nan
    )
    max_touch = max(float(rows["total_touch_events"].max()), 1.0)
    touch_component = np.log1p(rows["total_touch_events"].clip(lower=0)) / np.log1p(max_touch)
    hit_component = (rows["reaction_hit_rate_pct"] / 100.0).fillna(0.0).clip(0.0, 1.0)
    effect_component = rows["reaction_median_fixed_atr"].fillna(0.0).clip(lower=0.0, upper=1.5) / 1.5
    sr_component = _numeric(rows, "sr_strength_score", 0.0).clip(0.0, 100.0) / 100.0
    rows["reaction_quality_score"] = 100.0 * (
        0.35 * touch_component
        + 0.30 * hit_component
        + 0.20 * effect_component
        + 0.15 * sr_component
    )
    proximity_component = 1.0 / (1.0 + rows["abs_distance_atr"].fillna(np.inf).clip(lower=0.0))
    rows["near_action_score"] = 100.0 * (
        0.45 * proximity_component
        + 0.35 * touch_component
        + 0.20 * (rows["reaction_quality_score"].fillna(0.0).clip(0.0, 100.0) / 100.0)
    )
    rows["symbol"] = rows["ticker"].astype(str).str.upper()
    rows["ma_label"] = (
        rows["timeframe"].astype(str)
        + ":"
        + rows["ma_type"].astype(str)
        + rows["period"].astype(int).astype(str)
    )
    rows["evidence_label"] = rows.apply(_evidence_label, axis=1)
    for column in ("current_price", "current_ma", "distance_pct", "distance_atr", "sr_strength_score"):
        if column in rows:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
        else:
            rows[column] = np.nan
    for column in _BEHAVIOR_OUTPUT_COLUMNS:
        if column not in rows:
            rows[column] = np.nan if column not in {"sector", "industry", "index_memberships"} else ""
    return rows[_BEHAVIOR_OUTPUT_COLUMNS].copy()


def _rank_behavior_rows(
    rows: pd.DataFrame,
    sort_columns: list[str],
    ascending: list[bool],
    top_n: int,
    *,
    one_row_per_instrument: bool,
) -> pd.DataFrame:
    ranked = rows.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
    )
    if one_row_per_instrument:
        ranked = ranked.drop_duplicates(["asset_class", "symbol"], keep="first")
    return ranked.head(top_n).reset_index(drop=True)


def build_behavior_profiles(
    candidates: pd.DataFrame,
    top_n: int = 20,
    min_touches: int = DEFAULT_BEHAVIOR_MIN_TOUCHES,
) -> dict[str, pd.DataFrame]:
    """Build explanatory MA behaviour tables independent of strict certification."""

    rows = _prepare_behavior_rows(candidates, min_touches=min_touches)
    empty = pd.DataFrame(columns=_BEHAVIOR_OUTPUT_COLUMNS)
    if rows.empty:
        return {
            "most_visited": empty.copy(),
            "best_reactions": empty.copy(),
            "near_price": empty.copy(),
        }
    top_n = max(1, int(top_n))
    one_row_per_instrument = (
        rows[["asset_class", "symbol"]].drop_duplicates().shape[0] > 1
    )
    most_visited = _rank_behavior_rows(
        rows,
        [
            "total_touch_events",
            "recent_touch_events",
            "reaction_quality_score",
            "abs_distance_atr",
        ],
        [False, False, False, True],
        top_n,
        one_row_per_instrument=one_row_per_instrument,
    )
    best_reactions = _rank_behavior_rows(
        rows,
        [
            "reaction_quality_score",
            "reaction_hit_rate_pct",
            "reaction_median_fixed_atr",
            "total_touch_events",
            "abs_distance_atr",
        ],
        [False, False, False, False, True],
        top_n,
        one_row_per_instrument=one_row_per_instrument,
    )

    near_sort_columns = [
        "near_action_score",
        "abs_distance_atr",
        "total_touch_events",
        "reaction_quality_score",
        "sr_strength_score",
    ]
    near_ascending = [False, True, False, False, False]
    if one_row_per_instrument:
        near_price = _rank_behavior_rows(
            rows,
            near_sort_columns,
            near_ascending,
            top_n,
            one_row_per_instrument=True,
        )
    else:
        near_selected = []
        side_n = min(top_n, 5)
        for _, group in rows.groupby(["asset_class", "symbol", "side"], sort=False):
            near_selected.append(
                group.sort_values(
                    near_sort_columns,
                    ascending=near_ascending,
                    na_position="last",
                ).head(side_n)
            )
        near_price = (
            pd.concat(near_selected, ignore_index=True)
            if near_selected
            else empty.copy()
        )
    return {
        "most_visited": most_visited.reset_index(drop=True),
        "best_reactions": best_reactions.reset_index(drop=True),
        "near_price": near_price.reset_index(drop=True),
    }


def _format_behavior_table(title: str, table: pd.DataFrame, max_rows: int = 80) -> list[str]:
    lines = [f"\n{title}"]
    if table is None or table.empty:
        lines.append("  none")
        return lines
    shown = table.head(max_rows)
    for (asset_label, symbol), group in shown.groupby(["asset_label", "symbol"], sort=False):
        display = str(group["display_name"].iloc[0]) if "display_name" in group else symbol
        lines.append(f"  {asset_label} {symbol} ({display})")
        for _, row in group.iterrows():
            lines.append(
                "    "
                f"{row['side']} {row['ma_label']} @ {_fmt_float(row['current_ma'], 4)} "
                f"uzak={_fmt_float(abs(row['distance_pct']), 2)}%/{_fmt_float(row['abs_distance_atr'], 2)}ATR "
                f"temas={int(row['total_touch_events'])} "
                f"tepki={_fmt_float(row['reaction_hit_rate_pct'], 1)}% "
                f"medATR={_fmt_float(row['reaction_median_fixed_atr'], 2)} "
                f"skor={_fmt_float(row['reaction_quality_score'], 1)}"
            )
    if len(table) > len(shown):
        lines.append(f"  ... +{len(table) - len(shown)} more rows")
    return lines


def _fmt_float(value: object, decimals: int) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(number) else f"{number:.{decimals}f}"


def format_behavior_profiles(profiles: dict[str, pd.DataFrame], max_rows: int = 80) -> str:
    lines = ["MA DAVRANIS PROFILI"]
    lines.extend(_format_behavior_table("En sik ugradigi ortalamalar", profiles.get("most_visited"), max_rows))
    lines.extend(_format_behavior_table("En cok tepki aldigi ortalamalar", profiles.get("best_reactions"), max_rows))
    lines.extend(_format_behavior_table("Fiyata yakin guclu temasli destek/direnc", profiles.get("near_price"), max_rows))
    lines.append("\nNot: Bu tablolar kesif ve davranis okumasidir; CERTIFIED etiketi ayrica belirtilir.")
    return "\n".join(lines)


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
