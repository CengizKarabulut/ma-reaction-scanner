#!/usr/bin/env python3
"""Collapse a full MA scan into the handful of levels worth watching.

A single-symbol scan produces one row per (timeframe, MA type, period, side) -
224 rows for the default grid on one timeframe alone.  That is the right
granularity for research and the wrong one for actually following a stock.

Three things make the full table hard to act on:

1. **Most rows are not live.**  Only the side price is currently on can act as
   support or resistance today; the other side is history.
2. **Neighbouring averages are the same level.**  SMA200, VWMA200 and EMA200
   can sit within half an ATR of each other.  Printing three rows implies three
   levels when there is one zone.
3. **Rank order is not watch order.**  For monitoring, what matters is which
   level price reaches first, not which scored highest.

This module answers the practical question: *if I follow this stock with
moving averages, which ones do I actually put on the chart, and where are
they now?*
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WatchlistConfig:
    """Gates and grouping rules for the compact watch set."""

    cluster_atr: float = 0.50
    min_touches: int = 5
    min_level_score: float = 35.0
    min_plateau_ratio: float = 0.60
    require_positive_adherence: bool = False
    max_zones_per_side: int = 3
    max_distance_atr: float = 8.0

    def __post_init__(self) -> None:
        if self.cluster_atr <= 0:
            raise ValueError("Kumeleme mesafesi sifirdan buyuk olmalidir")
        if self.max_zones_per_side < 1:
            raise ValueError("Taraf basina en az bir bolge gosterilmelidir")
        if self.min_touches < 1:
            raise ValueError("Minimum temas pozitif olmalidir")
        if not 0.0 <= self.min_plateau_ratio <= 1.0:
            raise ValueError("Plato esigi 0 ile 1 arasinda olmalidir")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_REQUIRED = ("side", "current_ma", "distance_atr")


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def apply_quality_gate(frame: pd.DataFrame, config: WatchlistConfig) -> pd.DataFrame:
    """Keep only rows that are live, well evidenced and parameter-robust.

    The plateau gate is skipped where no neighbouring period was scanned
    (``plateau_neighbors == 0``): an untestable row should not be discarded as
    if it had failed the test.
    """

    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=list(getattr(frame, "columns", _REQUIRED)))

    kept = frame.copy()
    if "active_side" in kept.columns:
        kept = kept[kept["active_side"].fillna(False).astype(bool)]
    if "filter_pass" in kept.columns:
        kept = kept[kept["filter_pass"].fillna(True).astype(bool)]

    touches = _numeric(kept, "level_touches", 0.0).fillna(0.0)
    score = _numeric(kept, "level_score", 0.0).fillna(0.0)
    distance = _numeric(kept, "distance_atr").abs()
    kept = kept[
        (touches >= config.min_touches)
        & (score >= config.min_level_score)
        & (distance <= config.max_distance_atr)
    ]

    if "plateau_ratio" in kept.columns:
        ratio = _numeric(kept, "plateau_ratio")
        neighbours = _numeric(kept, "plateau_neighbors", 0.0).fillna(0.0)
        untestable = (neighbours <= 0) | ratio.isna()
        kept = kept[untestable | (ratio >= config.min_plateau_ratio)]

    if config.require_positive_adherence and "adherence_excess_pct" in kept.columns:
        kept = kept[_numeric(kept, "adherence_excess_pct").fillna(0.0) > 0]

    return kept


def _cluster_positions(distances: np.ndarray, cluster_atr: float) -> np.ndarray:
    """Assign cluster ids to ATR distances sorted ascending."""

    labels = np.zeros(len(distances), dtype=int)
    current = 0
    for index in range(1, len(distances)):
        if distances[index] - distances[index - 1] > cluster_atr:
            current += 1
        labels[index] = current
    return labels


def _confidence(ma_families: int, best_score: float, touches: float) -> str:
    if ma_families >= 3 and best_score >= 45.0 and touches >= 8:
        return "Guclu"
    if ma_families >= 2 or best_score >= 45.0:
        return "Orta"
    return "Zayif"


def build_watchlist(
    frame: pd.DataFrame,
    config: WatchlistConfig | None = None,
) -> pd.DataFrame:
    """Return one row per price zone worth watching, nearest to price first.

    Zones are formed on ``distance_atr`` rather than on price, because that
    column is already normalised by volatility and shared across every row of
    the same symbol and timeframe.  Two averages half an ATR apart are, for
    watching purposes, the same line on the chart.

    Touch counts are aggregated with ``max`` and not ``sum``: members of a zone
    are near-identical averages reacting to the same swings, so adding their
    touches would count the same event several times.
    """

    config = config or WatchlistConfig()
    missing = [column for column in _REQUIRED if column not in getattr(frame, "columns", [])]
    if frame is None or len(frame) == 0 or missing:
        return pd.DataFrame(
            columns=[
                "symbol", "timeframe", "side", "zone_low", "zone_high", "zone_mid",
                "distance_pct", "distance_atr", "ma_list", "ma_families",
                "level_touches", "hold_rate_pct", "median_bounce_atr",
                "level_score", "plateau_ratio", "confidence", "analysis_basis",
            ]
        )

    kept = apply_quality_gate(frame, config)
    if len(kept) == 0:
        return build_watchlist(pd.DataFrame(columns=frame.columns), config)

    group_keys = [k for k in ("symbol", "timeframe", "side") if k in kept.columns]
    records: list[dict[str, object]] = []
    for key, group in kept.groupby(group_keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        context = dict(zip(group_keys, key))
        ordered = group.assign(_d=_numeric(group, "distance_atr")).sort_values("_d")
        labels = _cluster_positions(ordered["_d"].to_numpy(dtype=float), config.cluster_atr)
        ordered = ordered.assign(_cluster=labels)

        for _, zone in ordered.groupby("_cluster", sort=True):
            ma_values = _numeric(zone, "current_ma").dropna()
            if ma_values.empty:
                continue
            names = [str(name) for name in zone.get("ma", pd.Series(dtype=str)).tolist()]
            families = {str(item) for item in zone.get("ma_type", pd.Series(dtype=str)).tolist()}
            touches = float(_numeric(zone, "level_touches").max())
            best_score = float(_numeric(zone, "level_score").max())
            records.append(
                {
                    **context,
                    "zone_low": float(ma_values.min()),
                    "zone_high": float(ma_values.max()),
                    "zone_mid": float(ma_values.median()),
                    "distance_pct": float(_numeric(zone, "distance_pct").median()),
                    "distance_atr": float(_numeric(zone, "distance_atr").median()),
                    "ma_list": ", ".join(names),
                    "ma_families": len(families),
                    "level_touches": touches,
                    "hold_rate_pct": float(_numeric(zone, "hold_rate_pct").median()),
                    "median_bounce_atr": float(_numeric(zone, "median_bounce_atr").median()),
                    "level_score": best_score,
                    "plateau_ratio": float(_numeric(zone, "plateau_ratio").min()),
                    "confidence": _confidence(len(families), best_score, touches),
                    "analysis_basis": str(zone.get("analysis_basis", pd.Series(["nominal"])).iloc[0]),
                }
            )

    if not records:
        return build_watchlist(pd.DataFrame(columns=frame.columns), config)

    result = pd.DataFrame(records)
    result["_abs"] = result["distance_atr"].abs()
    result = (
        result.sort_values(group_keys + ["_abs"])
        .groupby(group_keys, sort=False, dropna=False)
        .head(config.max_zones_per_side)
        .sort_values(["_abs"])
        .drop(columns=["_abs"])
        .reset_index(drop=True)
    )
    return result


_WATCH_COLUMNS = {
    "symbol": "Varlik",
    "timeframe": "Zaman Dilimi",
    "side": "Taraf",
    "zone_low": "Bolge Alt",
    "zone_high": "Bolge Ust",
    "zone_mid": "Bolge Orta",
    "distance_pct": "Uzaklik %",
    "distance_atr": "Uzak ATR",
    "ma_list": "Ortalamalar",
    "ma_families": "MA Ailesi",
    "level_touches": "Temas",
    "hold_rate_pct": "Tutma %",
    "median_bounce_atr": "Sicrama ATR",
    "level_score": "Skor",
    "plateau_ratio": "Plato",
    "confidence": "Guven",
    "analysis_basis": "Analiz Bazi",
}


def _short_text(value: object, max_chars: int | None) -> str:
    text = str(value).strip()
    if max_chars is None or len(text) <= max_chars:
        return text
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if max_chars <= 3:
        return f"... +{len(parts)}" if parts else text[:max(0, max_chars)]

    if len(parts) > 1:
        kept: list[str] = []
        for part in parts:
            omitted = len(parts) - len(kept) - 1
            suffix = f", ... +{omitted}" if omitted > 0 else ""
            candidate = ", ".join([*kept, part]) + suffix
            if len(candidate) > max_chars:
                break
            kept.append(part)
        if kept:
            omitted = len(parts) - len(kept)
            suffix = f", ... +{omitted}" if omitted > 0 else ""
            return ", ".join(kept) + suffix

    return text[: max_chars - 3].rstrip() + "..."


def format_watchlist_text(
    watchlist: pd.DataFrame,
    *,
    symbol: str = "",
    price: float | None = None,
    timeframe: str = "",
    trend: str = "",
    include_row_timeframe: bool = True,
    max_ma_list_chars: int | None = None,
) -> str:
    """Render the watch set as a fixed-width block for Telegram or a terminal."""

    header = " ".join(part for part in (symbol, timeframe) if part)
    lines = [f"{header} izleme seti".strip()]
    if price is not None and np.isfinite(price):
        lines.append(f"Fiyat: {price:,.2f}")
    if trend:
        lines.append(f"Trend: {trend}")
    if watchlist is None or len(watchlist) == 0:
        lines.append("")
        lines.append("Esikleri gecen seviye yok. Kaliteli seviye bulunamamasi da bir")
        lines.append("bilgidir: bu hisse su an ortalamalara gore izlenmeye uygun degil.")
        return "\n".join(lines)

    for side in ("Direnc", "Direnç", "Destek"):
        rows = watchlist[watchlist["side"].astype(str) == side]
        if rows.empty:
            continue
        lines.append("")
        lines.append(f"{side.upper()}")
        for _, row in rows.iterrows():
            low, high = float(row["zone_low"]), float(row["zone_high"])
            band = (
                f"{low:,.2f}"
                if abs(high - low) < 0.005
                else f"{low:,.2f}-{high:,.2f}"
            )
            row_timeframe = str(row.get("timeframe", "")).strip()
            timeframe_part = (
                f"{row_timeframe:<4} "
                if include_row_timeframe and row_timeframe
                else ""
            )
            lines.append(
                f"  {timeframe_part}{band:>17}  {float(row['distance_pct']):+6.1f}%  "
                f"{str(row['confidence']):<6} {int(row['level_touches']):>3} temas  "
                f"tutma %{float(row['hold_rate_pct']):.0f}"
            )
            lines.append(
                f"                     {_short_text(row['ma_list'], max_ma_list_chars)}"
            )
    return "\n".join(lines)


def watchlist_table(watchlist: pd.DataFrame) -> pd.DataFrame:
    """Turkish-labelled view for CSV and HTML output."""

    if watchlist is None or len(watchlist) == 0:
        return pd.DataFrame(columns=list(_WATCH_COLUMNS.values()))
    available = [column for column in _WATCH_COLUMNS if column in watchlist.columns]
    return watchlist[available].rename(columns=_WATCH_COLUMNS)
