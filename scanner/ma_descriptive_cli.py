#!/usr/bin/env python3
"""Betimsel MA saygı tarayıcısı.

Bu modül guarded/null/FDR/holdout hattından bağımsızdır. Amaç tek veya çoklu
varlıkta hangi hareketli ortalamaların geçmişte daha çok ziyaret edildiğini,
dokunuş sonrası tepki aldığını ve altına/üstüne taşınca ne kadar sürede geri
alındığını sade bir karneyle göstermektir.
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, replace
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .asset_universe import (
        ASSET_CLASSES,
        build_custom_instruments,
        list_universes,
        resolve_universe,
    )
    from .ma_core import (
        MA_TYPES,
        AnalysisConfig,
        TIMEFRAME_CONFIGS,
        compute_ma,
        detect_behavior_touches,
        normalize_ohlcv,
        prepare_frame,
        precompute_forward_outcomes,
    )
    from .ma_data import MarketDataProvider
    from .notifier import send_photo, send_telegram
    from .stock_metadata import enrich_stock_instruments
except ImportError:  # pragma: no cover - direct script execution
    from asset_universe import (  # type: ignore
        ASSET_CLASSES,
        build_custom_instruments,
        list_universes,
        resolve_universe,
    )
    from ma_core import (  # type: ignore
        MA_TYPES,
        AnalysisConfig,
        TIMEFRAME_CONFIGS,
        compute_ma,
        detect_behavior_touches,
        normalize_ohlcv,
        prepare_frame,
        precompute_forward_outcomes,
    )
    from ma_data import MarketDataProvider  # type: ignore
    from notifier import send_photo, send_telegram  # type: ignore
    from stock_metadata import enrich_stock_instruments  # type: ignore


DEFAULT_DESC_PERIODS: tuple[int, ...] = (
    5,
    8,
    10,
    13,
    20,
    21,
    22,
    34,
    50,
    55,
    89,
    100,
    144,
    200,
    233,
    377,
)
DEFAULT_REPORT_MIN_VISITS = 1
_SCORE_FULL_ACTIVITY_VISITS = 10
_DETAIL_EVENTS_PER_MA = 5
_INTRADAY_TIMEFRAMES = {"5m", "15m", "30m", "1h", "4h"}
_SIDE_LABEL = {1: "Destek", -1: "Direnç"}
_TOUCH_LABEL = {1: "Dokunuş-tepki", -1: "Direnç reddi"}
_BAD_SIDE_LABEL = {1: "altında", -1: "üstünde"}
_GOOD_SIDE_LABEL = {1: "üstünde", -1: "altında"}
_SORT_COLUMNS = {
    "visits": ["ziyaret", "saygı_skoru", "tepki_oranı_%", "uzaklık_ATR"],
    "score": ["saygı_skoru", "ziyaret", "tepki_oranı_%", "uzaklık_ATR"],
}

_DNA_CLASS_ORDER = {"Ana DNA": 0, "Guclu": 1, "Izleme": 2, "Zayif": 3}
_DNA_OUTPUT_COLUMNS = [
    "symbol",
    "varlik_turu",
    "sektor",
    "timeframe",
    "MA",
    "tur",
    "periyot",
    "fiyat",
    "seviye",
    "guncel_taraf",
    "temas",
    "temas_gucu_%",
    "tepki_%",
    "geri_%",
    "ort_tepki_ATR",
    "uzak_%",
    "uzak_ATR",
    "dna_skoru",
    "guncel_aksiyon_skoru",
    "dna_sinifi",
    "yorum",
]


@dataclass(frozen=True)
class CrossingEpisode:
    start: object
    end: object
    bars: int
    recovered: bool


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_str_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip().upper() for item in value.split(",") if item.strip())


def format_tr(value: object, decimals: int = 1) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "-"
    return f"{float(number):.{decimals}f}".replace(".", ",")


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _valid_index(mask: np.ndarray) -> int:
    idx = len(mask) - 1
    while idx >= 0 and not mask[idx]:
        idx -= 1
    return idx


def current_streak(state: np.ndarray, valid: np.ndarray) -> tuple[bool, int]:
    idx = _valid_index(valid)
    if idx < 0:
        return False, 0
    last_state = bool(state[idx])
    length = 0
    while idx >= 0 and valid[idx] and bool(state[idx]) == last_state:
        length += 1
        idx -= 1
    return last_state, length


def crossing_episodes(index: pd.Index, state: np.ndarray, valid: np.ndarray) -> list[CrossingEpisode]:
    """Return consecutive True runs, marked recovered when the next valid bar flips."""

    episodes: list[CrossingEpisode] = []
    i = 0
    while i < len(state):
        if valid[i] and state[i]:
            j = i
            while j + 1 < len(state) and valid[j + 1] and state[j + 1]:
                j += 1
            recovered = j + 1 < len(state) and valid[j + 1] and not state[j + 1]
            episodes.append(
                CrossingEpisode(
                    start=index[i],
                    end=index[j],
                    bars=int(j - i + 1),
                    recovered=bool(recovered),
                )
            )
            i = j + 1
        else:
            i += 1
    return episodes


def _config_for_timeframe(
    timeframe: str,
    *,
    zone_atr: float | None = None,
    separation_atr: float | None = None,
    horizon: int | None = None,
    adx_threshold: float | None = None,
) -> AnalysisConfig:
    cfg = TIMEFRAME_CONFIGS.get(timeframe, AnalysisConfig())
    updates: dict[str, float | int] = {}
    if zone_atr is not None:
        updates["zone_atr"] = float(zone_atr)
    if separation_atr is not None:
        updates["separation_atr"] = float(separation_atr)
    if horizon is not None:
        updates["horizon"] = int(horizon)
    if adx_threshold is not None:
        updates["adx_threshold"] = float(adx_threshold)
    return replace(cfg, **updates) if updates else cfg


def _latest_side_state(side: int, close: np.ndarray, ma: np.ndarray, valid: np.ndarray) -> tuple[str, int]:
    if side == 1:
        state = valid & (close < ma)
    else:
        state = valid & (close > ma)
    bad_side, length = current_streak(state, valid)
    return (_BAD_SIDE_LABEL[side] if bad_side else _GOOD_SIDE_LABEL[side]), length


def _episode_event_name(side: int, recovered: bool) -> str:
    if side == 1:
        return "Altına sarkma→geri alma" if recovered else "Altında kalıyor"
    return "Üstüne taşma→geri inme" if recovered else "Üstünde kalıyor"


def _format_event_timestamp(value: object, timeframe: str) -> str:
    timestamp = pd.Timestamp(value)
    if timeframe in _INTRADAY_TIMEFRAMES:
        return timestamp.strftime("%Y-%m-%d %H:%M")
    return str(timestamp.date())


def _format_episode_range(start: object, end: object, timeframe: str) -> str:
    return f"{_format_event_timestamp(start, timeframe)}→{_format_event_timestamp(end, timeframe)}"


def _validate_ma_types(ma_types: Sequence[str]) -> tuple[str, ...]:
    values = tuple(item.upper() for item in ma_types)
    unknown = sorted(set(values) - set(MA_TYPES))
    if unknown:
        raise ValueError(f"Bilinmeyen MA türü: {', '.join(unknown)}")
    return values


def _score_row(
    *,
    visits: int,
    reaction_rate: float,
    recovery_rate: float,
    favorable_bar_pct: float,
    avg_atr_move: float,
) -> float:
    activity = min(1.0, visits / _SCORE_FULL_ACTIVITY_VISITS)
    reaction = reaction_rate / 100.0 if np.isfinite(reaction_rate) else 0.0
    recovery = recovery_rate / 100.0 if np.isfinite(recovery_rate) else 0.0
    stability = favorable_bar_pct / 100.0 if np.isfinite(favorable_bar_pct) else 0.0
    move = float(np.clip((avg_atr_move if np.isfinite(avg_atr_move) else 0.0) / 2.0, 0.0, 1.0))
    return 100.0 * activity * (0.35 * reaction + 0.30 * recovery + 0.20 * stability + 0.15 * move)



def _num_value(value: object, default: float = np.nan) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(default) if pd.isna(number) else float(number)


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _dna_side(price: float, level: float) -> str:
    if np.isfinite(price) and np.isfinite(level) and price >= level:
        return "Destek"
    if np.isfinite(price) and np.isfinite(level):
        return "Direnc"
    return "-"


def _dna_class(score: float, visits: int, max_visits: int, reaction: float, recovery: float) -> str:
    core_visit_floor = max(10, int(np.ceil(max_visits * 0.50)))
    strong_visit_floor = max(5, int(np.ceil(max_visits * 0.35)))
    reaction_ok = np.isfinite(reaction) and reaction >= 50.0
    recovery_ok = np.isfinite(recovery) and recovery >= 65.0
    quality_ok = bool(reaction_ok or recovery_ok)
    if visits >= core_visit_floor and score >= 70.0 and quality_ok:
        return "Ana DNA"
    if visits >= strong_visit_floor and score >= 55.0 and quality_ok:
        return "Guclu"
    if visits > 0 and score >= 40.0:
        return "Izleme"
    return "Zayif"


def _dna_comment(
    dna_class: str,
    visits: int,
    reaction: float,
    recovery: float,
    move_atr: float,
    action_score: float,
) -> str:
    parts: list[str] = []
    if dna_class == "Ana DNA":
        parts.append("karakteristik ortalama")
    elif dna_class == "Guclu":
        parts.append("guclu aday")
    elif dna_class == "Izleme":
        parts.append("izlenebilir")
    else:
        parts.append("zayif kanit")
    if visits >= 20:
        parts.append("cok temas")
    elif visits >= 10:
        parts.append("yeterli temas")
    else:
        parts.append("az temas")
    if np.isfinite(reaction) and reaction >= 65:
        parts.append("tepki guclu")
    elif np.isfinite(reaction) and reaction < 45:
        parts.append("tepki zayif")
    if np.isfinite(recovery) and recovery >= 70:
        parts.append("geri alma guclu")
    if np.isfinite(move_atr) and move_atr > 0.75:
        parts.append("hareket buyuk")
    if action_score >= 70:
        parts.append("su an yakin/onemli")
    return "; ".join(parts)


def build_ma_dna_profile(scorecard: pd.DataFrame, min_visits: int = DEFAULT_REPORT_MIN_VISITS) -> pd.DataFrame:
    """Build a per-symbol MA DNA profile from the descriptive scorecard.

    DNA score is historical character: repeated visits, reaction reliability,
    recovery behaviour and ATR-sized reaction. Current action score adds the
    present distance to that historical character, so a far-but-important MA is
    not confused with a level that is relevant right now.
    """

    if scorecard is None or scorecard.empty:
        return pd.DataFrame(columns=_DNA_OUTPUT_COLUMNS)
    work = scorecard.copy()
    visits = pd.to_numeric(work.get("ziyaret", pd.Series(dtype=float)), errors="coerce").fillna(0)
    min_visits = max(0, int(min_visits))
    work = work[visits >= min_visits].copy()
    if work.empty:
        return pd.DataFrame(columns=_DNA_OUTPUT_COLUMNS)

    records: list[dict[str, object]] = []
    group_cols = [col for col in ["symbol", "timeframe"] if col in work.columns]
    grouped = work.groupby(group_cols, sort=False, dropna=False) if group_cols else [((), work)]
    for _, group in grouped:
        group_visits = pd.to_numeric(group.get("ziyaret", pd.Series(dtype=float)), errors="coerce").fillna(0)
        max_visits = max(1, int(group_visits.max()))
        total_visits = max(1.0, float(group_visits.sum()))
        move_values = pd.to_numeric(group.get("ort_tepki_ATR", pd.Series(dtype=float)), errors="coerce")
        positive_moves = move_values[np.isfinite(move_values) & (move_values > 0)]
        max_move = float(positive_moves.quantile(0.90)) if not positive_moves.empty else 1.0
        max_move = max(max_move, 0.25)
        for _, row in group.iterrows():
            visit_count = int(max(0, _num_value(row.get("ziyaret"), 0.0)))
            reaction = _num_value(row.get("tepki_oran\u0131_%"), np.nan)
            recovery = _num_value(row.get("geri_d\u00f6n\u00fc\u015f_%"), np.nan)
            move_atr = _num_value(row.get("ort_tepki_ATR"), np.nan)
            price = _num_value(row.get("fiyat"), np.nan)
            level = _num_value(row.get("ma_de\u011feri"), np.nan)
            distance_pct = _num_value(row.get("uzakl\u0131k_%"), np.nan)
            distance_atr = _num_value(row.get("uzakl\u0131k_ATR"), np.nan)
            visit_power = _clip01(visit_count / max_visits)
            reaction_power = _clip01(reaction / 100.0)
            recovery_power = _clip01(recovery / 100.0)
            move_power = _clip01(max(0.0, move_atr if np.isfinite(move_atr) else 0.0) / max_move)
            dna_score = 100.0 * (
                0.45 * visit_power
                + 0.25 * reaction_power
                + 0.15 * recovery_power
                + 0.15 * move_power
            )
            abs_atr = abs(distance_atr) if np.isfinite(distance_atr) else np.nan
            abs_pct = abs(distance_pct) if np.isfinite(distance_pct) else np.nan
            if np.isfinite(abs_atr):
                proximity = 1.0 / (1.0 + abs_atr)
            elif np.isfinite(abs_pct):
                proximity = 1.0 / (1.0 + abs_pct / 2.0)
            else:
                proximity = 0.0
            action_score = float(dna_score * (0.25 + 0.75 * proximity))
            dna_class = _dna_class(dna_score, visit_count, max_visits, reaction, recovery)
            records.append(
                {
                    "symbol": str(row.get("symbol", "")).upper(),
                    "varlik_turu": row.get("varl\u0131k_t\u00fcr\u00fc", ""),
                    "sektor": row.get("sekt\u00f6r", ""),
                    "timeframe": row.get("timeframe", ""),
                    "MA": row.get("MA", ""),
                    "tur": row.get("t\u00fcr", ""),
                    "periyot": int(_num_value(row.get("periyot"), 0.0)),
                    "fiyat": price,
                    "seviye": level,
                    "guncel_taraf": _dna_side(price, level),
                    "temas": visit_count,
                    "temas_gucu_%": 100.0 * visit_count / max_visits,
                    "tepki_%": reaction,
                    "geri_%": recovery,
                    "ort_tepki_ATR": move_atr,
                    "uzak_%": distance_pct,
                    "uzak_ATR": distance_atr,
                    "dna_skoru": round(float(dna_score), 1),
                    "guncel_aksiyon_skoru": round(action_score, 1),
                    "dna_sinifi": dna_class,
                    "yorum": _dna_comment(dna_class, visit_count, reaction, recovery, move_atr, action_score),
                    "_class_rank": _DNA_CLASS_ORDER.get(dna_class, 9),
                    "_touch_share": 100.0 * visit_count / total_visits,
                }
            )
    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=_DNA_OUTPUT_COLUMNS)
    result = result.sort_values(
        ["symbol", "timeframe", "_class_rank", "dna_skoru", "temas", "guncel_aksiyon_skoru"],
        ascending=[True, True, True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return result.drop(columns=["_class_rank", "_touch_share"], errors="ignore")


def _sort_scorecard(frame: pd.DataFrame, sort_by: str = "visits") -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = _SORT_COLUMNS.get(sort_by, _SORT_COLUMNS["visits"])
    return frame.sort_values(
        columns,
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def scan_ma_respect(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    ma_types: Sequence[str] = MA_TYPES,
    periods: Sequence[int] = DEFAULT_DESC_PERIODS,
    side: int = 1,
    config: AnalysisConfig | None = None,
    sort_by: str = "visits",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build descriptive MA respect scorecard, event detail, and current-state tables."""

    if side not in {1, -1, 0}:
        raise ValueError("side must be +1 for support, -1 for resistance, or 0 for auto")
    selected_ma_types = _validate_ma_types(ma_types)
    cfg = config or TIMEFRAME_CONFIGS.get(timeframe, AnalysisConfig())
    df = prepare_frame(normalize_ohlcv(frame), cfg)
    forward = precompute_forward_outcomes(df, cfg)
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"]
    current_price = float(df["Close"].iloc[-1])
    current_atr = float(df["ATR"].iloc[-1]) if "ATR" in df else np.nan
    rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    current_rows: list[dict[str, object]] = []

    for ma_type in selected_ma_types:
        for period in periods:
            period = int(period)
            if period >= len(df) - cfg.horizon - 5:
                continue
            if ma_type == "HMA" and period < 13:
                continue
            ma_series = compute_ma(ma_type, df["Close"], volume, period)
            ma = ma_series.reindex(df.index).to_numpy(dtype=float)
            if not np.isfinite(ma[-1]):
                continue

            label = f"{ma_type}{period}"
            valid = np.isfinite(ma) & np.isfinite(close)
            current_ma = float(ma[-1])
            scan_side = side
            if scan_side == 0:
                scan_side = 1 if current_price >= current_ma else -1
            visits = detect_behavior_touches(df, ma_series, cfg, side=scan_side)
            measured = 0
            reaction_count = 0
            pct_moves: list[float] = []
            atr_moves: list[float] = []

            for touch in visits:
                measurement = forward.measurement(touch.position, scan_side)
                if measurement is None:
                    continue
                measured += 1
                is_reaction = measurement.first_hit == 1
                if is_reaction:
                    reaction_count += 1
                atr_moves.append(float(measurement.favorable_atr))
                pct_move = (
                    measurement.favorable_atr * touch.atr / touch.entry * 100.0
                    if touch.entry
                    else np.nan
                )
                if np.isfinite(pct_move):
                    pct_moves.append(float(pct_move))
                events.append(
                    {
                        "symbol": symbol.upper(),
                        "timeframe": timeframe,
                        "taraf": _SIDE_LABEL[scan_side],
                        "MA": label,
                        "tarih": _format_event_timestamp(touch.timestamp, timeframe),
                        "tarih_sort": pd.Timestamp(touch.timestamp),
                        "ma_değeri": float(touch.ma_value),
                        "olay": _TOUCH_LABEL[scan_side],
                        "bar": "-",
                        "tepki": "Evet" if is_reaction else "Hayır",
                        "sonraki_%": pct_move,
                        "sonraki_ATR": float(measurement.favorable_atr),
                        "rejim": touch.regime,
                    }
                )

            reaction_rate = reaction_count / measured * 100.0 if measured else np.nan
            avg_pct_move = float(np.mean(pct_moves)) if pct_moves else np.nan
            avg_atr_move = float(np.mean(atr_moves)) if atr_moves else np.nan

            if scan_side == 1:
                episode_state = valid & (close < ma)
                favorable_side = close > ma
            else:
                episode_state = valid & (close > ma)
                favorable_side = close < ma
            episodes = crossing_episodes(df.index, episode_state, valid)
            recovered_count = sum(1 for episode in episodes if episode.recovered)
            recovery_rate = recovered_count / len(episodes) * 100.0 if episodes else np.nan
            avg_episode_bars = float(np.mean([episode.bars for episode in episodes])) if episodes else 0.0
            max_episode_bars = max((episode.bars for episode in episodes), default=0)

            for episode in episodes:
                events.append(
                    {
                        "symbol": symbol.upper(),
                        "timeframe": timeframe,
                        "taraf": _SIDE_LABEL[scan_side],
                        "MA": label,
                        "tarih": _format_episode_range(episode.start, episode.end, timeframe),
                        "tarih_sort": pd.Timestamp(episode.start),
                        "ma_değeri": np.nan,
                        "olay": _episode_event_name(scan_side, episode.recovered),
                        "bar": int(episode.bars),
                        "tepki": "Evet" if episode.recovered else "Hayır",
                        "sonraki_%": np.nan,
                        "sonraki_ATR": np.nan,
                        "rejim": "",
                    }
                )

            valid_count = int(valid.sum())
            favorable_bar_pct = float(np.mean(favorable_side[valid]) * 100.0) if valid_count else np.nan
            status, streak = _latest_side_state(scan_side, close, ma, valid)
            distance_pct = (current_ma - current_price) / current_price * 100.0
            distance_atr = (
                (current_ma - current_price) / current_atr
                if np.isfinite(current_atr) and current_atr > 0
                else np.nan
            )
            score = _score_row(
                visits=len(visits),
                reaction_rate=reaction_rate,
                recovery_rate=recovery_rate,
                favorable_bar_pct=favorable_bar_pct,
                avg_atr_move=avg_atr_move,
            )

            row = {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "taraf": _SIDE_LABEL[scan_side],
                "MA": label,
                "tür": ma_type,
                "periyot": period,
                "fiyat": current_price,
                "ma_değeri": current_ma,
                "üst/alt_bar_%": favorable_bar_pct,
                "ziyaret": len(visits),
                "tepki_sayısı": reaction_count,
                "tepki_oranı_%": reaction_rate,
                "sarkma_epizodu": len(episodes),
                "geri_dönen": recovered_count,
                "geri_dönüş_%": recovery_rate,
                "ort_sarkma_bar": avg_episode_bars,
                "en_uzun_sarkma_bar": max_episode_bars,
                "ort_tepki_%": avg_pct_move,
                "ort_tepki_ATR": avg_atr_move,
                "şu_an": f"{status} {streak} bar",
                "uzaklık_%": distance_pct,
                "uzaklık_ATR": distance_atr,
                "saygı_skoru": score,
            }
            rows.append(row)
            current_rows.append(row.copy())

    scorecard = _sort_scorecard(pd.DataFrame(rows), sort_by=sort_by)
    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df.sort_values(["MA", "tarih_sort"]).reset_index(drop=True)
        events_df = events_df.drop(columns=["tarih_sort"])
    current = _sort_scorecard(pd.DataFrame(current_rows), sort_by=sort_by)
    return df, scorecard, events_df, current


def _row_identity(row: pd.Series, include_symbol: bool = False) -> str:
    if include_symbol and "symbol" in row:
        asset = str(row.get("varlık_türü", ""))
        suffix = f" ({asset})" if asset else ""
        return f"{row['symbol']}{suffix} — {row['MA']}"
    return str(row["MA"])


def _ma_card_lines(row: pd.Series, rank: int, include_symbol: bool = False) -> list[str]:
    return [
        f"{rank}. {_row_identity(row, include_symbol)} | {int(row['ziyaret'])} temas",
        (
            f"   Tepki %{format_tr(row['tepki_oranı_%'], 0)} | "
            f"Geri alma %{format_tr(row['geri_dönüş_%'], 0)} | "
            f"Ort tepki %{format_tr(row['ort_tepki_%'], 1)}"
        ),
        (
            f"   Şu an {row['şu_an']} | "
            f"Uzaklık %{format_tr(row['uzaklık_%'], 1)}"
        ),
    ]


def _dna_identity(row: pd.Series, include_symbol: bool = False) -> str:
    ma = str(row.get("MA", ""))
    if include_symbol:
        symbol = str(row.get("symbol", ""))
        return f"{symbol} | {ma}" if symbol else ma
    return ma


def _format_dna_block(
    dna_profile: pd.DataFrame | None,
    *,
    top: int = 5,
    include_symbol: bool = False,
) -> list[str]:
    if dna_profile is None or dna_profile.empty:
        return ["", "MA DNA okumasi", "DNA profili uretilemedi."]
    shown = _limit_frame(dna_profile, top if top > 0 else 8)
    lines = ["", "MA DNA okumasi"]
    for rank, (_, row) in enumerate(shown.iterrows(), 1):
        lines.append(
            f"{rank}. {_dna_identity(row, include_symbol)} | {row.get('dna_sinifi', '-')} | "
            f"DNA {format_tr(row.get('dna_skoru'), 1)} | "
            f"Aksiyon {format_tr(row.get('guncel_aksiyon_skoru'), 1)}"
        )
        lines.append(
            f"   {row.get('guncel_taraf', '-')} | {int(row.get('temas', 0))} temas | "
            f"tepki %{format_tr(row.get('tepki_%'), 0)} | uzak %{format_tr(row.get('uzak_%'), 1)}"
        )
        comment = str(row.get("yorum", "")).strip()
        if comment:
            lines.append(f"   Yorum: {comment}")
    lines.append("DNA skoru gecmis karakteri, Aksiyon skoru bugunku yakinligi ekler.")
    return lines


def _filtered_scorecard(scorecard: pd.DataFrame, min_visits: int) -> pd.DataFrame:
    if scorecard.empty:
        return scorecard
    return scorecard[pd.to_numeric(scorecard["ziyaret"], errors="coerce").fillna(0) >= min_visits]


def format_report(
    symbol: str,
    timeframe: str,
    prepared: pd.DataFrame,
    scorecard: pd.DataFrame,
    events: pd.DataFrame,
    current: pd.DataFrame,
    dna_profile: pd.DataFrame | None = None,
    *,
    top: int = 20,
    detail_top: int = 5,
    min_visits: int = DEFAULT_REPORT_MIN_VISITS,
    sort_by: str = "visits",
) -> str:
    current_price = float(prepared["Close"].iloc[-1]) if not prepared.empty else np.nan
    last_date = _format_event_timestamp(prepared.index[-1], timeframe) if not prepared.empty else "-"
    sorted_scorecard = _sort_scorecard(scorecard, sort_by=sort_by)
    visible = _filtered_scorecard(sorted_scorecard, min_visits)
    max_visits = int(sorted_scorecard["ziyaret"].max()) if not sorted_scorecard.empty else 0
    period_values = (
        sorted(int(value) for value in sorted_scorecard["periyot"].dropna().unique())
        if not sorted_scorecard.empty
        else list(DEFAULT_DESC_PERIODS)
    )
    ma_type_values = (
        [str(value) for value in sorted_scorecard["tür"].dropna().drop_duplicates().tolist()]
        if not sorted_scorecard.empty
        else list(MA_TYPES)
    )
    heading = "En çok temas alan ortalamalar" if sort_by == "visits" else "En yüksek saygı skoru"
    lines = [
        f"MA Saygı Özeti - {symbol.upper()} ({timeframe})",
        f"Fiyat {format_tr(current_price, 2)} | Son bar {last_date} | Veri {len(prepared)} bar",
        f"Taranan {len(ma_type_values)} MA türü x {len(period_values)} periyot | Eşik {min_visits}+ temas",
        "Tam liste: ma_respect_scorecard.csv | Olaylar: ma_respect_events.csv",
        "",
        heading,
    ]
    if visible.empty:
        lines.append(f"{min_visits}+ temaslı MA bulunamadı. En yüksek ham temas: {max_visits}.")
        lines.append("Eşiği düşürmek mümkün; ama 1-2 temas iyi ortalama sayılmaz.")
    else:
        for rank, (_, row) in enumerate(_limit_frame(visible, top).iterrows(), 1):
            lines.extend(_ma_card_lines(row, rank))
    lines.extend(_format_dna_block(dna_profile, top=5, include_symbol=False))
    lines.extend(["", "Fiyata yakın güçlüler"])
    current_visible = _filtered_scorecard(current, min_visits)
    if current_visible.empty:
        lines.append("Bu eşikte fiyata yakın güçlü MA yok.")
    else:
        near = current_visible.copy()
        near = _limit_frame(near.reindex(near["uzaklık_%"].abs().sort_values().index), detail_top)
        for rank, (_, row) in enumerate(near.iterrows(), 1):
            side_text = "fiyatın üstünde" if float(row["uzaklık_%"]) > 0 else "fiyatın altında"
            lines.append(
                f"{rank}. {row['MA']} | {side_text} | uzaklık %{format_tr(row['uzaklık_%'], 1)} | "
                f"{int(row['ziyaret'])} temas"
            )
    lines.extend([
        "",
        "Kısa okuma: önce temas sayısı, sonra tepki% ve geri alma% değerine bak.",
        "Telegram sade özet gönderir; tam ham karne CSV artifact içindedir.",
    ])
    return "\n".join(lines)


def format_universe_report(
    scorecard: pd.DataFrame,
    errors: list[str],
    *,
    universe: str,
    dna_profile: pd.DataFrame | None = None,
    timeframe: str,
    top: int,
    per_symbol_top: int,
    min_visits: int,
    sort_by: str,
) -> str:
    sorted_scorecard = _sort_scorecard(scorecard, sort_by=sort_by)
    visible = _filtered_scorecard(sorted_scorecard, min_visits)
    heading = "Genel ilk liste - temas sayısına göre" if sort_by == "visits" else "Genel ilk liste - saygı skoruna göre"
    unique_symbols = sorted_scorecard["symbol"].nunique() if not sorted_scorecard.empty else 0
    lines = [
        f"MA Saygı Özeti - {universe} / {timeframe}",
        f"Taranan varlık {unique_symbols} | Eşik {min_visits}+ temas",
        "Tam liste: ma_respect_scorecard.csv | Varlık özeti: ma_respect_top_per_symbol.csv",
        "",
        heading,
    ]
    if visible.empty:
        max_visits = int(sorted_scorecard["ziyaret"].max()) if not sorted_scorecard.empty else 0
        lines.append(f"{min_visits}+ temaslı MA bulunamadı. En yüksek ham temas: {max_visits}.")
    else:
        for rank, (_, row) in enumerate(_limit_frame(visible, top).iterrows(), 1):
            lines.extend(_ma_card_lines(row, rank, include_symbol=True))
    lines.extend(_format_dna_block(dna_profile, top=8, include_symbol=True))
    if not visible.empty and per_symbol_top > 0:
        lines.extend(["", f"Varlık başına kısa liste ({per_symbol_top} MA)"])
        symbol_order = (
            visible.groupby("symbol", sort=False)[["ziyaret", "saygı_skoru"]]
            .max()
            .sort_values(["ziyaret", "saygı_skoru"], ascending=False)
            .index
        )
        symbol_limit = len(symbol_order) if int(top) <= 0 else max(1, int(top))
        for symbol in symbol_order[:symbol_limit]:
            group = visible[visible["symbol"] == symbol].head(per_symbol_top)
            if group.empty:
                continue
            compact = "; ".join(
                f"{row['MA']} ({int(row['ziyaret'])} temas, %{format_tr(row['tepki_oranı_%'], 0)} tepki)"
                for _, row in group.iterrows()
            )
            lines.append(f"- {symbol}: {compact}")
    if errors:
        lines.extend(["", "Atlanan / hata alan varlıklar"])
        lines.extend(errors[:10])
        if len(errors) > 10:
            lines.append(f"... {len(errors) - 10} hata daha var; tam log Actions çıktısında.")
    lines.extend(["", "Telegram sade özet gönderir; tam ham karne CSV artifact içindedir."])
    return "\n".join(lines)


def _image_cell(value: object, max_chars: int = 24) -> str:
    text = str(value) if value is not None else "-"
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[: max_chars - 1] + "\u2026"
    return text or "-"


def _limit_frame(frame: pd.DataFrame, top: int) -> pd.DataFrame:
    top = int(top)
    if top <= 0:
        return frame
    return frame.head(max(1, top))


def _current_side_label(row: pd.Series) -> str:
    price = pd.to_numeric(pd.Series([row.get("fiyat")]), errors="coerce").iloc[0]
    level = pd.to_numeric(pd.Series([row.get("ma_de\u011feri")]), errors="coerce").iloc[0]
    if pd.isna(price) or pd.isna(level):
        return str(row.get("taraf", "-"))
    if float(price) >= float(level):
        return "Destek"
    return "Diren\u00e7"


def _numeric_column(frame: pd.DataFrame, column: str, *, absolute: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").dropna().astype(float)
    if absolute:
        values = values.abs()
    return values


def _metric_heatmap(frame: pd.DataFrame) -> dict[str, dict[str, float | bool]]:
    specs = {
        "Temas": ("ziyaret", False, False),
        "Tepki": ("tepki_oranı_%", False, False),
        "Geri": ("geri_dönüş_%", False, False),
        "Uzak": ("uzaklık_%", True, True),
        "DNA": ("dna_skoru", False, False),
        "Aksiyon": ("guncel_aksiyon_skoru", False, False),
    }
    result: dict[str, dict[str, float | bool]] = {}
    for header, (column, absolute, lower_is_better) in specs.items():
        values = _numeric_column(frame, column, absolute=absolute)
        if values.empty:
            continue
        result[header] = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "lower_is_better": bool(lower_is_better),
        }
    return result


def _respect_image_rows(frame: pd.DataFrame, *, include_symbol: bool) -> tuple[list[str], list[list[str]], dict[str, dict[str, float | bool]]]:
    if include_symbol:
        headers = ["Varl\u0131k", "MA", "Seviye", "Taraf", "Temas", "Tepki", "Geri", "Uzak"]
    else:
        headers = ["MA", "Seviye", "Taraf", "Temas", "Tepki", "Geri", "Uzak", "Durum"]
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        distance = pd.to_numeric(pd.Series([row.get("uzakl\u0131k_%")]), errors="coerce").iloc[0]
        distance_text = "-" if pd.isna(distance) else format_tr(abs(float(distance)), 1) + "%"
        common = [
            _image_cell(row.get("MA"), 14),
            format_tr(row.get("ma_de\u011feri"), 2),
            _current_side_label(row),
            str(int(row.get("ziyaret", 0))),
            "%" + format_tr(row.get("tepki_oran\u0131_%"), 0),
            "%" + format_tr(row.get("geri_d\u00f6n\u00fc\u015f_%"), 0),
            distance_text,
        ]
        if include_symbol:
            rows.append([_image_cell(row.get("symbol"), 12), *common])
        else:
            rows.append([*common, _image_cell(row.get("\u015fu_an"), 18)])
    return headers, rows, _metric_heatmap(frame)


def _parse_percent_text(value: str) -> float | None:
    try:
        return float(value.replace("%", "").replace("+", "").replace(",", "."))
    except Exception:
        return None


def _metric_value(header: str, value: str) -> float | None:
    parsed = _parse_percent_text(value)
    if parsed is None:
        return None
    if header == "Uzak":
        return abs(parsed)
    return parsed


def _cell_style(
    header: str,
    value: str,
    heatmap: dict[str, dict[str, float | bool]] | None = None,
) -> tuple[str, str | None, bool]:
    if header == "Taraf":
        if "Destek" in value:
            return "#0f7a3a", None, True
        if "Diren" in value:
            return "#b42318", None, True
    if header == "Sinif":
        if "Ana" in value:
            return "#0f7a3a", "#e8f7ee", True
        if "Guclu" in value:
            return "#1d4ed8", "#e8f1ff", True
        if "Izleme" in value:
            return "#9a6700", "#fff7df", True
        return "#b42318", "#fff1ed", True
    stats = (heatmap or {}).get(header)
    metric = _metric_value(header, value)
    if stats and metric is not None:
        low = float(stats["min"])
        avg = float(stats["mean"])
        high = float(stats["max"])
        lower_is_better = bool(stats["lower_is_better"])
        if high <= low:
            return "#1d4ed8", "#e8f1ff", True
        ratio = (high - metric) / (high - low) if lower_is_better else (metric - low) / (high - low)
        if ratio >= 0.67:
            return "#0f7a3a", "#e8f7ee", True
        if (metric <= avg if lower_is_better else metric >= avg):
            return "#1d4ed8", "#e8f1ff", True
        if ratio >= 0.33:
            return "#9a6700", "#fff7df", True
        return "#b42318", "#fff1ed", True
    if header in {"Tepki", "Geri"}:
        number = _parse_percent_text(value)
        if number is None:
            return "#111827", None, False
        if number >= 70:
            return "#0f7a3a", None, True
        if number < 40:
            return "#b42318", None, True
    return "#111827", None, False


def render_respect_table_image(
    headers: list[str],
    rows: list[list[str]],
    *,
    title: str,
    subtitle: str,
    badge: str = "",
    footer: str = "",
    max_rows_per_image: int = 22,
    heatmap: dict[str, dict[str, float | bool]] | None = None,
) -> list[bytes]:
    if not rows:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import patches
    except ImportError:
        return []

    chunks: list[bytes] = []
    chunk_size = max(1, int(max_rows_per_image))
    for chunk_start in range(0, len(rows), chunk_size):
        chunk = rows[chunk_start : chunk_start + chunk_size]
        page_no = chunk_start // chunk_size + 1
        page_count = (len(rows) + chunk_size - 1) // chunk_size
        fig_h = max(5.2, 2.0 + 0.34 * (len(chunk) + 1))
        fig_w = 14.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.patch.set_facecolor("#eef3fb")
        card = patches.FancyBboxPatch(
            (0.02, 0.03),
            0.96,
            0.94,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=0,
            facecolor="#ffffff",
        )
        ax.add_patch(card)
        ax.text(0.05, 0.92, title, fontsize=22, fontweight="bold", color="#102a43", va="center")
        if subtitle:
            ax.text(0.05, 0.865, subtitle, fontsize=12.5, color="#667085", va="center")
        badge_text = badge
        if page_count > 1:
            badge_text = f"{badge} | Liste {page_no}/{page_count}" if badge else f"Liste {page_no}/{page_count}"
        if badge_text:
            badge_box = patches.FancyBboxPatch(
                (0.75, 0.89),
                0.18,
                0.045,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                linewidth=0,
                facecolor="#e8f1ff",
            )
            ax.add_patch(badge_box)
            ax.text(0.84, 0.912, badge_text, fontsize=11, fontweight="bold", color="#1d4ed8", ha="center", va="center")

        left, right = 0.05, 0.95
        table_top = 0.79
        row_h = min(0.052, 0.62 / max(2, len(chunk) + 1))
        table_w = right - left
        if len(headers) == 8:
            weights = [1.35, 1.0, 0.95, 0.75, 0.85, 0.85, 0.9, 1.25]
        else:
            weights = [1.0] * len(headers)
        total_w = sum(weights)
        col_w = [w / total_w * table_w for w in weights]
        x_positions = [left]
        for width in col_w[:-1]:
            x_positions.append(x_positions[-1] + width)

        header_rect = patches.FancyBboxPatch(
            (left, table_top - row_h),
            table_w,
            row_h,
            boxstyle="round,pad=0.003,rounding_size=0.01",
            linewidth=0,
            facecolor="#e8f1ff",
        )
        ax.add_patch(header_rect)
        for col, header in enumerate(headers):
            ax.text(
                x_positions[col] + col_w[col] * 0.04,
                table_top - row_h / 2,
                header,
                fontsize=11,
                fontweight="bold",
                color="#102a43",
                va="center",
                ha="left",
            )

        y = table_top - row_h
        for row_index, row in enumerate(chunk, 1):
            y -= row_h
            bg = "#ffffff" if row_index % 2 else "#f6f8fb"
            ax.add_patch(patches.Rectangle((left, y), table_w, row_h, linewidth=0, facecolor=bg))
            ax.plot([left, right], [y, y], color="#d9e2ec", linewidth=0.65)
            for col, value in enumerate(row):
                header = headers[col]
                color, cell_bg, strong = _cell_style(header, value, heatmap)
                if cell_bg:
                    ax.add_patch(
                        patches.Rectangle(
                            (x_positions[col], y),
                            col_w[col],
                            row_h,
                            linewidth=0,
                            facecolor=cell_bg,
                        )
                    )
                weight = "bold" if strong or col == 0 or header in {"MA", "Varlık"} else "normal"
                ax.text(
                    x_positions[col] + col_w[col] * 0.04,
                    y + row_h / 2,
                    value,
                    fontsize=10.2,
                    fontweight=weight,
                    color=color,
                    va="center",
                    ha="left",
                )
        if footer:
            ax.text(0.05, 0.085, footer, fontsize=11.5, color="#667085", va="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        chunks.append(buf.read())
    return chunks


def render_respect_images(
    scorecard: pd.DataFrame,
    current: pd.DataFrame,
    *,
    label: str,
    timeframe: str,
    top: int,
    detail_top: int,
    min_visits: int,
    sort_by: str,
    include_symbol: bool,
) -> list[bytes]:
    sorted_scorecard = _sort_scorecard(scorecard, sort_by=sort_by)
    visible = _limit_frame(_filtered_scorecard(sorted_scorecard, min_visits), top)
    if visible.empty:
        return []
    symbol_count = sorted_scorecard["symbol"].nunique() if "symbol" in sorted_scorecard else 1
    max_visits = int(sorted_scorecard["ziyaret"].max()) if not sorted_scorecard.empty else 0
    if include_symbol:
        subtitle = f"{timeframe} | {symbol_count} varl\u0131k | e\u015fik {min_visits}+ temas"
    else:
        price = format_tr(visible["fiyat"].iloc[0], 2) if "fiyat" in visible else "-"
        subtitle = f"{timeframe} | fiyat {price} | e\u015fik {min_visits}+ temas"
    avg_visits = float(visible["ziyaret"].mean()) if not visible.empty else 0.0
    badge = f"{len(visible)} sat\u0131r | max {max_visits} | ort {format_tr(avg_visits, 1)} temas"
    headers, rows, heatmap = _respect_image_rows(visible, include_symbol=include_symbol)
    images = render_respect_table_image(
        headers,
        rows,
        title=f"MA Sayg\u0131 \u00d6zeti - {label}",
        subtitle=subtitle,
        badge=badge,
        footer="Renkler: yeşil güçlü, mavi ortalama üstü, sarı orta, kırmızı zayıf. Tam liste CSV artifact içinde.",
        heatmap=heatmap,
    )
    current_visible = _filtered_scorecard(current, min_visits)
    if detail_top >= 0 and not current_visible.empty:
        near = current_visible.copy()
        near = _limit_frame(near.reindex(near["uzakl\u0131k_%"].abs().sort_values().index), detail_top)
        if not near.empty:
            near_headers, near_rows, near_heatmap = _respect_image_rows(near, include_symbol=include_symbol)
            images.extend(
                render_respect_table_image(
                    near_headers,
                    near_rows,
                    title=f"Fiyata Yak\u0131n G\u00fc\u00e7l\u00fc MA - {label}",
                    subtitle=subtitle,
                    badge=f"{len(near)} sat\u0131r",
                    footer="Yakınlık tek başına sinyal değildir; temas ve tepkiyle birlikte okunmalı.",
                    heatmap=near_heatmap,
                )
            )
    return images


def _dna_metric_heatmap(frame: pd.DataFrame) -> dict[str, dict[str, float | bool]]:
    specs = {
        "Temas": ("temas", False, False),
        "Tepki": ("tepki_%", False, False),
        "DNA": ("dna_skoru", False, False),
        "Aksiyon": ("guncel_aksiyon_skoru", False, False),
    }
    result: dict[str, dict[str, float | bool]] = {}
    for header, (column, absolute, lower_is_better) in specs.items():
        values = _numeric_column(frame, column, absolute=absolute)
        if values.empty:
            continue
        result[header] = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "lower_is_better": bool(lower_is_better),
        }
    return result


def _dna_image_rows(frame: pd.DataFrame, *, include_symbol: bool) -> tuple[list[str], list[list[str]], dict[str, dict[str, float | bool]]]:
    if include_symbol:
        headers = ["Varlik", "MA", "Taraf", "Temas", "Tepki", "DNA", "Aksiyon", "Sinif"]
    else:
        headers = ["MA", "Seviye", "Taraf", "Temas", "Tepki", "DNA", "Aksiyon", "Sinif"]
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        common = [
            _image_cell(row.get("MA"), 14),
            str(row.get("guncel_taraf", "-")),
            str(int(row.get("temas", 0))),
            "%" + format_tr(row.get("tepki_%"), 0),
            format_tr(row.get("dna_skoru"), 1),
            format_tr(row.get("guncel_aksiyon_skoru"), 1),
            str(row.get("dna_sinifi", "-")),
        ]
        if include_symbol:
            rows.append([_image_cell(row.get("symbol"), 12), *common])
        else:
            rows.append([
                common[0],
                format_tr(row.get("seviye"), 2),
                *common[1:],
            ])
    return headers, rows, _dna_metric_heatmap(frame)


def render_dna_profile_images(
    dna_profile: pd.DataFrame,
    *,
    label: str,
    timeframe: str,
    top: int,
    include_symbol: bool,
) -> list[bytes]:
    if dna_profile is None or dna_profile.empty:
        return []
    profile = dna_profile.sort_values(
        ["dna_skoru", "temas", "guncel_aksiyon_skoru"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    visual_top = int(top)
    if include_symbol and visual_top <= 0:
        visual_top = 40
    visible = _limit_frame(profile, visual_top)
    if visible.empty:
        return []
    max_dna = float(pd.to_numeric(visible["dna_skoru"], errors="coerce").max())
    headers, rows, heatmap = _dna_image_rows(visible, include_symbol=include_symbol)
    badge = f"{len(visible)} satir | max DNA {format_tr(max_dna, 1)}"
    subtitle = f"{timeframe} | DNA=gecmis karakter, Aksiyon=bugunku yakinlik"
    footer = "Ana DNA uzak olsa da karakteristiktir; Aksiyon skoru su an izlenebilirligi ekler. Tam liste CSV artifact icinde."
    return render_respect_table_image(
        headers,
        rows,
        title=f"MA DNA Profili - {label}",
        subtitle=subtitle,
        badge=badge,
        footer=footer,
        heatmap=heatmap,
    )


def _telegram_image_caption(report: str, *, image_index: int, image_count: int) -> str:
    first_lines = [line for line in report.splitlines()[:4] if line.strip()]
    caption = "\n".join(first_lines)
    if image_count > 1:
        caption = f"{caption}\nG\u00f6rsel {image_index}/{image_count}"
    return caption[:1000]


def send_respect_telegram(token: str, chat_id: str, report: str, images: Sequence[bytes]) -> bool:
    if not images:
        return send_telegram(token, chat_id, report, parse_mode=None)
    ok = True
    for index, image in enumerate(images, 1):
        ok = send_photo(
            token,
            chat_id,
            image,
            caption=_telegram_image_caption(report, image_index=index, image_count=len(images)),
        ) and ok
    return ok

def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def load_frame(args: argparse.Namespace, instrument: object | None = None) -> pd.DataFrame:
    if args.csv:
        return pd.read_csv(args.csv, index_col=0, parse_dates=True)
    provider = MarketDataProvider(source=args.source)
    ticker = str(getattr(instrument, "symbol", args.ticker))
    asset_class = str(getattr(instrument, "asset_class", args.asset_class))
    market = str(getattr(instrument, "market", args.market))
    result = provider.fetch(
        ticker,
        args.timeframe,
        prefer_cache=args.prefer_cache,
        asset_class=asset_class,
        market=market,
    )
    return result.frame


def _resolve_instruments(args: argparse.Namespace) -> list[object]:
    if args.universe == "custom":
        symbols = _csv_list(args.tickers) if args.tickers else _csv_list(args.ticker)
        instruments = build_custom_instruments(
            symbols,
            asset_class=args.asset_class,
            market=args.market,
        )
    else:
        instruments = resolve_universe(args.universe, sector=args.sector)
    instruments = enrich_stock_instruments(instruments)
    if args.max_symbols and args.max_symbols > 0:
        instruments = instruments[: args.max_symbols]
    return list(instruments)


def _attach_instrument_metadata(frame: pd.DataFrame, instrument: object) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result.insert(1, "varlık_türü", getattr(instrument, "asset_label", ""))
    result.insert(2, "evren", getattr(instrument, "universe", ""))
    result.insert(3, "sektör", getattr(instrument, "sector", ""))
    return result


def _write_outputs(
    output_dir: Path,
    report: str,
    scorecard: pd.DataFrame,
    events: pd.DataFrame,
    current: pd.DataFrame,
    dna_profile: pd.DataFrame | None = None,
    images: Sequence[bytes] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ma_respect_report.txt").write_text(report, encoding="utf-8")
    scorecard.to_csv(output_dir / "ma_respect_scorecard.csv", index=False, encoding="utf-8-sig")
    events.to_csv(output_dir / "ma_respect_events.csv", index=False, encoding="utf-8-sig")
    current.to_csv(output_dir / "ma_respect_current.csv", index=False, encoding="utf-8-sig")
    dna_out = dna_profile if dna_profile is not None else pd.DataFrame(columns=_DNA_OUTPUT_COLUMNS)
    dna_out.to_csv(output_dir / "ma_dna_profile.csv", index=False, encoding="utf-8-sig")
    if not dna_out.empty:
        dna_top = (
            dna_out.sort_values(["symbol", "dna_skoru", "temas"], ascending=[True, False, False])
            .groupby("symbol", group_keys=False)
            .head(5)
            .reset_index(drop=True)
        )
        dna_top.to_csv(output_dir / "ma_dna_top_per_symbol.csv", index=False, encoding="utf-8-sig")
    for index, image in enumerate(images or [], 1):
        (output_dir / f"ma_respect_visual_{index}.png").write_bytes(image)
    if not scorecard.empty:
        top_per_symbol = (
            _sort_scorecard(scorecard, sort_by="visits")
            .groupby("symbol", group_keys=False)
            .head(5)
            .reset_index(drop=True)
        )
        top_per_symbol.to_csv(
            output_dir / "ma_respect_top_per_symbol.csv",
            index=False,
            encoding="utf-8-sig",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="ASELS", help="Tek sembol veya custom için ilk sembol")
    parser.add_argument("--tickers", default="", help="Custom sembol listesi: ASELS,THYAO,BTC-USD")
    parser.add_argument("--universe", default="custom", help="Friendly universe key; ör. bist_all_stocks")
    parser.add_argument("--label", default="", help="Custom report/image title")
    parser.add_argument("--sector", default="Tümü / uygulanmaz")
    parser.add_argument("--list-universes", action="store_true")
    parser.add_argument("--asset-class", default="stock", choices=list(ASSET_CLASSES))
    parser.add_argument("--market", default="BIST", choices=["BIST", "GLOBAL"])
    parser.add_argument("--timeframe", default="1d", choices=["1h", "4h", "1d", "1wk", "1mo"])
    parser.add_argument("--source", default="auto", choices=["auto", "borsapy", "yfinance"])
    parser.add_argument("--prefer-cache", action="store_true")
    parser.add_argument("--csv", default=None, help="OHLCV CSV ile tek sembolde çevrimdışı çalıştır")
    parser.add_argument("--periods", default=",".join(map(str, DEFAULT_DESC_PERIODS)))
    parser.add_argument("--ma-types", default=",".join(MA_TYPES))
    parser.add_argument("--side", default="auto", choices=["auto", "support", "resistance"])
    parser.add_argument("--lookback", type=int, default=750, help="Son N bar; 0 tüm veri")
    parser.add_argument("--zone-atr", type=float, default=None)
    parser.add_argument("--separation-atr", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--adx-threshold", type=float, default=None)
    parser.add_argument("--top", type=int, default=0, help="Telegram/image rows; 0 shows all rows")
    parser.add_argument("--detail-top", type=int, default=10, help="Near-price image rows; 0 shows all rows")
    parser.add_argument("--per-symbol-top", type=int, default=2)
    parser.add_argument("--min-visits", type=int, default=DEFAULT_REPORT_MIN_VISITS)
    parser.add_argument("--sort-by", default="visits", choices=["visits", "score"])
    parser.add_argument("--max-symbols", type=int, default=0, help="Test için ilk N varlık; 0 sınır yok")
    parser.add_argument("--output-dir", default="reports/descriptive")
    parser.add_argument("--telegram", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_console_encoding()
    args = build_parser().parse_args(argv)
    if args.list_universes:
        print(pd.DataFrame(list_universes()).to_string(index=False))
        return 0
    instruments = _resolve_instruments(args)
    if args.csv and len(instruments) != 1:
        raise ValueError("--csv yalnız tek sembollü custom taramada kullanılabilir")
    periods = parse_int_list(args.periods)
    ma_types = parse_str_list(args.ma_types)
    side = 0 if args.side == "auto" else (1 if args.side == "support" else -1)
    config = _config_for_timeframe(
        args.timeframe,
        zone_atr=args.zone_atr,
        separation_atr=args.separation_atr,
        horizon=args.horizon,
        adx_threshold=args.adx_threshold,
    )
    all_scorecards: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    all_current: list[pd.DataFrame] = []
    single_report_parts: list[str] = []
    errors: list[str] = []

    for instrument in instruments:
        symbol = str(getattr(instrument, "symbol", args.ticker)).upper()
        try:
            frame = load_frame(args, instrument)
            if args.lookback and args.lookback > 0:
                frame = frame.tail(args.lookback)
            prepared, scorecard, events, current = scan_ma_respect(
                frame,
                symbol=symbol,
                timeframe=args.timeframe,
                ma_types=ma_types,
                periods=periods,
                side=side,
                config=config,
                sort_by=args.sort_by,
            )
            scorecard = _attach_instrument_metadata(scorecard, instrument)
            events = _attach_instrument_metadata(events, instrument)
            current = _attach_instrument_metadata(current, instrument)
            all_scorecards.append(scorecard)
            all_events.append(events)
            all_current.append(current)
            if len(instruments) == 1:
                single_report_parts.append(
                    format_report(
                        symbol,
                        args.timeframe,
                        prepared,
                        scorecard,
                        events,
                        current,
                        build_ma_dna_profile(scorecard, min_visits=max(0, int(args.min_visits))),
                        top=args.top,
                        detail_top=args.detail_top,
                        min_visits=max(0, int(args.min_visits)),
                        sort_by=args.sort_by,
                    )
                )
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
            print(f"{symbol}: hata: {exc}", file=sys.stderr)

    scorecard_all = pd.concat(all_scorecards, ignore_index=True) if all_scorecards else pd.DataFrame()
    events_all = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    current_all = pd.concat(all_current, ignore_index=True) if all_current else pd.DataFrame()
    scorecard_all = _sort_scorecard(scorecard_all, sort_by=args.sort_by)
    current_all = _sort_scorecard(current_all, sort_by=args.sort_by)
    dna_profile_all = build_ma_dna_profile(scorecard_all, min_visits=max(0, int(args.min_visits)))

    if len(instruments) == 1 and single_report_parts:
        report = single_report_parts[0]
    else:
        report = format_universe_report(
            scorecard_all,
            errors,
            universe=args.label or args.universe,
            dna_profile=dna_profile_all,
            timeframe=args.timeframe,
            top=args.top,
            per_symbol_top=args.per_symbol_top,
            min_visits=max(0, int(args.min_visits)),
            sort_by=args.sort_by,
        )
    output_dir = Path(args.output_dir)
    image_label = (
        str(getattr(instruments[0], "symbol", args.ticker)).upper()
        if len(instruments) == 1 and instruments
        else (args.label or args.universe)
    )
    images = render_respect_images(
        scorecard_all,
        current_all,
        label=image_label,
        timeframe=args.timeframe,
        top=args.top,
        detail_top=args.detail_top,
        min_visits=max(0, int(args.min_visits)),
        sort_by=args.sort_by,
        include_symbol=len(instruments) != 1,
    )
    images.extend(
        render_dna_profile_images(
            dna_profile_all,
            label=image_label,
            timeframe=args.timeframe,
            top=args.top,
            include_symbol=len(instruments) != 1,
        )
    )
    _write_outputs(output_dir, report, scorecard_all, events_all, current_all, dna_profile_all, images)
    print(report)
    if images:
        print(f"\nVisual output: {len(images)} PNG")
    print(f"\nOutputs: {output_dir.resolve()}")
    if args.telegram:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if token and chat_id:
            ok = send_respect_telegram(token, chat_id, report, images)
            print("Telegram: " + ("sent" if ok else "failed"))
        else:
            print("Telegram: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID missing, skipped")
    return 0 if not scorecard_all.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
