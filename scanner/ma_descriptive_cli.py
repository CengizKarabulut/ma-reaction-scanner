#!/usr/bin/env python3
"""Betimsel MA saygı tarayıcısı.

Bu modül guarded/null/FDR/holdout hattından bağımsızdır. Amaç tek veya çoklu
varlıkta hangi hareketli ortalamaların geçmişte daha çok ziyaret edildiğini,
dokunuş sonrası tepki aldığını ve altına/üstüne taşınca ne kadar sürede geri
alındığını sade bir karneyle göstermektir.
"""

from __future__ import annotations

import argparse
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
    from .notifier import send_telegram
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
    from notifier import send_telegram  # type: ignore
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
DEFAULT_REPORT_MIN_VISITS = 5
_SCORE_FULL_ACTIVITY_VISITS = 10
_SIDE_LABEL = {1: "Destek", -1: "Direnç"}
_TOUCH_LABEL = {1: "Dokunuş-tepki", -1: "Direnç reddi"}
_BAD_SIDE_LABEL = {1: "altında", -1: "üstünde"}
_GOOD_SIDE_LABEL = {1: "üstünde", -1: "altında"}
_SORT_COLUMNS = {
    "visits": ["ziyaret", "saygı_skoru", "tepki_oranı_%", "uzaklık_ATR"],
    "score": ["saygı_skoru", "ziyaret", "tepki_oranı_%", "uzaklık_ATR"],
}


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

    if side not in {1, -1}:
        raise ValueError("side must be +1 for support or -1 for resistance")
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
            visits = detect_behavior_touches(df, ma_series, cfg, side=side)
            measured = 0
            reaction_count = 0
            pct_moves: list[float] = []
            atr_moves: list[float] = []

            for touch in visits:
                measurement = forward.measurement(touch.position, side)
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
                        "taraf": _SIDE_LABEL[side],
                        "MA": label,
                        "tarih": str(pd.Timestamp(touch.timestamp).date()),
                        "tarih_sort": pd.Timestamp(touch.timestamp),
                        "ma_değeri": float(touch.ma_value),
                        "olay": _TOUCH_LABEL[side],
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

            if side == 1:
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
                        "taraf": _SIDE_LABEL[side],
                        "MA": label,
                        "tarih": f"{pd.Timestamp(episode.start).date()}→{pd.Timestamp(episode.end).date()}",
                        "tarih_sort": pd.Timestamp(episode.start),
                        "ma_değeri": np.nan,
                        "olay": _episode_event_name(side, episode.recovered),
                        "bar": int(episode.bars),
                        "tepki": "Evet" if episode.recovered else "Hayır",
                        "sonraki_%": np.nan,
                        "sonraki_ATR": np.nan,
                        "rejim": "",
                    }
                )

            valid_count = int(valid.sum())
            favorable_bar_pct = float(np.mean(favorable_side[valid]) * 100.0) if valid_count else np.nan
            status, streak = _latest_side_state(side, close, ma, valid)
            current_ma = float(ma[-1])
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
                "taraf": _SIDE_LABEL[side],
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


def _score_row_text(row: pd.Series, include_symbol: bool = False) -> str:
    prefix = f"{row['symbol']} | " if include_symbol and "symbol" in row else ""
    asset = f"{row.get('varlık_türü', '')} | " if include_symbol and row.get("varlık_türü", "") else ""
    return (
        f"{prefix}{asset}{row['MA']:>7} | {int(row['ziyaret']):>2} temas | "
        f"tepki {format_tr(row['tepki_oranı_%'], 0):>3}% | "
        f"geri {format_tr(row['geri_dönüş_%'], 0):>3}% | "
        f"{format_tr(row['ort_tepki_%'], 1)}% | {row['şu_an']} | "
        f"uzak {format_tr(row['uzaklık_%'], 1)}% | skor {format_tr(row['saygı_skoru'], 1)}"
    )


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
    *,
    top: int = 20,
    detail_top: int = 5,
    min_visits: int = DEFAULT_REPORT_MIN_VISITS,
    sort_by: str = "visits",
) -> str:
    current_price = float(prepared["Close"].iloc[-1]) if not prepared.empty else np.nan
    last_date = pd.Timestamp(prepared.index[-1]).date() if not prepared.empty else "-"
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
    heading = "ziyaret/temas sayısına göre" if sort_by == "visits" else "saygı skoruna göre"
    lines = [
        f"📊 {symbol.upper()} — {timeframe} betimsel MA saygı taraması",
        f"Son bar: {last_date} | Fiyat: {format_tr(current_price, 2)} | Bar: {len(prepared)}",
        f"Taranan periyotlar: {','.join(map(str, period_values))}",
        "MA türleri: " + ",".join(ma_type_values),
        "",
        "Bu rapor garanti değildir; ham geçmiş davranışı gösterir.",
        "Amaç: hissenin hangi ortalamalara tekrar tekrar temas edip tepki verdiğini görmek.",
        f"Telegram eşiği: en az {min_visits} ziyaret. Tam ham karne CSV çıktısındadır.",
        "",
        f"=== KARNE — {heading} ===",
        "MA | ziyaret | tepki% | geri dönüş% | ort tepki | şu an | uzaklık | skor",
    ]
    if visible.empty:
        lines.append(
            f"{min_visits}+ ziyaretli MA bulunamadı. En yüksek ham ziyaret sayısı: {max_visits}."
        )
        lines.append("Bu sembolde ya veri az, ya da mevcut toleransla sık temas eden MA yok.")
    else:
        for _, row in visible.head(max(1, int(top))).iterrows():
            lines.append(_score_row_text(row))
    lines.extend(
        [
            "",
            "Not: Ziyaret sayısı düşükse satır 'iyi ortalama' diye okunmamalı.",
            "Saygı skoru; ziyaret, tepki, geri dönüş, üst/alt zaman ve tepki büyüklüğünü birlikte tartar.",
            "",
            f"=== OLAY DÖKÜMÜ — en iyi {max(1, int(detail_top))} MA ===",
        ]
    )
    top_labels = set(visible.head(max(1, int(detail_top)))["MA"].tolist()) if not visible.empty else set()
    detail = events[events["MA"].isin(top_labels)] if not events.empty else pd.DataFrame()
    if detail.empty:
        lines.append("Bu eşikte olay dökümü yok.")
    else:
        for ma_label in visible.head(max(1, int(detail_top)))["MA"]:
            ma_events = detail[detail["MA"] == ma_label].head(30)
            if ma_events.empty:
                continue
            lines.append(f"• {ma_label}")
            for _, event in ma_events.iterrows():
                move = "" if pd.isna(event.get("sonraki_%")) else f" | sonraki {format_tr(event['sonraki_%'], 1)}%"
                bars = "" if event.get("bar") == "-" else f" | {event['bar']} bar"
                reaction = "" if not event.get("tepki") else f" | tepki: {event['tepki']}"
                lines.append(f"  {event['tarih']} | {event['olay']}{bars}{reaction}{move}")
    lines.extend(["", "=== MEVCUT DURUM — fiyata yakın iyi ortalamalar ==="])
    current_visible = _filtered_scorecard(current, min_visits)
    if current_visible.empty:
        lines.append("Bu eşikte fiyata yakın iyi ortalama gösterilmedi.")
    else:
        near = current_visible.copy()
        near = near.reindex(near["uzaklık_%"].abs().sort_values().index).head(max(1, int(detail_top)))
        for _, row in near.iterrows():
            side_text = "fiyat MA'nın altında" if float(row["uzaklık_%"]) > 0 else "fiyat MA'nın üstünde"
            lines.append(
                f"{row['MA']}: {side_text}; uzaklık {format_tr(row['uzaklık_%'], 1)}%, "
                f"{row['şu_an']}, ziyaret {int(row['ziyaret'])}, skor {format_tr(row['saygı_skoru'], 1)}"
            )
    return "\n".join(lines)


def format_universe_report(
    scorecard: pd.DataFrame,
    errors: list[str],
    *,
    universe: str,
    timeframe: str,
    top: int,
    per_symbol_top: int,
    min_visits: int,
    sort_by: str,
) -> str:
    sorted_scorecard = _sort_scorecard(scorecard, sort_by=sort_by)
    visible = _filtered_scorecard(sorted_scorecard, min_visits)
    heading = "temas sayısına göre" if sort_by == "visits" else "saygı skoruna göre"
    unique_symbols = sorted_scorecard["symbol"].nunique() if not sorted_scorecard.empty else 0
    lines = [
        f"📊 Betimsel MA saygı taraması — {universe} / {timeframe}",
        f"Taranan varlık: {unique_symbols} | Gösterim eşiği: {min_visits}+ ziyaret",
        f"Sıralama: {heading}. Tam liste CSV artifact içindedir.",
        "",
        "=== GENEL TOP ===",
        "Varlık | Tür | MA | ziyaret | tepki% | geri% | ort tepki | şu an | uzaklık | skor",
    ]
    if visible.empty:
        max_visits = int(sorted_scorecard["ziyaret"].max()) if not sorted_scorecard.empty else 0
        lines.append(f"{min_visits}+ ziyaretli MA bulunamadı. En yüksek ham ziyaret: {max_visits}.")
    else:
        for _, row in visible.head(max(1, int(top))).iterrows():
            lines.append(_score_row_text(row, include_symbol=True))
    if not visible.empty and per_symbol_top > 0:
        lines.extend(["", f"=== VARLIK BAŞINA EN İYİ {per_symbol_top} ==="])
        symbol_order = (
            visible.groupby("symbol", sort=False)[["ziyaret", "saygı_skoru"]]
            .max()
            .sort_values(["ziyaret", "saygı_skoru"], ascending=False)
            .index
        )
        for symbol in symbol_order[: max(1, int(top))]:
            group = visible[visible["symbol"] == symbol].head(per_symbol_top)
            if group.empty:
                continue
            lines.append(f"• {symbol}")
            for _, row in group.iterrows():
                lines.append("  " + _score_row_text(row))
    if errors:
        lines.extend(["", "=== ATLANAN / HATA ALAN VARLIKLAR ==="])
        lines.extend(errors[:20])
        if len(errors) > 20:
            lines.append(f"... {len(errors) - 20} hata daha var; tam log Actions çıktısında.")
    return "\n".join(lines)


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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ma_respect_report.txt").write_text(report, encoding="utf-8")
    scorecard.to_csv(output_dir / "ma_respect_scorecard.csv", index=False, encoding="utf-8-sig")
    events.to_csv(output_dir / "ma_respect_events.csv", index=False, encoding="utf-8-sig")
    current.to_csv(output_dir / "ma_respect_current.csv", index=False, encoding="utf-8-sig")
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
    parser.add_argument("--side", default="support", choices=["support", "resistance"])
    parser.add_argument("--lookback", type=int, default=750, help="Son N bar; 0 tüm veri")
    parser.add_argument("--zone-atr", type=float, default=None)
    parser.add_argument("--separation-atr", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--adx-threshold", type=float, default=None)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--detail-top", type=int, default=5)
    parser.add_argument("--per-symbol-top", type=int, default=3)
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
    side = 1 if args.side == "support" else -1
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
                        top=args.top,
                        detail_top=args.detail_top,
                        min_visits=max(1, int(args.min_visits)),
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

    if len(instruments) == 1 and single_report_parts:
        report = single_report_parts[0]
    else:
        report = format_universe_report(
            scorecard_all,
            errors,
            universe=args.universe,
            timeframe=args.timeframe,
            top=args.top,
            per_symbol_top=args.per_symbol_top,
            min_visits=max(1, int(args.min_visits)),
            sort_by=args.sort_by,
        )
    output_dir = Path(args.output_dir)
    _write_outputs(output_dir, report, scorecard_all, events_all, current_all)
    print(report)
    print(f"\nÇıktılar: {output_dir.resolve()}")
    if args.telegram:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if token and chat_id:
            ok = send_telegram(token, chat_id, report, parse_mode=None)
            print("Telegram: " + ("gönderildi" if ok else "başarısız"))
        else:
            print("Telegram: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID yok, atlandı")
    return 0 if not scorecard_all.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())