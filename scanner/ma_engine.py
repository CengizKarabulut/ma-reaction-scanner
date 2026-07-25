#!/usr/bin/env python3
"""Unified moving-average trend and reaction engine.

Trend state, historical reaction quality, and current proximity are deliberately
separate. A current nearby level cannot improve weak historical evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


MA_TYPES: tuple[str, ...] = ("SMA", "EMA", "WMA", "VWMA", "KAMA", "ALMA", "HMA")
TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo")
DEFAULT_PERIODS: tuple[int, ...] = (
    5, 8, 10, 13, 20, 21, 22, 34, 50, 55, 89, 100, 144, 200, 233, 377
)


@dataclass(frozen=True)
class ScanConfig:
    ma_types: tuple[str, ...] = MA_TYPES
    periods: tuple[int, ...] = DEFAULT_PERIODS
    trend_slope_bars: int = 10
    trend_slope_threshold_atr: float = 0.10
    atr_period: int = 14
    touch_zone_atr: float = 0.20
    separation_atr: float = 2.0
    min_touches: int = 12
    stop_buffer_atr: float = 0.20
    trailing_stop_atr: float = 2.0
    max_holding_bars: int = 20
    roundtrip_cost_bps: float = 25.0
    min_edge_r: float = 0.10
    near_distance_atr: float = 1.0

    def __post_init__(self) -> None:
        unknown = sorted(set(self.ma_types) - set(MA_TYPES))
        if unknown:
            raise ValueError(f"Bilinmeyen MA türleri: {', '.join(unknown)}")
        if not self.ma_types or not self.periods:
            raise ValueError("En az bir MA türü ve periyot seçilmelidir")
        if any(int(period) < 2 for period in self.periods):
            raise ValueError("MA periyotları en az 2 olmalıdır")
        integers = (
            self.trend_slope_bars,
            self.atr_period,
            self.min_touches,
            self.max_holding_bars,
        )
        if any(int(value) < 1 for value in integers):
            raise ValueError("Bar ve olay eşikleri pozitif olmalıdır")
        non_negative = (
            self.trend_slope_threshold_atr,
            self.touch_zone_atr,
            self.separation_atr,
            self.stop_buffer_atr,
            self.trailing_stop_atr,
            self.roundtrip_cost_bps,
            self.min_edge_r,
            self.near_distance_atr,
        )
        if any(float(value) < 0 for value in non_negative):
            raise ValueError("ATR, maliyet ve eşik ayarları negatif olamaz")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ma_types"] = list(self.ma_types)
        payload["periods"] = list(self.periods)
        return payload


@dataclass(frozen=True)
class Touch:
    position: int
    timestamp: object
    side: int
    ma_value: float
    atr: float


@dataclass(frozen=True)
class Trade:
    touch_position: int
    entry_position: int
    exit_position: int
    side: int
    entry: float
    initial_stop: float
    exit_price: float
    net_r: float
    reached_1r: bool
    reached_2r: bool
    mfe_r: float
    mae_r: float
    exit_reason: str


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("OHLCV verisi boş")
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(item[0]) for item in df.columns]
    lookup = {str(column).strip().lower(): column for column in df.columns}
    required = ("open", "high", "low", "close")
    missing = [name for name in required if name not in lookup]
    if missing:
        raise ValueError(f"Eksik OHLC sütunları: {', '.join(missing)}")
    out = pd.DataFrame(index=pd.to_datetime(df.index))
    for name in required:
        out[name.title()] = pd.to_numeric(df[lookup[name]], errors="coerce")
    out["Volume"] = (
        pd.to_numeric(df[lookup["volume"]], errors="coerce").fillna(0.0)
        if "volume" in lookup
        else 0.0
    )
    out = (
        out[~out.index.duplicated(keep="last")]
        .sort_index()
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    if out.empty:
        raise ValueError("Temizleme sonrası OHLCV verisi boş")
    if (out[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC fiyatları pozitif olmalıdır")
    tolerance = 1e-12
    if (
        out["High"] + tolerance
        < out[["Open", "Close", "Low"]].max(axis=1)
    ).any() or (
        out["Low"] - tolerance
        > out[["Open", "Close", "High"]].min(axis=1)
    ).any():
        raise ValueError("OHLC sıralaması geçersiz")
    return out


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["Close"].shift(1)
    return pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def prepare_frame(frame: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    df = normalize_ohlcv(frame)
    df["ATR"] = true_range(df).ewm(
        alpha=1.0 / atr_period,
        adjust=False,
        min_periods=atr_period,
    ).mean()
    return df


def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1.0, period + 1.0)
    denominator = float(weights.sum())
    return series.rolling(period).apply(
        lambda values: float(np.dot(values, weights) / denominator),
        raw=True,
    )


def _kama(series: pd.Series, period: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) < period:
        return pd.Series(result, index=series.index)
    change = np.abs(series - series.shift(period)).to_numpy(dtype=float)
    volatility = series.diff().abs().rolling(period).sum().to_numpy(dtype=float)
    fast, slow = 2.0 / 3.0, 2.0 / 31.0
    result[period - 1] = float(np.nanmean(values[:period]))
    for index in range(period, len(values)):
        efficiency = change[index] / volatility[index] if volatility[index] > 0 else 0.0
        smoothing = (efficiency * (fast - slow) + slow) ** 2
        result[index] = result[index - 1] + smoothing * (
            values[index] - result[index - 1]
        )
    return pd.Series(result, index=series.index)


def _alma(series: pd.Series, period: int) -> pd.Series:
    center = 0.85 * (period - 1)
    width = period / 6.0
    weights = np.exp(-((np.arange(period) - center) ** 2) / (2.0 * width * width))
    weights /= weights.sum()
    return series.rolling(period).apply(
        lambda values: float(np.dot(values, weights)),
        raw=True,
    )


def compute_ma(
    ma_type: str,
    close: pd.Series,
    volume: pd.Series,
    period: int,
) -> pd.Series:
    ma_type, period = ma_type.upper(), int(period)
    if ma_type == "SMA":
        return close.rolling(period).mean()
    if ma_type == "EMA":
        return close.ewm(span=period, adjust=False, min_periods=period).mean()
    if ma_type == "WMA":
        return _wma(close, period)
    if ma_type == "VWMA":
        denominator = volume.rolling(period).sum().replace(0.0, np.nan)
        return (close * volume).rolling(period).sum() / denominator
    if ma_type == "KAMA":
        return _kama(close, period)
    if ma_type == "ALMA":
        return _alma(close, period)
    if ma_type == "HMA":
        half, root = max(2, period // 2), max(2, int(np.sqrt(period)))
        return _wma(2.0 * _wma(close, half) - _wma(close, period), root)
    raise ValueError(f"Bilinmeyen MA türü: {ma_type}")


def trend_state(
    close: pd.Series,
    ma: pd.Series,
    atr: pd.Series,
    slope_bars: int,
    threshold_atr: float,
) -> tuple[str, float, str]:
    valid = pd.DataFrame({"close": close, "ma": ma, "atr": atr}).dropna()
    if len(valid) <= slope_bars:
        return "Yetersiz veri", np.nan, "-"
    latest, earlier = valid.iloc[-1], valid.iloc[-1 - slope_bars]
    atr_value = float(latest["atr"])
    slope = (
        (float(latest["ma"]) - float(earlier["ma"])) / atr_value
        if np.isfinite(atr_value) and atr_value > 0
        else np.nan
    )
    position = "Üstünde" if latest["close"] >= latest["ma"] else "Altında"
    if not np.isfinite(slope):
        state = "Yetersiz veri"
    elif abs(slope) < threshold_atr:
        state = "Yatay"
    elif position == "Üstünde" and slope > 0:
        state = "Yükselen"
    elif position == "Altında" and slope < 0:
        state = "Alçalan"
    else:
        state = "Geçiş"
    return state, float(slope), position


def detect_touches(
    df: pd.DataFrame,
    ma: pd.Series,
    side: int,
    config: ScanConfig,
) -> list[Touch]:
    if side not in {1, -1}:
        raise ValueError("side +1 veya -1 olmalıdır")
    values = ma.reindex(df.index).to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    atr = df["ATR"].to_numpy(dtype=float)
    touches: list[Touch] = []
    was_far, previous_in_zone = False, False
    last_touch = -10**9
    final_position = len(df) - config.max_holding_bars - 2
    for index in range(1, max(1, final_position)):
        if not np.isfinite(values[index]) or not np.isfinite(atr[index]) or atr[index] <= 0:
            was_far, previous_in_zone = False, False
            continue
        signed_distance = (close[index] - values[index]) / atr[index]
        zone = config.touch_zone_atr * atr[index]
        in_zone = low[index] <= values[index] + zone and high[index] >= values[index] - zone
        if not in_zone:
            if side == 1 and signed_distance >= config.separation_atr:
                was_far = True
            elif side == 1 and high[index] < values[index] - zone:
                # A later approach from below is resistance, not support.
                was_far = False
            elif side == -1 and signed_distance <= -config.separation_atr:
                was_far = True
            elif side == -1 and low[index] > values[index] + zone:
                # Mirrored case: do not retain resistance eligibility after a
                # cross to the support side.
                was_far = False
        independent = index - last_touch > config.max_holding_bars
        if in_zone and not previous_in_zone and was_far and independent:
            touches.append(Touch(index, df.index[index], side, float(values[index]), float(atr[index])))
            last_touch, was_far = index, False
        previous_in_zone = in_zone
    return touches


def simulate_trade(df: pd.DataFrame, touch: Touch, config: ScanConfig) -> Trade | None:
    entry_position = touch.position + 1
    if entry_position >= len(df):
        return None
    direction = touch.side
    entry = float(df["Open"].iloc[entry_position])
    if direction == 1:
        initial_stop = min(float(df["Low"].iloc[touch.position]), touch.ma_value) - config.stop_buffer_atr * touch.atr
        risk = entry - initial_stop
    else:
        initial_stop = max(float(df["High"].iloc[touch.position]), touch.ma_value) + config.stop_buffer_atr * touch.atr
        risk = initial_stop - entry
    if not np.isfinite(risk) or risk <= max(entry * 1e-6, 1e-12):
        return None
    stop, best_close = float(initial_stop), entry
    max_favorable = max_adverse = 0.0
    reached_1r = reached_2r = False
    last_position = min(len(df) - 1, entry_position + config.max_holding_bars - 1)
    exit_price, exit_position, exit_reason = float(df["Close"].iloc[last_position]), last_position, "Süre sonu"
    for position in range(entry_position, min(len(df), entry_position + config.max_holding_bars)):
        high, low, close = (
            float(df["High"].iloc[position]),
            float(df["Low"].iloc[position]),
            float(df["Close"].iloc[position]),
        )
        bar_open = float(df["Open"].iloc[position])
        if direction == 1:
            favorable, adverse, stop_hit = high - entry, entry - low, low <= stop
            gap_through_stop = bar_open <= stop
        else:
            favorable, adverse, stop_hit = entry - low, high - entry, high >= stop
            gap_through_stop = bar_open >= stop
        max_favorable, max_adverse = max(max_favorable, favorable), max(max_adverse, adverse)
        if stop_hit:
            # A stale stop is not executable when the bar opens through it.
            exit_price = bar_open if gap_through_stop else stop
            exit_position = position
            exit_reason = "Takip eden stop" if stop != initial_stop else "İlk stop"
            break
        reached_1r, reached_2r = reached_1r or favorable >= risk, reached_2r or favorable >= 2.0 * risk
        if direction == 1:
            best_close = max(best_close, close)
            stop = max(stop, best_close - config.trailing_stop_atr * touch.atr)
        else:
            best_close = min(best_close, close)
            stop = min(stop, best_close + config.trailing_stop_atr * touch.atr)
        exit_price, exit_position = close, position
    gross_r = direction * (exit_price - entry) / risk
    cost_r = (config.roundtrip_cost_bps / 10_000.0) * entry / risk
    return Trade(
        touch.position, entry_position, exit_position, direction, entry,
        initial_stop, float(exit_price), float(gross_r - cost_r),
        bool(reached_1r), bool(reached_2r), float(max_favorable / risk),
        float(max_adverse / risk), exit_reason,
    )


def _random_baseline(df: pd.DataFrame, side: int, config: ScanConfig) -> tuple[float, float]:
    atr = df["ATR"].to_numpy(dtype=float)
    candidates = np.flatnonzero(np.isfinite(atr))
    warmup = max(config.atr_period, max(config.periods))
    candidates = candidates[
        (candidates >= warmup)
        & (candidates < len(df) - config.max_holding_bars - 1)
    ]
    if len(candidates) > 160:
        candidates = candidates[np.linspace(0, len(candidates) - 1, 160, dtype=int)]
    results: list[float] = []
    for position in candidates:
        entry = float(df["Open"].iloc[int(position) + 1])
        atr_value = float(atr[position])
        touch = Touch(int(position), df.index[position], side, entry - side * atr_value, atr_value)
        trade = simulate_trade(df, touch, config)
        if trade is not None:
            results.append(trade.net_r)
    if not results:
        return np.nan, np.nan
    values = np.asarray(results, dtype=float)
    return float(np.median(values)), float(100.0 * np.mean(values > 0))


def _stability(trades: Sequence[Trade]) -> int:
    if len(trades) < 3:
        return 0
    folds = np.array_split(np.asarray([trade.net_r for trade in trades], dtype=float), 3)
    return int(sum(bool(len(fold) and np.median(fold) > 0) for fold in folds))


def compatibility_class(
    touches: int,
    median_r: float,
    edge_r: float,
    win_rate: float,
    baseline_win_rate: float,
    stability: int,
    config: ScanConfig,
) -> str:
    if touches < config.min_touches:
        return "Yetersiz veri"
    beats_win_rate = np.isfinite(baseline_win_rate) and win_rate > baseline_win_rate
    if median_r > 0 and edge_r >= config.min_edge_r and beats_win_rate and stability == 3:
        return "Güçlü uyum"
    if median_r > 0 and edge_r > 0 and stability >= 2:
        return "Uyumlu"
    if median_r > 0 or edge_r > 0:
        return "İzleme"
    return "Uyumsuz"


def scan_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    config: ScanConfig,
    metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    df = prepare_frame(frame, config.atr_period)
    current_price, current_atr = float(df["Close"].iloc[-1]), float(df["ATR"].iloc[-1])
    baselines = {side: _random_baseline(df, side, config) for side in (1, -1)}
    rows: list[dict[str, object]] = []
    metadata = metadata or {}
    for ma_type in config.ma_types:
        for period in config.periods:
            if period + config.trend_slope_bars >= len(df):
                continue
            ma = compute_ma(ma_type, df["Close"], df["Volume"], period)
            if not np.isfinite(ma.iloc[-1]):
                continue
            state, slope_atr, price_position = trend_state(
                df["Close"], ma, df["ATR"],
                config.trend_slope_bars, config.trend_slope_threshold_atr,
            )
            current_ma = float(ma.iloc[-1])
            distance_atr = (current_ma - current_price) / current_atr if current_atr > 0 else np.nan
            for side in (1, -1):
                trades = [
                    trade for touch in detect_touches(df, ma, side, config)
                    if (trade := simulate_trade(df, touch, config)) is not None
                ]
                values = np.asarray([trade.net_r for trade in trades], dtype=float)
                count = len(trades)
                median_r = float(np.median(values)) if count else np.nan
                mean_r = float(np.mean(values)) if count else np.nan
                win_rate = float(100.0 * np.mean(values > 0)) if count else np.nan
                rate_1r = float(100.0 * np.mean([trade.reached_1r for trade in trades])) if count else np.nan
                rate_2r = float(100.0 * np.mean([trade.reached_2r for trade in trades])) if count else np.nan
                median_mfe = float(np.median([trade.mfe_r for trade in trades])) if count else np.nan
                median_mae = float(np.median([trade.mae_r for trade in trades])) if count else np.nan
                stability = _stability(trades)
                baseline_r, baseline_win = baselines[side]
                edge_r = median_r - baseline_r if np.isfinite(median_r) and np.isfinite(baseline_r) else np.nan
                quality = compatibility_class(
                    count, median_r, edge_r, win_rate, baseline_win, stability, config
                )
                active_side = (
                    (side == 1 and current_price >= current_ma)
                    or (side == -1 and current_price <= current_ma)
                )
                rows.append({
                    **metadata,
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "ma_type": ma_type,
                    "period": int(period),
                    "ma": f"{ma_type}{period}",
                    "side": "Destek" if side == 1 else "Direnç",
                    "current_price": current_price,
                    "current_ma": current_ma,
                    "distance_atr": float(distance_atr),
                    "distance_pct": float((current_ma - current_price) / current_price * 100.0),
                    "active_side": bool(active_side),
                    "trend_state": state,
                    "price_position": price_position,
                    "slope_atr": slope_atr,
                    "touches": count,
                    "win_rate_pct": win_rate,
                    "median_net_r": median_r,
                    "mean_net_r": mean_r,
                    "baseline_median_r": baseline_r,
                    "baseline_win_rate_pct": baseline_win,
                    "edge_r": edge_r,
                    "reached_1r_pct": rate_1r,
                    "reached_2r_pct": rate_2r,
                    "median_mfe_r": median_mfe,
                    "median_mae_r": median_mae,
                    "positive_periods": stability,
                    "compatibility": quality,
                })
    return pd.DataFrame(rows)


_QUALITY_RANK = {
    "Güçlü uyum": 4, "Uyumlu": 3, "İzleme": 2,
    "Uyumsuz": 1, "Yetersiz veri": 0,
}


def aggregate_trend(states: Iterable[str]) -> tuple[str, str]:
    usable = [value for value in states if value != "Yetersiz veri"]
    if not usable:
        return "Yetersiz veri", "0/0"
    rising = sum(value == "Yükselen" for value in usable)
    falling = sum(value == "Alçalan" for value in usable)
    if rising / len(usable) >= 0.70:
        label = "Güçlü yükselen"
    elif rising / len(usable) >= 0.50:
        label = "Yükselen"
    elif falling / len(usable) >= 0.70:
        label = "Güçlü alçalan"
    elif falling / len(usable) >= 0.50:
        label = "Alçalan"
    else:
        label = "Karışık/Geçiş"
    return label, f"{rising}↑ {falling}↓ / {len(usable)}"


def build_market_summary(detail: pd.DataFrame, near_distance_atr: float = 1.0) -> pd.DataFrame:
    if detail is None or detail.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    identity = ["asset_class", "symbol"] if "asset_class" in detail else ["symbol"]
    for key, symbol_rows in detail.groupby(identity, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(identity, key))
        first = symbol_rows.iloc[0]
        for column in ("display_name", "market", "sector", "industry", "index_memberships"):
            if column in symbol_rows:
                record[column] = first.get(column, "")
        timeframe_labels: list[str] = []
        for timeframe in TIMEFRAMES:
            tf_rows = symbol_rows[symbol_rows["timeframe"] == timeframe]
            if tf_rows.empty:
                record[f"{timeframe}_trend"], record[f"{timeframe}_votes"] = "-", "-"
                continue
            label, votes = aggregate_trend(
                tf_rows.drop_duplicates(["ma_type", "period"])["trend_state"]
            )
            record[f"{timeframe}_trend"], record[f"{timeframe}_votes"] = label, votes
            timeframe_labels.append(f"{timeframe}:{label}")
        active = symbol_rows[symbol_rows["active_side"].fillna(False)].copy()
        if active.empty:
            active = symbol_rows.copy()
        active["_quality_rank"] = active["compatibility"].map(_QUALITY_RANK).fillna(0)
        active["_abs_distance"] = pd.to_numeric(active["distance_atr"], errors="coerce").abs()
        active = active.sort_values(
            ["_quality_rank", "positive_periods", "edge_r", "median_net_r", "_abs_distance"],
            ascending=[False, False, False, False, True],
            na_position="last",
        )
        best = active.iloc[0]
        support, resistance = active[active["side"] == "Destek"], active[active["side"] == "Direnç"]
        record.update({
            "trend_summary": " | ".join(timeframe_labels),
            "best_timeframe": best["timeframe"],
            "best_ma": best["ma"],
            "best_side": best["side"],
            "best_compatibility": best["compatibility"],
            "best_touches": int(best["touches"]),
            "best_win_rate_pct": best["win_rate_pct"],
            "best_median_net_r": best["median_net_r"],
            "best_edge_r": best["edge_r"],
            "best_distance_atr": best["distance_atr"],
            "best_support": support.iloc[0]["ma"] if not support.empty else "-",
            "best_resistance": resistance.iloc[0]["ma"] if not resistance.empty else "-",
        })
        distance = abs(float(best["distance_atr"])) if np.isfinite(best["distance_atr"]) else np.inf
        compatible = best["compatibility"] in {"Güçlü uyum", "Uyumlu"}
        if compatible and distance <= near_distance_atr:
            record["setup"] = "Desteğe yakın" if best["side"] == "Destek" else "Dirence yakın"
        elif compatible:
            record["setup"] = "Uyumlu MA uzakta"
        elif best["compatibility"] == "İzleme":
            record["setup"] = "İzleme"
        else:
            record["setup"] = "Kurulum yok"
        record["_sort_quality"] = int(_QUALITY_RANK.get(str(best["compatibility"]), 0))
        record["_sort_stability"] = int(best["positive_periods"])
        record["_sort_edge"] = float(best["edge_r"]) if np.isfinite(best["edge_r"]) else -999.0
        record["_sort_distance"] = distance
        records.append(record)
    return (
        pd.DataFrame(records)
        .sort_values(
            ["_sort_quality", "_sort_stability", "_sort_edge", "_sort_distance", "symbol"],
            ascending=[False, False, False, True, True],
        )
        .drop(columns=["_sort_quality", "_sort_stability", "_sort_edge", "_sort_distance"])
        .reset_index(drop=True)
    )
