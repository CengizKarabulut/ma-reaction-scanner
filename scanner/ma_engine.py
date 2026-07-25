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
    quality_lookback: int = 60
    min_price: float = 1.0
    min_daily_turnover_try: float = 1_000_000.0
    max_zero_volume_pct: float = 20.0
    max_gap_pct: float = 15.0
    max_abs_edge_r: float = 5.0
    min_side_adherence_pct: float = 60.0
    min_positive_periods: int = 2
    watch_distance_atr: float = 1.50
    max_cross_rate_per_100: float = 20.0
    rsi_period: int = 14
    rsi_smoothing_period: int = 9
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    smi_period: int = 14
    smi_smoothing_period: int = 3
    smi_signal_period: int = 3
    ichimoku_conversion_period: int = 9
    ichimoku_base_period: int = 26
    ichimoku_span_b_period: int = 52
    bollinger_period: int = 20
    bollinger_stddev: float = 2.0
    indicator_extreme_lookback: int = 100
    extreme_percentile: float = 90.0
    adx_period: int = 14
    min_adx: float = 20.0
    volume_lookback: int = 20
    min_relative_volume: float = 1.20
    use_volume_confirmation: bool = True
    use_adx_confirmation: bool = True

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
            self.quality_lookback,
            self.min_positive_periods,
            self.rsi_period,
            self.rsi_smoothing_period,
            self.macd_fast_period,
            self.macd_slow_period,
            self.macd_signal_period,
            self.smi_period,
            self.smi_smoothing_period,
            self.smi_signal_period,
            self.ichimoku_conversion_period,
            self.ichimoku_base_period,
            self.ichimoku_span_b_period,
            self.bollinger_period,
            self.indicator_extreme_lookback,
            self.adx_period,
            self.volume_lookback,
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
            self.min_price,
            self.min_daily_turnover_try,
            self.max_zero_volume_pct,
            self.max_gap_pct,
            self.max_abs_edge_r,
            self.min_side_adherence_pct,
            self.watch_distance_atr,
            self.max_cross_rate_per_100,
            self.min_adx,
            self.min_relative_volume,
            self.bollinger_stddev,
            self.extreme_percentile,
        )
        if any(float(value) < 0 for value in non_negative):
            raise ValueError("ATR, maliyet ve eşik ayarları negatif olamaz")
        if self.max_zero_volume_pct > 100 or self.min_side_adherence_pct > 100:
            raise ValueError("Yüzde eşikleri 0-100 arasında olmalıdır")
        if self.macd_fast_period >= self.macd_slow_period:
            raise ValueError("MACD hızlı periyodu yavaş periyottan küçük olmalıdır")
        if not 50.0 <= self.extreme_percentile <= 100.0:
            raise ValueError("Aşırılık yüzdeliği 50-100 arasında olmalıdır")
        if self.bollinger_stddev == 0:
            raise ValueError("Bollinger standart sapma çarpanı sıfır olamaz")

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


def normalize_ohlcv(
    frame: pd.DataFrame,
    *,
    allow_non_positive: bool = False,
) -> pd.DataFrame:
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
    prices = out[["Open", "High", "Low", "Close"]]
    if (prices == 0).any().any():
        raise ValueError("OHLC fiyatları sıfır olamaz")
    if not allow_non_positive and (prices < 0).any().any():
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


def prepare_frame(
    frame: pd.DataFrame,
    atr_period: int = 14,
    *,
    allow_non_positive: bool = False,
) -> pd.DataFrame:
    df = normalize_ohlcv(frame, allow_non_positive=allow_non_positive)
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
    if not np.isfinite(risk) or risk <= max(abs(entry) * 1e-6, 1e-12):
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
        gap_through_stop = (
            (direction == 1 and bar_open <= stop)
            or (direction == -1 and bar_open >= stop)
        )
        if gap_through_stop:
            # The opening gap occurred before execution and belongs in MAE.
            gap_adverse = entry - bar_open if direction == 1 else bar_open - entry
            max_adverse = max(max_adverse, gap_adverse)
            # Exit at the open without consuming post-exit intrabar range.
            exit_price, exit_position = bar_open, position
            exit_reason = "Takip eden stop" if stop != initial_stop else "\u0130lk stop"
            break
        if direction == 1:
            favorable, adverse, stop_hit = high - entry, entry - low, low <= stop
        else:
            favorable, adverse, stop_hit = entry - low, high - entry, high >= stop
        max_favorable, max_adverse = max(max_favorable, favorable), max(max_adverse, adverse)
        if stop_hit:
            exit_price, exit_position = stop, position
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
    cost_r = (config.roundtrip_cost_bps / 10_000.0) * abs(entry) / risk
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


def compatibility_score(
    touches: int,
    side_adherence_pct: float,
    win_rate_pct: float,
    median_r: float,
    edge_r: float,
    stability: int,
    config: ScanConfig,
    cross_rate_per_100: float = 0.0,
    compatibility: str = "Güçlü uyum",
) -> float:
    """Return a 0-100 score whose ceiling reflects evidence quality."""
    def bounded(value: float, low: float, high: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, low, high))

    evidence = min(max(touches, 0) / max(config.min_touches, 1), 1.0) * 20.0
    adherence = bounded(side_adherence_pct, 0.0, 100.0) / 100.0 * 15.0
    win_quality = bounded(win_rate_pct, 0.0, 100.0) / 100.0 * 15.0
    median_quality = bounded(median_r, 0.0, 1.0) * 15.0
    edge_quality = bounded(edge_r, 0.0, 1.0) * 15.0
    stability_quality = bounded(float(stability), 0.0, 3.0) / 3.0 * 10.0
    noise_ratio = bounded(
        cross_rate_per_100 / max(config.max_cross_rate_per_100, 1e-9),
        0.0,
        1.0,
    )
    noise_quality = (1.0 - noise_ratio) * 10.0 if touches > 0 else 0.0
    raw_score = (
        evidence + adherence + win_quality + median_quality
        + edge_quality + stability_quality + noise_quality
    )
    ceilings = {
        "Yetersiz veri": 39.0,
        "Uyumsuz": 49.0,
        "İzleme": 59.0,
        "Uyumlu": 79.0,
        "Güçlü uyum": 100.0,
    }
    return round(min(raw_score, ceilings.get(compatibility, 49.0)), 2)


def confirmation_indicators(
    df: pd.DataFrame,
    config: ScanConfig,
) -> dict[str, float | bool | str]:
    """Return current confirmation values without blending them into MA history."""
    close, high, low = df["Close"], df["High"], df["Low"]

    def latest(series: pd.Series) -> float:
        value = series.iloc[-1]
        return float(value) if np.isfinite(value) else np.nan

    def percentile_rank(series: pd.Series, *, absolute: bool = False) -> float:
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.abs() if absolute else values
        values = values.dropna().tail(config.indicator_extreme_lookback)
        if values.empty:
            return np.nan
        return float(100.0 * (values <= values.iloc[-1]).mean())

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(
        alpha=1.0 / config.rsi_period,
        adjust=False,
        min_periods=config.rsi_period,
    ).mean()
    avg_loss = losses.ewm(
        alpha=1.0 / config.rsi_period,
        adjust=False,
        min_periods=config.rsi_period,
    ).mean()
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi_smoothing = rsi.ewm(
        span=config.rsi_smoothing_period,
        adjust=False,
        min_periods=config.rsi_smoothing_period,
    ).mean()
    rsi_value, rsi_smoothing_value = latest(rsi), latest(rsi_smoothing)
    if np.isfinite(rsi_value) and np.isfinite(rsi_smoothing_value):
        rsi_relation = "yumuşatma üstünde" if rsi_value >= rsi_smoothing_value else "yumuşatma altında"
        if rsi_value >= 70:
            rsi_status = f"Aşırı alım; {rsi_relation}"
        elif rsi_value <= 30:
            rsi_status = f"Aşırı satım; {rsi_relation}"
        else:
            rsi_status = rsi_relation.capitalize()
    else:
        rsi_status = "Yetersiz veri"

    macd_line = close.ewm(
        span=config.macd_fast_period, adjust=False,
        min_periods=config.macd_fast_period,
    ).mean() - close.ewm(
        span=config.macd_slow_period, adjust=False,
        min_periods=config.macd_slow_period,
    ).mean()
    macd_signal = macd_line.ewm(
        span=config.macd_signal_period,
        adjust=False,
        min_periods=config.macd_signal_period,
    ).mean()
    macd_histogram = macd_line - macd_signal
    macd_value, macd_signal_value = latest(macd_line), latest(macd_signal)
    macd_gap = latest(macd_histogram)
    macd_gap_percentile = percentile_rank(macd_histogram, absolute=True)
    macd_stretched = bool(np.isfinite(macd_gap_percentile) and macd_gap_percentile >= config.extreme_percentile)
    atr_value = latest(df["ATR"])
    macd_gap_atr = macd_gap / atr_value if np.isfinite(atr_value) and atr_value > 0 else np.nan
    current_macd_above = bool(np.isfinite(macd_value) and np.isfinite(macd_signal_value) and macd_value >= macd_signal_value)
    previous_macd_above = bool(
        len(macd_line) > 1
        and np.isfinite(macd_line.iloc[-2])
        and np.isfinite(macd_signal.iloc[-2])
        and macd_line.iloc[-2] >= macd_signal.iloc[-2]
    )
    if current_macd_above and not previous_macd_above:
        macd_status = "Yukarı kesişim"
    elif not current_macd_above and previous_macd_above:
        macd_status = "Aşağı kesişim"
    elif current_macd_above:
        macd_status = "MACD Signal üstünde"
    elif np.isfinite(macd_value) and np.isfinite(macd_signal_value):
        macd_status = "MACD Signal altında"
    else:
        macd_status = "Yetersiz veri"
    if macd_stretched and np.isfinite(macd_gap):
        stretch_direction = "Pozitif" if macd_gap >= 0 else "Negatif"
        macd_status = f"{macd_status}; {stretch_direction} açılma yüksek"

    highest = high.rolling(config.smi_period, min_periods=config.smi_period).max()
    lowest = low.rolling(config.smi_period, min_periods=config.smi_period).min()
    midpoint_distance = close - (highest + lowest) / 2.0
    half_range = (highest - lowest) / 2.0
    smooth_distance = midpoint_distance.ewm(
        span=config.smi_smoothing_period, adjust=False,
        min_periods=config.smi_smoothing_period,
    ).mean().ewm(
        span=config.smi_smoothing_period, adjust=False,
        min_periods=config.smi_smoothing_period,
    ).mean()
    smooth_range = half_range.ewm(
        span=config.smi_smoothing_period, adjust=False,
        min_periods=config.smi_smoothing_period,
    ).mean().ewm(
        span=config.smi_smoothing_period, adjust=False,
        min_periods=config.smi_smoothing_period,
    ).mean()
    smi = 100.0 * smooth_distance / smooth_range.replace(0.0, np.nan)
    smi_signal = smi.ewm(
        span=config.smi_signal_period, adjust=False,
        min_periods=config.smi_signal_period,
    ).mean()
    smi_value, smi_signal_value = latest(smi), latest(smi_signal)
    current_smi_above = bool(
        np.isfinite(smi_value) and np.isfinite(smi_signal_value)
        and smi_value >= smi_signal_value
    )
    previous_smi_above = bool(
        len(smi) > 1 and np.isfinite(smi.iloc[-2])
        and np.isfinite(smi_signal.iloc[-2]) and smi.iloc[-2] >= smi_signal.iloc[-2]
    )
    if current_smi_above and not previous_smi_above:
        smi_relation = "Yukarı kesişim"
    elif not current_smi_above and previous_smi_above:
        smi_relation = "Aşağı kesişim"
    elif current_smi_above:
        smi_relation = "SMI Signal üstünde"
    elif np.isfinite(smi_value) and np.isfinite(smi_signal_value):
        smi_relation = "SMI Signal altında"
    else:
        smi_relation = "Yetersiz veri"
    if np.isfinite(smi_value) and smi_value >= 40:
        smi_status = f"{smi_relation}; aşırı alım bölgesi"
    elif np.isfinite(smi_value) and smi_value <= -40:
        smi_status = f"{smi_relation}; aşırı satım bölgesi"
    else:
        smi_status = smi_relation

    conversion = (
        high.rolling(config.ichimoku_conversion_period).max()
        + low.rolling(config.ichimoku_conversion_period).min()
    ) / 2.0
    base = (
        high.rolling(config.ichimoku_base_period).max()
        + low.rolling(config.ichimoku_base_period).min()
    ) / 2.0
    span_a = ((conversion + base) / 2.0).shift(config.ichimoku_base_period)
    span_b = (
        (
            high.rolling(config.ichimoku_span_b_period).max()
            + low.rolling(config.ichimoku_span_b_period).min()
        ) / 2.0
    ).shift(config.ichimoku_base_period)
    conversion_value, base_value = latest(conversion), latest(base)
    span_a_value, span_b_value = latest(span_a), latest(span_b)
    cloud_top = max(span_a_value, span_b_value) if np.isfinite(span_a_value) and np.isfinite(span_b_value) else np.nan
    cloud_bottom = min(span_a_value, span_b_value) if np.isfinite(span_a_value) and np.isfinite(span_b_value) else np.nan
    current_close = latest(close)
    if np.isfinite(cloud_top) and current_close > cloud_top:
        cloud_position = "Bulut üstünde"
    elif np.isfinite(cloud_bottom) and current_close < cloud_bottom:
        cloud_position = "Bulut altında"
    elif np.isfinite(cloud_top):
        cloud_position = "Bulut içinde"
    else:
        cloud_position = "Yetersiz veri"
    line_relation = (
        "Tenkan Kijun üstünde"
        if np.isfinite(conversion_value) and np.isfinite(base_value) and conversion_value >= base_value
        else "Tenkan Kijun altında"
        if np.isfinite(conversion_value) and np.isfinite(base_value)
        else "Çizgi verisi yok"
    )
    ichimoku_status = f"{cloud_position}; {line_relation}"

    bollinger_mid = close.rolling(
        config.bollinger_period, min_periods=config.bollinger_period
    ).mean()
    bollinger_std = close.rolling(
        config.bollinger_period, min_periods=config.bollinger_period
    ).std(ddof=0)
    bollinger_upper = bollinger_mid + config.bollinger_stddev * bollinger_std
    bollinger_lower = bollinger_mid - config.bollinger_stddev * bollinger_std
    bb_mid, bb_upper, bb_lower = latest(bollinger_mid), latest(bollinger_upper), latest(bollinger_lower)
    bollinger_width_series = (bollinger_upper - bollinger_lower) / bollinger_mid.abs() * 100.0
    bb_width_percentile = percentile_rank(bollinger_width_series)
    bb_width_pct = (
        (bb_upper - bb_lower) / abs(bb_mid) * 100.0
        if np.isfinite(bb_mid) and bb_mid != 0
        else np.nan
    )
    bb_percent_b = (
        (current_close - bb_lower) / (bb_upper - bb_lower) * 100.0
        if np.isfinite(bb_upper) and np.isfinite(bb_lower) and bb_upper != bb_lower
        else np.nan
    )
    if np.isfinite(bb_upper) and current_close > bb_upper:
        bollinger_status = "Üst bant dışında"
    elif np.isfinite(bb_lower) and current_close < bb_lower:
        bollinger_status = "Alt bant dışında"
    elif np.isfinite(bb_mid) and current_close >= bb_mid:
        bollinger_status = "Orta-üst bölgede"
    elif np.isfinite(bb_mid):
        bollinger_status = "Alt-orta bölgede"
    else:
        bollinger_status = "Yetersiz veri"
    if np.isfinite(bb_width_percentile):
        if bb_width_percentile >= config.extreme_percentile:
            bollinger_status = f"{bollinger_status}; bant genişliği yüksek"
        elif bb_width_percentile <= 100.0 - config.extreme_percentile:
            bollinger_status = f"{bollinger_status}; sıkışma"

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    smoothed_tr = true_range(df).ewm(
        alpha=1.0 / config.adx_period,
        adjust=False,
        min_periods=config.adx_period,
    ).mean()
    plus_di = 100.0 * plus_dm.ewm(
        alpha=1.0 / config.adx_period,
        adjust=False,
        min_periods=config.adx_period,
    ).mean() / smoothed_tr
    minus_di = 100.0 * minus_dm.ewm(
        alpha=1.0 / config.adx_period,
        adjust=False,
        min_periods=config.adx_period,
    ).mean() / smoothed_tr
    denominator = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    adx = dx.ewm(
        alpha=1.0 / config.adx_period,
        adjust=False,
        min_periods=config.adx_period,
    ).mean()

    volume = pd.to_numeric(df["Volume"], errors="coerce")
    volume_baseline = volume.rolling(
        config.volume_lookback,
        min_periods=config.volume_lookback,
    ).median().shift(1)
    baseline = latest(volume_baseline)
    current_volume = latest(volume)
    volume_available = bool(np.isfinite(baseline) and baseline > 0)
    relative_volume = current_volume / baseline if volume_available else np.nan
    return {
        "rsi": rsi_value,
        "rsi_smoothing": rsi_smoothing_value,
        "rsi_status": rsi_status,
        "macd": macd_value,
        "macd_signal": macd_signal_value,
        "macd_histogram": macd_gap,
        "macd_gap_atr": float(macd_gap_atr),
        "macd_gap_percentile": float(macd_gap_percentile),
        "macd_stretched": macd_stretched,
        "macd_status": macd_status,
        "smi": smi_value,
        "smi_signal": smi_signal_value,
        "smi_status": smi_status,
        "ichimoku_conversion": conversion_value,
        "ichimoku_base": base_value,
        "ichimoku_span_a": span_a_value,
        "ichimoku_span_b": span_b_value,
        "ichimoku_status": ichimoku_status,
        "bollinger_mid": bb_mid,
        "bollinger_upper": bb_upper,
        "bollinger_lower": bb_lower,
        "bollinger_percent_b": float(bb_percent_b),
        "bollinger_width_pct": float(bb_width_pct),
        "bollinger_width_percentile": float(bb_width_percentile),
        "bollinger_status": bollinger_status,
        "adx": latest(adx),
        "relative_volume": float(relative_volume),
        "volume_available": volume_available,
    }


def decision_state(
    *,
    touches: int,
    median_r: float,
    edge_r: float,
    stability: int,
    side_adherence_pct: float,
    cross_rate_per_100: float,
    side: int,
    trend_state_value: str,
    active_side: bool,
    distance_atr: float,
    price_trigger: bool,
    relative_volume: float,
    volume_available: bool,
    adx: float,
    filter_reasons: Sequence[str],
    config: ScanConfig,
) -> tuple[str, str, bool, bool, bool]:
    volume_confirmed = bool(
        not config.use_volume_confirmation
        or not volume_available
        or (
            np.isfinite(relative_volume)
            and relative_volume >= config.min_relative_volume
        )
    )
    adx_confirmed = bool(
        not config.use_adx_confirmation
        or (np.isfinite(adx) and adx >= config.min_adx)
    )
    if filter_reasons:
        return "Filtre Dışı", ", ".join(filter_reasons), False, volume_confirmed, adx_confirmed
    if touches < config.min_touches:
        return (
            "Yetersiz Veri",
            f"Temas {touches}/{config.min_touches}",
            False,
            volume_confirmed,
            adx_confirmed,
        )

    quality_reasons: list[str] = []
    if not np.isfinite(median_r) or median_r <= 0:
        quality_reasons.append("Medyan R pozitif değil")
    if not np.isfinite(edge_r) or edge_r < config.min_edge_r:
        quality_reasons.append("Edge eşik altında")
    if stability < config.min_positive_periods:
        quality_reasons.append(
            f"Pozitif dönem {stability}/{config.min_positive_periods}"
        )
    if (
        not np.isfinite(side_adherence_pct)
        or side_adherence_pct < config.min_side_adherence_pct
    ):
        quality_reasons.append("Taraf koruma düşük")
    if (
        np.isfinite(cross_rate_per_100)
        and cross_rate_per_100 > config.max_cross_rate_per_100
    ):
        quality_reasons.append("Gürültü yüksek")
    trend_aligned = (
        (side == 1 and trend_state_value == "Yükselen")
        or (side == -1 and trend_state_value == "Alçalan")
    )
    if not trend_aligned:
        quality_reasons.append("Trend yönü uyumsuz")
    quality_pass = not quality_reasons
    if not active_side:
        return "İşlem Yok", "Güncel rol aktif değil", quality_pass, volume_confirmed, adx_confirmed
    if not quality_pass:
        return "Uyumsuz", "; ".join(quality_reasons), False, volume_confirmed, adx_confirmed
    if not np.isfinite(distance_atr):
        return "İşlem Yok", "ATR uzaklığı hesaplanamadı", True, volume_confirmed, adx_confirmed

    absolute_distance = abs(distance_atr)
    if absolute_distance > config.watch_distance_atr:
        return "Uzak", f"{absolute_distance:.2f} ATR uzakta", True, volume_confirmed, adx_confirmed
    if absolute_distance > config.touch_zone_atr:
        return "Yaklaşıyor", f"{absolute_distance:.2f} ATR uzakta", True, volume_confirmed, adx_confirmed
    if not price_trigger:
        return "Tetik Bekliyor", "Fiyat teyidi yok", True, volume_confirmed, adx_confirmed
    confirmations = []
    if not volume_confirmed:
        confirmations.append("RVOL düşük")
    if not adx_confirmed:
        confirmations.append("ADX düşük")
    if confirmations:
        return "Tetik Bekliyor", "; ".join(confirmations), True, volume_confirmed, adx_confirmed
    return "Güçlü Aday", "Kalite, temas ve teyit uygun", True, volume_confirmed, adx_confirmed

def market_quality_metrics(
    df: pd.DataFrame,
    timeframe: str,
    config: ScanConfig,
) -> dict[str, float]:
    recent = df.tail(config.quality_lookback)
    turnover = pd.to_numeric(df["Close"] * df["Volume"], errors="coerce")
    if timeframe in {"5m", "15m", "30m", "1h", "4h"}:
        daily_turnover = turnover.groupby(df.index.normalize()).sum(min_count=1)
    elif timeframe == "1wk":
        daily_turnover = turnover / 5.0
    elif timeframe == "1mo":
        daily_turnover = turnover / 21.0
    else:
        daily_turnover = turnover
    median_daily_turnover = float(
        daily_turnover.dropna().tail(config.quality_lookback).median()
    )
    recent_volume = pd.to_numeric(recent["Volume"], errors="coerce")
    zero_volume_pct = float(100.0 * (recent_volume <= 0).mean())
    recent_gap = (
        (df["Open"] / df["Close"].shift(1) - 1.0)
        .abs()
        .mul(100.0)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tail(config.quality_lookback)
    )
    max_recent_gap_pct = float(recent_gap.max()) if not recent_gap.empty else 0.0
    return {
        "median_daily_turnover_try": median_daily_turnover,
        "zero_volume_pct": zero_volume_pct,
        "max_recent_gap_pct": max_recent_gap_pct,
    }


def quality_filter_reasons(
    *,
    asset_class: str,
    market: str,
    price: float,
    metrics: dict[str, float],
    edge_r: float,
    config: ScanConfig,
) -> list[str]:
    reasons: list[str] = []
    if asset_class != "stock":
        return reasons
    if market == "BIST":
        if price < config.min_price:
            reasons.append("Fiyat")
        turnover = metrics["median_daily_turnover_try"]
        if not np.isfinite(turnover) or turnover < config.min_daily_turnover_try:
            reasons.append("Likidite")
    if metrics["zero_volume_pct"] > config.max_zero_volume_pct:
        reasons.append("Sifir hacim")
    if metrics["max_recent_gap_pct"] > config.max_gap_pct:
        reasons.append("Gap")
    if np.isfinite(edge_r) and abs(edge_r) > config.max_abs_edge_r:
        reasons.append("Aykiri Edge")
    return reasons

def scan_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    config: ScanConfig,
    metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    metadata = metadata or {}
    df = prepare_frame(
        frame,
        config.atr_period,
        allow_non_positive=metadata.get("asset_class") == "commodity",
    )
    current_price, current_atr = float(df["Close"].iloc[-1]), float(df["ATR"].iloc[-1])
    latest_time = pd.Timestamp(df.index[-1])
    if latest_time.tzinfo is None:
        latest_time = latest_time.tz_localize("Europe/Istanbul")
    else:
        latest_time = latest_time.tz_convert("Europe/Istanbul")
    price_time = latest_time.isoformat()
    quality_metrics = market_quality_metrics(df, timeframe, config)
    indicator_metrics = confirmation_indicators(df, config)
    latest_open = float(df["Open"].iloc[-1])
    latest_high = float(df["High"].iloc[-1])
    latest_low = float(df["Low"].iloc[-1])
    baselines = {side: _random_baseline(df, side, config) for side in (1, -1)}
    rows: list[dict[str, object]] = []
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
            relative = (df["Close"] - ma).dropna()
            above_ma_pct = float(100.0 * (relative >= 0).mean()) if not relative.empty else np.nan
            below_ma_pct = float(100.0 * (relative < 0).mean()) if not relative.empty else np.nan
            signs = np.sign(relative).replace(0, np.nan).ffill()
            cross_count = int((signs.diff().abs() == 2).sum())
            cross_rate_per_100 = cross_count / max(len(relative) - 1, 1) * 100.0
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
                side_adherence = above_ma_pct if side == 1 else below_ma_pct
                score = compatibility_score(
                    count,
                    side_adherence,
                    win_rate,
                    median_r,
                    edge_r,
                    stability,
                    config,
                    cross_rate_per_100,
                    quality,
                )
                filter_reasons = quality_filter_reasons(
                    asset_class=str(metadata.get("asset_class", "")),
                    market=str(metadata.get("market", "")),
                    price=current_price,
                    metrics=quality_metrics,
                    edge_r=edge_r,
                    config=config,
                )
                active_side = (
                    (side == 1 and current_price >= current_ma)
                    or (side == -1 and current_price <= current_ma)
                )
                price_trigger = bool(
                    (
                        side == 1
                        and latest_low <= current_ma + config.touch_zone_atr * current_atr
                        and current_price >= current_ma
                        and current_price > latest_open
                    )
                    or (
                        side == -1
                        and latest_high >= current_ma - config.touch_zone_atr * current_atr
                        and current_price <= current_ma
                        and current_price < latest_open
                    )
                )
                decision, decision_reason, quality_pass, volume_confirmed, adx_confirmed = decision_state(
                    touches=count,
                    median_r=median_r,
                    edge_r=edge_r,
                    stability=stability,
                    side_adherence_pct=side_adherence,
                    cross_rate_per_100=cross_rate_per_100,
                    side=side,
                    trend_state_value=state,
                    active_side=active_side,
                    distance_atr=distance_atr,
                    price_trigger=price_trigger,
                    relative_volume=float(indicator_metrics["relative_volume"]),
                    volume_available=bool(indicator_metrics["volume_available"]),
                    adx=float(indicator_metrics["adx"]),
                    filter_reasons=filter_reasons,
                    config=config,
                )
                rows.append({
                    **metadata,
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "ma_type": ma_type,
                    "period": int(period),
                    "ma": f"{ma_type}{period}",
                    "side": "Destek" if side == 1 else "Direnç",
                    "price_time": price_time,
                    "current_price": current_price,
                    "current_ma": current_ma,
                    "distance_value": float(current_ma - current_price),
                    "distance_atr": float(distance_atr),
                    "distance_pct": float((current_ma - current_price) / current_price * 100.0),
                    "above_ma_pct": above_ma_pct,
                    "below_ma_pct": below_ma_pct,
                    "side_adherence_pct": above_ma_pct if side == 1 else below_ma_pct,
                    "wrong_side_pct": below_ma_pct if side == 1 else above_ma_pct,
                    "cross_count": cross_count,
                    "cross_rate_per_100": cross_rate_per_100,
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
                    "compatibility_score": score,
                    **indicator_metrics,
                    "price_trigger": price_trigger,
                    "volume_confirmed": volume_confirmed,
                    "adx_confirmed": adx_confirmed,
                    "quality_pass": quality_pass,
                    "decision": decision,
                    "decision_reason": decision_reason,
                    **quality_metrics,
                    "filter_pass": not filter_reasons,
                    "filter_status": "Uygun" if not filter_reasons else "Filtre disi",
                    "filter_reasons": ", ".join(filter_reasons),
                })
    return pd.DataFrame(rows)


_QUALITY_RANK = {
    "Güçlü uyum": 4, "Uyumlu": 3, "İzleme": 2,
    "Uyumsuz": 1, "Yetersiz veri": 0,
}
_DECISION_RANK = {
    "Güçlü Aday": 7, "Tetik Bekliyor": 6, "Yaklaşıyor": 5,
    "Uzak": 4, "İşlem Yok": 3, "Uyumsuz": 2,
    "Yetersiz Veri": 1, "Filtre Dışı": 0,
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
        if "filter_pass" not in active:
            active["filter_pass"] = True
        eligible = active[active["filter_pass"].fillna(False)].copy()
        if not eligible.empty:
            active = eligible
        if "quality_pass" in active:
            quality_candidates = active[active["quality_pass"].fillna(False)]
            if not quality_candidates.empty:
                active = quality_candidates.copy()
        active["_decision_rank"] = active.get(
            "decision", pd.Series("İşlem Yok", index=active.index)
        ).map(_DECISION_RANK).fillna(0)
        active["_quality_rank"] = active["compatibility"].map(_QUALITY_RANK).fillna(0)
        if "compatibility_score" not in active:
            active["compatibility_score"] = active["_quality_rank"] * 20.0
        active["_abs_distance"] = pd.to_numeric(active["distance_atr"], errors="coerce").abs()
        active = active.sort_values(
            [
                "_decision_rank",
                "compatibility_score",
                "touches",
                "_quality_rank",
                "positive_periods",
                "edge_r",
                "median_net_r",
                "_abs_distance",
            ],
            ascending=[False, False, False, False, False, False, False, True],
            na_position="last",
        )
        best = active.iloc[0]
        support, resistance = active[active["side"] == "Destek"], active[active["side"] == "Direnç"]
        record.update({
            "trend_summary": " | ".join(timeframe_labels),
            "best_timeframe": best["timeframe"],
            "price_time": best.get("price_time", ""),
            "current_price": best.get("current_price", np.nan),
            "best_ma": best["ma"],
            "best_ma_value": best.get("current_ma", np.nan),
            "best_difference": best.get("distance_value", np.nan),
            "best_distance_pct": best.get("distance_pct", np.nan),
            "best_side": best["side"],
            "best_compatibility": best["compatibility"],
            "best_touches": int(best["touches"]),
            "best_side_adherence_pct": best.get("side_adherence_pct", np.nan),
            "best_compatibility_score": best.get("compatibility_score", np.nan),
            "best_decision": best.get("decision", "İşlem Yok"),
            "best_decision_reason": best.get("decision_reason", ""),
            "best_quality_pass": best.get("quality_pass", False),
            "best_cross_rate_per_100": best.get("cross_rate_per_100", np.nan),
            "best_relative_volume": best.get("relative_volume", np.nan),
            "best_adx": best.get("adx", np.nan),
            "best_rsi": best.get("rsi", np.nan),
            "best_rsi_smoothing": best.get("rsi_smoothing", np.nan),
            "best_rsi_status": best.get("rsi_status", ""),
            "best_macd": best.get("macd", np.nan),
            "best_macd_signal": best.get("macd_signal", np.nan),
            "best_macd_gap_atr": best.get("macd_gap_atr", np.nan),
            "best_macd_gap_percentile": best.get("macd_gap_percentile", np.nan),
            "best_macd_stretched": best.get("macd_stretched", False),
            "best_macd_status": best.get("macd_status", ""),
            "best_smi": best.get("smi", np.nan),
            "best_smi_signal": best.get("smi_signal", np.nan),
            "best_smi_status": best.get("smi_status", ""),
            "best_ichimoku_status": best.get("ichimoku_status", ""),
            "best_bollinger_percent_b": best.get("bollinger_percent_b", np.nan),
            "best_bollinger_width_pct": best.get("bollinger_width_pct", np.nan),
            "best_bollinger_width_percentile": best.get("bollinger_width_percentile", np.nan),
            "best_bollinger_status": best.get("bollinger_status", ""),
            "best_price_trigger": best.get("price_trigger", False),
            "best_volume_confirmed": best.get("volume_confirmed", False),
            "best_adx_confirmed": best.get("adx_confirmed", False),
            "best_win_rate_pct": best["win_rate_pct"],
            "best_median_net_r": best["median_net_r"],
            "best_edge_r": best["edge_r"],
            "best_distance_atr": best["distance_atr"],
            "median_daily_turnover_try": best.get("median_daily_turnover_try", np.nan),
            "zero_volume_pct": best.get("zero_volume_pct", np.nan),
            "max_recent_gap_pct": best.get("max_recent_gap_pct", np.nan),
            "filter_status": best.get("filter_status", "Uygun"),
            "filter_reasons": best.get("filter_reasons", ""),
            "best_support": support.iloc[0]["ma"] if not support.empty else "-",
            "best_support_value": (
                support.iloc[0].get("current_ma", np.nan) if not support.empty else np.nan
            ),
            "best_resistance": resistance.iloc[0]["ma"] if not resistance.empty else "-",
            "best_resistance_value": (
                resistance.iloc[0].get("current_ma", np.nan)
                if not resistance.empty else np.nan
            ),
        })
        distance = abs(float(best["distance_atr"])) if np.isfinite(best["distance_atr"]) else np.inf
        decision_label = str(best.get("decision", "İşlem Yok"))
        if decision_label == "Güçlü Aday":
            record["setup"] = (
                "Güçlü destek adayı" if best["side"] == "Destek"
                else "Güçlü direnç adayı"
            )
        else:
            record["setup"] = decision_label
        record["_sort_filter"] = int(bool(best.get("filter_pass", True)))
        record["_sort_decision"] = int(_DECISION_RANK.get(str(best.get("decision", "")), 0))
        record["_sort_score"] = float(best.get("compatibility_score", 0.0))
        record["_sort_quality"] = int(_QUALITY_RANK.get(str(best["compatibility"]), 0))
        record["_sort_stability"] = int(best["positive_periods"])
        record["_sort_edge"] = float(best["edge_r"]) if np.isfinite(best["edge_r"]) else -999.0
        record["_sort_distance"] = distance
        records.append(record)
    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "_sort_filter",
                "_sort_decision",
                "_sort_score",
                "_sort_quality",
                "_sort_stability",
                "_sort_edge",
                "_sort_distance",
                "symbol",
            ],
            ascending=[False, False, False, False, False, False, True, True],
        )
        .drop(
            columns=[
                "_sort_filter",
                "_sort_decision",
                "_sort_score",
                "_sort_quality",
                "_sort_stability",
                "_sort_edge",
                "_sort_distance",
            ]
        )
        .reset_index(drop=True)
    )
