import unittest

import numpy as np
import pandas as pd

from scanner.ma_engine import (
    MA_TYPES,
    ScanConfig,
    Touch,
    aggregate_trend,
    build_market_summary,
    compute_ma,
    detect_touches,
    prepare_frame,
    simulate_trade,
    trend_state,
)


def sample_frame(length: int = 500, monotonic: bool = False) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=length, freq="D")
    if monotonic:
        wave = np.linspace(100.0, 200.0, length)
    else:
        wave = 100.0 + np.linspace(0, 25, length) + 5.0 * np.sin(np.arange(length) / 9.0)
    return pd.DataFrame(
        {
            "Open": wave,
            "High": wave + 1.25,
            "Low": wave - 1.0,
            "Close": wave + 0.25,
            "Volume": np.linspace(1_000, 2_000, length),
        },
        index=index,
    )


class MovingAverageEngineTests(unittest.TestCase):
    def test_all_seven_ma_types_compute(self):
        frame = sample_frame()
        for ma_type in MA_TYPES:
            result = compute_ma(ma_type, frame["Close"], frame["Volume"], 20)
            self.assertTrue(np.isfinite(result.iloc[-1]), ma_type)

    def test_trend_requires_price_position_and_slope(self):
        frame = prepare_frame(sample_frame(monotonic=True))
        ma = compute_ma("EMA", frame["Close"], frame["Volume"], 20)
        state, slope, position = trend_state(frame["Close"], ma, frame["ATR"], 10, 0.05)
        self.assertEqual(state, "Yükselen")
        self.assertGreater(slope, 0)
        self.assertEqual(position, "Üstünde")

    def test_trade_enters_at_next_open(self):
        frame = prepare_frame(sample_frame(100))
        config = ScanConfig(
            ma_types=("EMA",),
            periods=(20,),
            max_holding_bars=5,
            trailing_stop_atr=2.0,
            roundtrip_cost_bps=25.0,
        )
        position = 50
        touch = Touch(
            position,
            frame.index[position],
            1,
            float(frame["Close"].iloc[position]),
            float(frame["ATR"].iloc[position]),
        )
        trade = simulate_trade(frame, touch, config)
        self.assertIsNotNone(trade)
        self.assertEqual(trade.entry_position, position + 1)
        self.assertEqual(trade.entry, float(frame["Open"].iloc[position + 1]))

    def test_stop_gap_fills_at_bar_open(self):
        index = pd.date_range("2024-01-01", periods=4, freq="D")
        frame = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 95.0, 95.0],
                "High": [101.0, 101.0, 96.0, 96.0],
                "Low": [99.0, 99.5, 94.0, 94.0],
                "Close": [100.0, 100.5, 95.0, 95.0],
                "ATR": [1.0] * 4,
            },
            index=index,
        )
        config = ScanConfig(max_holding_bars=3, trailing_stop_atr=10.0)
        trade = simulate_trade(frame, Touch(0, index[0], 1, 100.0, 1.0), config)
        self.assertIsNotNone(trade)
        self.assertEqual(trade.exit_position, 2)
        self.assertEqual(trade.exit_price, 95.0)
        self.assertLess(trade.net_r, -1.0)

    def test_crossing_wrong_side_resets_touch_eligibility(self):
        index = pd.date_range("2024-01-01", periods=10, freq="D")
        frame = pd.DataFrame(
            {
                "Open": [100.0, 102.0, 98.0, 99.8, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0],
                "High": [100.2, 102.2, 98.4, 100.1, 98.4, 98.4, 98.4, 98.4, 98.4, 98.4],
                "Low": [99.8, 101.8, 97.6, 99.7, 97.6, 97.6, 97.6, 97.6, 97.6, 97.6],
                "Close": [100.0, 102.0, 98.0, 99.9, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0],
                "ATR": [1.0] * 10,
            },
            index=index,
        )
        ma = pd.Series(100.0, index=index)
        config = ScanConfig(
            touch_zone_atr=0.5,
            separation_atr=1.0,
            max_holding_bars=2,
        )
        self.assertEqual(detect_touches(frame, ma, 1, config), [])
    def test_aggregate_trend_uses_clear_majority(self):
        label, votes = aggregate_trend(["Yükselen"] * 7 + ["Geçiş"] * 3)
        self.assertEqual(label, "Güçlü yükselen")
        self.assertEqual(votes, "7↑ 0↓ / 10")

    def test_market_summary_has_one_row_per_symbol(self):
        rows = []
        for symbol in ("ASELS", "THYAO"):
            for timeframe in ("1h", "1d"):
                for side in ("Destek", "Direnç"):
                    rows.append(
                        {
                            "asset_class": "stock",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "ma_type": "EMA",
                            "period": 20,
                            "ma": "EMA20",
                            "side": side,
                            "active_side": side == "Destek",
                            "trend_state": "Yükselen",
                            "compatibility": "Uyumlu",
                            "positive_periods": 3,
                            "edge_r": 0.25,
                            "median_net_r": 0.30,
                            "distance_atr": -0.4,
                            "touches": 16,
                            "win_rate_pct": 62.5,
                        }
                    )
        summary = build_market_summary(pd.DataFrame(rows))
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary["symbol"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
