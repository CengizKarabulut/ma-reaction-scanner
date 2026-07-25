import unittest

import numpy as np
import pandas as pd

from scanner.ma_engine import (
    MA_TYPES,
    ScanConfig,
    Touch,
    aggregate_trend,
    build_market_summary,
    compatibility_score,
    compute_ma,
    detect_touches,
    prepare_frame,
    quality_filter_reasons,
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
        initial_risk = trade.entry - trade.initial_stop
        self.assertAlmostEqual(trade.mfe_r, 1.0 / initial_risk)
        self.assertAlmostEqual(trade.mae_r, 5.0 / initial_risk)

    def test_negative_commodity_entry_keeps_transaction_cost_positive(self):
        index = pd.date_range("2020-04-19", periods=2, freq="D")
        frame = pd.DataFrame(
            {
                "Open": [-2.5, -2.0],
                "High": [-1.5, -1.0],
                "Low": [-3.0, -2.5],
                "Close": [-2.5, -1.5],
                "ATR": [1.0, 1.0],
            },
            index=index,
        )
        config = ScanConfig(max_holding_bars=1, roundtrip_cost_bps=25.0)
        trade = simulate_trade(
            frame,
            Touch(0, index[0], 1, -2.5, 1.0),
            config,
        )

        self.assertIsNotNone(trade)
        risk = trade.entry - trade.initial_stop
        gross_r = (trade.exit_price - trade.entry) / risk
        expected_cost_r = 25.0 / 10_000.0 * abs(trade.entry) / risk
        self.assertGreater(expected_cost_r, 0.0)
        self.assertAlmostEqual(trade.net_r, gross_r - expected_cost_r)
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
                            "current_price": 52.85,
                            "current_ma": 53.42,
                            "distance_value": 0.57,
                            "distance_pct": (53.42 - 52.85) / 52.85 * 100.0,
                            "filter_pass": True,
                            "filter_status": "Uygun",
                        }
                    )
        summary = build_market_summary(pd.DataFrame(rows))
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary["symbol"].nunique(), 2)
        self.assertAlmostEqual(
            float(summary.iloc[0]["best_distance_pct"]),
            (53.42 - 52.85) / 52.85 * 100.0,
        )
        self.assertEqual(float(summary.iloc[0]["current_price"]), 52.85)
        self.assertEqual(float(summary.iloc[0]["best_ma_value"]), 53.42)

    def test_filtered_outlier_is_not_selected_as_best(self):
        rows = pd.DataFrame(
            [
                {
                    "symbol": "TEST",
                    "timeframe": "1d",
                    "ma_type": "SMA",
                    "period": 20,
                    "ma": "SMA20",
                    "side": "Destek",
                    "active_side": True,
                    "trend_state": "Yukselen",
                    "compatibility": "Guclu uyum",
                    "positive_periods": 3,
                    "edge_r": 43.0,
                    "median_net_r": 4.0,
                    "distance_atr": 0.1,
                    "distance_pct": 0.2,
                    "touches": 20,
                    "win_rate_pct": 70.0,
                    "current_price": 10.0,
                    "current_ma": 10.02,
                    "distance_value": 0.02,
                    "filter_pass": False,
                    "filter_status": "Filtre disi",
                    "filter_reasons": "Aykiri Edge",
                },
                {
                    "symbol": "TEST",
                    "timeframe": "1d",
                    "ma_type": "EMA",
                    "period": 20,
                    "ma": "EMA20",
                    "side": "Destek",
                    "active_side": True,
                    "trend_state": "Yukselen",
                    "compatibility": "Uyumlu",
                    "positive_periods": 3,
                    "edge_r": 1.0,
                    "median_net_r": 1.0,
                    "distance_atr": 0.2,
                    "distance_pct": 0.3,
                    "touches": 20,
                    "win_rate_pct": 65.0,
                    "current_price": 10.0,
                    "current_ma": 10.03,
                    "distance_value": 0.03,
                    "filter_pass": True,
                    "filter_status": "Uygun",
                    "filter_reasons": "",
                },
            ]
        )
        summary = build_market_summary(rows)
        self.assertEqual(summary.iloc[0]["best_ma"], "EMA20")
        self.assertEqual(summary.iloc[0]["filter_status"], "Uygun")

    def test_try_filters_are_not_applied_to_global_stocks(self):
        config = ScanConfig(
            min_price=100.0,
            min_daily_turnover_try=1_000_000.0,
            max_zero_volume_pct=20.0,
            max_gap_pct=15.0,
            max_abs_edge_r=5.0,
        )
        reasons = quality_filter_reasons(
            asset_class="stock",
            market="GLOBAL",
            price=12.0,
            metrics={
                "median_daily_turnover_try": 50_000.0,
                "zero_volume_pct": 0.0,
                "max_recent_gap_pct": 0.0,
            },
            edge_r=1.0,
            config=config,
        )
        self.assertEqual(reasons, [])

    def test_compatibility_score_is_transparent_zero_to_one_hundred(self):
        config = ScanConfig(min_touches=12)

        score = compatibility_score(12, 100.0, 100.0, 1.0, 1.0, 3, config)

        self.assertEqual(score, 100.0)
        self.assertEqual(
            compatibility_score(0, float("nan"), float("nan"), -1.0, -1.0, 0, config),
            0.0,
        )
    def test_prepare_frame_allows_historic_negative_commodity_settlement(self):
        frame = sample_frame(100)
        frame.loc[frame.index[20], ["Open", "High", "Low", "Close"]] = [
            -2.0, -1.0, -3.0, -2.5
        ]

        with self.assertRaisesRegex(ValueError, "pozitif"):
            prepare_frame(frame)

        prepared = prepare_frame(frame, allow_non_positive=True)
        self.assertEqual(float(prepared.loc[frame.index[20], "Close"]), -2.5)

    def test_prepare_frame_rejects_zero_even_for_commodity(self):
        frame = sample_frame(100)
        frame.loc[frame.index[20], ["Open", "High", "Low", "Close"]] = [
            0.0, 1.0, -1.0, 0.5
        ]

        with self.assertRaisesRegex(ValueError, "sıfır"):
            prepare_frame(frame, allow_non_positive=True)
if __name__ == "__main__":
    unittest.main()
