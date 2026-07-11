import unittest

import numpy as np
import pandas as pd

from scanner.ma_core import AnalysisConfig, TouchEvent
from scanner.ma_validation import CostModel, random_entry_benchmark, simulate_event_trades, trade_statistics


def trade_frame():
    index = pd.date_range("2025-01-01", periods=80, freq="D")
    close = np.full(80, 100.0)
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": 1000.0,
            "ATR": 1.0,
            "VOL_BIN": 1,
            "SESSION_BIN": 0,
        },
        index=index,
    )
    return frame


def event(frame, position, side):
    return TouchEvent(
        position=position, timestamp=frame.index[position], direction=side,
        regime="range", atr=1.0, entry=100.0, ma_value=100.0,
        volatility_bin=1, session_bin=0,
    )


class ValidationTests(unittest.TestCase):
    def test_long_enters_next_bar_and_exits_once(self):
        frame = trade_frame()
        frame.loc[frame.index[12], "High"] = 102.0
        cfg = AnalysisConfig(horizon=5, target_atr=1.0, stop_atr=1.0, null_iterations=19)
        trades = simulate_event_trades(
            frame, [event(frame, 10, 1)], cfg, 1, 0, len(frame),
            costs=CostModel(0, 0, 0),
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["entry_time"], frame.index[11])
        self.assertEqual(trades.iloc[0]["reason"], "TARGET")
        self.assertGreater(trades.iloc[0]["net_pnl"], 0)

    def test_short_target_is_implemented(self):
        frame = trade_frame()
        frame.loc[frame.index[22], "Low"] = 98.0
        cfg = AnalysisConfig(horizon=5, target_atr=1.0, stop_atr=1.0, null_iterations=19)
        trades = simulate_event_trades(
            frame, [event(frame, 20, -1)], cfg, -1, 0, len(frame),
            costs=CostModel(0, 0, 0),
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["reason"], "TARGET")
        self.assertEqual(trades.iloc[0]["execution_mode"], "direction_forecast_only")
        self.assertIn("not assumed executable", trades.iloc[0]["execution_note"])
        self.assertGreater(trades.iloc[0]["net_pnl"], 0)

    def test_costs_reduce_return(self):
        frame = trade_frame()
        frame.loc[frame.index[12], "High"] = 102.0
        cfg = AnalysisConfig(horizon=5, null_iterations=19)
        free = simulate_event_trades(frame, [event(frame, 10, 1)], cfg, 1, 0, len(frame), CostModel(0, 0, 0))
        costly = simulate_event_trades(frame, [event(frame, 10, 1)], cfg, 1, 0, len(frame), CostModel(10, 20, 10))
        self.assertLess(costly.iloc[0]["net_pnl"], free.iloc[0]["net_pnl"])

    def test_bist_cost_model_reports_bsmv_tick_and_t_plus_two(self):
        frame = trade_frame()
        frame.loc[frame.index[12], "High"] = 102.0
        cfg = AnalysisConfig(horizon=5, null_iterations=19)
        trades = simulate_event_trades(
            frame,
            [event(frame, 10, 1)],
            cfg,
            1,
            0,
            len(frame),
            CostModel(commission_bps=10, spread_bps=0, slippage_bps=0, tick_size=0.05),
        )
        row = trades.iloc[0]
        self.assertGreater(row["bsmv"], 0)
        self.assertEqual(row["settlement_lag_days"], 2)
        self.assertAlmostEqual((row["entry"] / 0.05) % 1, 0.0)
        self.assertEqual(trade_statistics(trades)["direction_forecast_only_trades"], 0)

    def test_profit_target_rounds_away_from_entry(self):
        frame = trade_frame()
        frame.loc[frame.index[12], "High"] = 101.24
        cfg = AnalysisConfig(horizon=5, target_atr=1.23, stop_atr=1.0, null_iterations=19)
        trades = simulate_event_trades(
            frame,
            [event(frame, 10, 1)],
            cfg,
            1,
            0,
            len(frame),
            CostModel(0, 0, 0, tick_size=0.05),
        )

        row = trades.iloc[0]
        self.assertAlmostEqual(row["target"], 101.25)
        self.assertEqual(row["reason"], "TIME")

    def test_short_profit_target_rounds_away_from_entry(self):
        frame = trade_frame()
        frame.loc[frame.index[22], "Low"] = 98.76
        cfg = AnalysisConfig(horizon=5, target_atr=1.23, stop_atr=1.0, null_iterations=19)
        trades = simulate_event_trades(
            frame,
            [event(frame, 20, -1)],
            cfg,
            -1,
            0,
            len(frame),
            CostModel(0, 0, 0, tick_size=0.05),
        )

        row = trades.iloc[0]
        self.assertAlmostEqual(row["target"], 98.75)
        self.assertEqual(row["reason"], "TIME")

    def test_random_benchmark_reports_empirical_probability(self):
        frame = trade_frame()
        frame.loc[frame.index[12], "High"] = 102.0
        cfg = AnalysisConfig(horizon=5, null_iterations=19)
        trades = simulate_event_trades(frame, [event(frame, 10, 1)], cfg, 1, 0, len(frame), CostModel(0, 0, 0))
        result = random_entry_benchmark(
            frame, trades, cfg, 1, 0, len(frame), CostModel(0, 0, 0), iterations=19, seed=1,
        )
        self.assertEqual(result["iterations"], 19)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_trade_statistics_include_drawdown(self):
        frame = trade_frame()
        cfg = AnalysisConfig(horizon=3, null_iterations=19)
        frame.loc[frame.index[12], "Low"] = 98.0
        trades = simulate_event_trades(frame, [event(frame, 10, 1)], cfg, 1, 0, len(frame), CostModel(0, 0, 0))
        stats = trade_statistics(trades)
        self.assertEqual(stats["trades"], 1)
        self.assertLessEqual(stats["max_drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()

