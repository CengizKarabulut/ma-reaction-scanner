import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scanner.paper_tracker import advance_watchlist, build_watchlist, ledger_summary


class PaperTrackerTests(unittest.TestCase):
    def test_only_certified_rows_become_watches_by_default(self):
        panel = pd.DataFrame([
            {"ticker": "TEST", "timeframe": "1d", "side": "support", "ma_type": "SMA",
             "period": 20, "current_ma": 100, "current_price": 102, "q_value": 0.02,
             "certified": True},
            {"ticker": "TEST", "timeframe": "1d", "side": "resistance", "ma_type": "EMA",
             "period": 50, "current_ma": 110, "current_price": 102, "q_value": 0.80,
             "certified": False},
        ])
        ledger = build_watchlist(panel, created_at="2025-01-01")
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger.iloc[0]["state"], "WATCHING")

    def test_future_touch_resolves_without_using_old_events(self):
        cycle = [105, 104, 103, 100, 102, 104, 105, 105, 104, 103, 100, 102, 104, 105]
        close = np.asarray(cycle * 12, dtype=float)
        index = pd.date_range("2025-01-01", periods=len(close), freq="D")
        frame = pd.DataFrame({
            "Open": close, "High": close + 0.2, "Low": close - 0.2,
            "Close": close, "Volume": 1000.0,
        }, index=index)
        panel = pd.DataFrame([{
            "ticker": "TEST", "timeframe": "1d", "side": "support", "ma_type": "SMA",
            "period": 20, "current_ma": 100, "current_price": 105, "q_value": 0.01,
            "certified": True,
        }])
        ledger = build_watchlist(
            panel, created_at="2025-01-01", watch_after={("TEST", "1d"): str(index[40])}
        )
        constant_ma = pd.Series(100.0, index=index)
        with patch("scanner.paper_tracker.compute_ma", return_value=constant_ma):
            advanced = advance_watchlist(ledger, lambda ticker, tf: frame)
        self.assertEqual(advanced.iloc[0]["state"], "RESOLVED")
        self.assertGreater(pd.Timestamp(advanced.iloc[0]["trigger_time"]), index[40])
        summary = ledger_summary(advanced)
        self.assertEqual(int(summary.iloc[0]["resolved"]), 1)


if __name__ == "__main__":
    unittest.main()

