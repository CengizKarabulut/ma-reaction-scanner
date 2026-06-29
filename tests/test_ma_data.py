import unittest

import numpy as np
import pandas as pd

from scanner.ma_data import fingerprint_frame, resample_ohlcv


class DataLayerTests(unittest.TestCase):
    def setUp(self):
        first = pd.date_range("2026-01-05 10:00", periods=8, freq="1h")
        second = pd.date_range("2026-01-06 10:00", periods=8, freq="1h")
        index = first.append(second)
        close = np.arange(len(index), dtype=float) + 100.0
        self.frame = pd.DataFrame(
            {
                "Open": close - 0.1,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
                "Volume": 1000.0,
            },
            index=index,
        )

    def test_four_hour_bars_anchor_to_bist_open(self):
        result = resample_ohlcv(self.frame, "4h")
        self.assertTrue(set(result.index.hour).issubset({10, 14}))
        self.assertEqual(len(result), 4)

    def test_weekly_and_monthly_resampling_preserve_ohlc(self):
        weekly = resample_ohlcv(self.frame, "1wk")
        monthly = resample_ohlcv(self.frame, "1mo")
        self.assertEqual(len(weekly), 1)
        self.assertEqual(len(monthly), 1)
        self.assertEqual(weekly.iloc[0]["Open"], self.frame.iloc[0]["Open"])
        self.assertEqual(weekly.iloc[0]["Close"], self.frame.iloc[-1]["Close"])

    def test_fingerprint_is_deterministic_and_data_sensitive(self):
        first = fingerprint_frame(self.frame)
        second = fingerprint_frame(self.frame.copy())
        changed = self.frame.copy()
        changed.iloc[-1, changed.columns.get_loc("Close")] += 1.0
        changed.iloc[-1, changed.columns.get_loc("High")] += 1.0
        self.assertEqual(first, second)
        self.assertNotEqual(first, fingerprint_frame(changed))


if __name__ == "__main__":
    unittest.main()

