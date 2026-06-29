import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scanner.ma_core import AnalysisConfig, detect_independent_touches, evaluate_candidate


def frame_with_indicators(close):
    close = np.asarray(close, dtype=float)
    index = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame({
        "Open": close,
        "High": close + 0.1,
        "Low": close - 0.1,
        "Close": close,
        "Volume": 1000.0,
        "ATR": 1.0,
        "ADX": 20.0,
        "VOL_BIN": 1,
        "SESSION_BIN": 0,
    }, index=index)


class RegressionTests(unittest.TestCase):
    def test_wide_touch_bar_cannot_create_its_own_preseparation(self):
        frame = frame_with_indicators([100.0] * 40)
        frame.iloc[10, frame.columns.get_loc("Close")] = 103.0
        frame.iloc[10, frame.columns.get_loc("Open")] = 100.0
        frame.iloc[10, frame.columns.get_loc("High")] = 103.2
        frame.iloc[10, frame.columns.get_loc("Low")] = 99.9
        ma = pd.Series(100.0, index=frame.index)
        cfg = AnalysisConfig(
            horizon=4, separation_atr=2.0, null_iterations=19,
            use_shift_control=False, use_horizontal_control=False,
        )
        self.assertEqual(detect_independent_touches(frame, ma, cfg), [])

    def test_touch_records_approach_shape_bin(self):
        frame = frame_with_indicators(([106, 105, 104, 102, 100, 102, 104, 105] * 8))
        ma = pd.Series(100.0, index=frame.index)
        cfg = AnalysisConfig(
            horizon=3, separation_atr=2.0, null_iterations=19,
            use_shift_control=False, use_horizontal_control=False,
        )
        events = detect_independent_touches(frame, ma, cfg)
        self.assertTrue(events)
        self.assertTrue(all(event.approach_bin >= 0 for event in events))

    def test_small_secondary_ensembles_are_percentile_gates_not_pvalue_gates(self):
        cycle = [105, 104, 103, 100, 102, 104, 105, 105]
        frame = frame_with_indicators(cycle * 30)
        ma = pd.Series(100.0, index=frame.index)
        cfg = AnalysisConfig(
            horizon=3, min_events=5, min_segment_events=1, null_iterations=19,
            use_shift_control=True, use_horizontal_control=True,
        )
        with patch("scanner.ma_core._matched_random_scores", return_value=[-2.0] * 19), \
             patch("scanner.ma_core._shift_control_scores", return_value=[-2.0, -1.5, -1.0]), \
             patch("scanner.ma_core._horizontal_control_scores", return_value=[-2.0, -1.0]):
            result = evaluate_candidate(frame, ma, "SMA", 20, 1, cfg, seed=1)
        self.assertTrue(result["secondary_controls_pass"])
        self.assertLess(result["p_value"], 0.10)
        # Coarse diagnostic p-values themselves cannot reach 0.10 here.
        self.assertGreater(result["p_shift"], 0.10)
        self.assertGreater(result["p_horizontal"], 0.10)


if __name__ == "__main__":
    unittest.main()

