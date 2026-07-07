import unittest

import numpy as np
import pandas as pd

from scanner.ma_core import AnalysisConfig, prepare_frame
from scanner.sr_baselines import causal_baseline_levels, compare_level_families


class BaselineTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2023-01-01", periods=240, freq="D")
        close = 100 + np.sin(np.arange(240) / 8.0) * 5 + np.arange(240) * 0.01
        self.frame = pd.DataFrame(
            {
                "Open": close,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": 1_000_000 + np.arange(240) * 100,
            },
            index=index,
        )

    def test_baselines_are_causal(self):
        before = causal_baseline_levels(self.frame, windows=(20,))
        changed = self.frame.copy()
        changed.iloc[-1, changed.columns.get_loc("Close")] += 50
        changed.iloc[-1, changed.columns.get_loc("High")] += 50
        after = causal_baseline_levels(changed, windows=(20,))
        for key in before:
            self.assertAlmostEqual(before[key][2].iloc[-1], after[key][2].iloc[-1])

    def test_family_comparison_uses_common_schema(self):
        cfg = AnalysisConfig(
            horizon=5, min_events=3, min_segment_events=1, null_iterations=19,
            use_shift_control=False, use_horizontal_control=False,
        )
        prepared = prepare_frame(self.frame, cfg)
        levels = causal_baseline_levels(self.frame, windows=(20, 50))
        result = compare_level_families(prepared, levels, cfg, "TEST", "1d")
        self.assertFalse(result.empty)
        self.assertTrue({"family", "q_value", "certified", "holdout_score"}.issubset(result.columns))


if __name__ == "__main__":
    unittest.main()

