import unittest

import pandas as pd

from scanner.ma_calibration import CalibrationConfig, score_bucket_summary


class CalibrationSummaryTests(unittest.TestCase):
    def test_score_buckets_are_reported_by_side_with_explicit_scope(self):
        frame = pd.DataFrame(
            [
                {
                    "side": "Destek",
                    "level_score": 32.0,
                    "level_touches": 8,
                    "hold_rate_pct": 55.0,
                    "break_rate_pct": 45.0,
                    "median_bounce_atr": 0.8,
                    "median_penetration_atr": 0.2,
                },
                {
                    "side": "Destek",
                    "level_score": 51.0,
                    "level_touches": 18,
                    "hold_rate_pct": 70.0,
                    "break_rate_pct": 30.0,
                    "median_bounce_atr": 1.4,
                    "median_penetration_atr": 0.1,
                },
                {
                    "side": "Direnc",
                    "level_score": 52.0,
                    "level_touches": 20,
                    "hold_rate_pct": 65.0,
                    "break_rate_pct": 35.0,
                    "median_bounce_atr": 2.5,
                    "median_penetration_atr": 0.4,
                },
            ]
        )

        summary = score_bucket_summary(frame)

        self.assertEqual(set(summary["analysis_scope"]), {"in_sample_aggregate"})
        self.assertIn("Destek", set(summary["side"]))
        self.assertIn("Direnc", set(summary["side"]))
        strong = summary[(summary["side"] == "Direnc") & (summary["score_bucket"] == "50-55")]
        self.assertEqual(int(strong.iloc[0]["row_count"]), 1)
        self.assertEqual(float(strong.iloc[0]["row_share_median_bounce_ge_2atr_pct"]), 100.0)

    def test_empty_or_missing_score_returns_shaped_frame(self):
        summary = score_bucket_summary(pd.DataFrame({"side": ["Destek"]}))

        self.assertEqual(len(summary), 0)
        self.assertIn("analysis_scope", summary.columns)
        self.assertIn("score_bucket", summary.columns)

    def test_bucket_edges_must_be_ordered(self):
        with self.assertRaises(ValueError):
            CalibrationConfig(score_buckets=(0.0, 50.0, 25.0))


if __name__ == "__main__":
    unittest.main()
