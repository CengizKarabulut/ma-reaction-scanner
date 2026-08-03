import unittest

import numpy as np
import pandas as pd

from scanner.ma_engine import Touch
from scanner.ma_levels import (
    LevelConfig,
    add_adherence_excess,
    add_plateau_scores,
    level_class,
    level_score,
    summarize_outcomes,
    touch_outcomes,
)


def frame_from(highs, lows, closes, opens=None, start="2024-01-01"):
    index = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": opens if opens is not None else closes,
            "High": highs,
            "Low": lows,
            "Close": closes,
        },
        index=index,
    )


class TouchOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.config = LevelConfig(reaction_bars=3, hold_bars=3, break_atr=0.5)

    def test_support_that_holds_reports_bounce_and_hold(self):
        frame = frame_from(
            highs=[100.5, 102.0, 103.0, 104.0],
            lows=[99.9, 100.1, 101.0, 102.0],
            closes=[100.0, 101.5, 102.5, 103.5],
        )
        touch = Touch(0, frame.index[0], 1, 100.0, 1.0)

        outcome = touch_outcomes(frame, [touch], 1, self.config)[0]

        self.assertAlmostEqual(outcome.bounce_atr, 4.0)
        self.assertAlmostEqual(outcome.penetration_atr, 0.0)
        self.assertTrue(outcome.held)

    def test_support_that_breaks_is_not_counted_as_held(self):
        frame = frame_from(
            highs=[100.5, 100.2, 99.0, 98.5],
            lows=[99.9, 97.5, 97.0, 96.0],
            closes=[100.0, 98.0, 97.5, 96.5],
        )
        touch = Touch(0, frame.index[0], 1, 100.0, 1.0)

        outcome = touch_outcomes(frame, [touch], 1, self.config)[0]

        self.assertFalse(outcome.held)
        self.assertAlmostEqual(outcome.penetration_atr, 4.0)

    def test_resistance_is_the_mirror_image_of_support(self):
        frame = frame_from(
            highs=[100.1, 99.9, 99.0, 98.0],
            lows=[99.5, 98.0, 97.0, 96.0],
            closes=[100.0, 98.5, 97.5, 96.5],
        )
        touch = Touch(0, frame.index[0], -1, 100.0, 1.0)

        outcome = touch_outcomes(frame, [touch], -1, self.config)[0]

        self.assertAlmostEqual(outcome.bounce_atr, 4.0)
        self.assertTrue(outcome.held)

    def test_bounce_is_scale_invariant_because_it_is_measured_in_atr(self):
        small = frame_from(
            highs=[10.05, 10.20, 10.30, 10.40],
            lows=[9.99, 10.01, 10.10, 10.20],
            closes=[10.00, 10.15, 10.25, 10.35],
        )
        large = frame_from(
            highs=[1000.5, 1020.0, 1030.0, 1040.0],
            lows=[999.9, 1001.0, 1010.0, 1020.0],
            closes=[1000.0, 1015.0, 1025.0, 1035.0],
        )

        small_outcome = touch_outcomes(
            small, [Touch(0, small.index[0], 1, 10.0, 0.1)], 1, self.config
        )[0]
        large_outcome = touch_outcomes(
            large, [Touch(0, large.index[0], 1, 1000.0, 10.0)], 1, self.config
        )[0]

        self.assertAlmostEqual(small_outcome.bounce_atr, large_outcome.bounce_atr)

    def test_non_positive_atr_touches_are_skipped(self):
        frame = frame_from(
            highs=[100.5, 102.0], lows=[99.9, 100.1], closes=[100.0, 101.5]
        )
        self.assertEqual(
            touch_outcomes(frame, [Touch(0, frame.index[0], 1, 100.0, 0.0)], 1, self.config),
            [],
        )

    def test_side_must_be_plus_or_minus_one(self):
        frame = frame_from(highs=[1.1], lows=[0.9], closes=[1.0])
        with self.assertRaises(ValueError):
            touch_outcomes(frame, [], 0, self.config)


class LevelScoreTests(unittest.TestCase):
    def setUp(self):
        self.config = LevelConfig(evidence_target_touches=20, bounce_cap_atr=2.0)

    def test_perfect_level_scores_one_hundred(self):
        metrics = {
            "level_touches": 20,
            "hold_rate_pct": 100.0,
            "median_bounce_atr": 2.0,
            "cross_per_100": 0.0,
        }
        self.assertEqual(level_score(metrics, self.config), 100.0)
        self.assertEqual(level_class(metrics, 100.0, self.config), "Guclu seviye")

    def test_untouched_average_scores_zero_not_a_cleanliness_bonus(self):
        metrics = summarize_outcomes([], valid_bars=500, cross_count=0, config=self.config)

        self.assertEqual(metrics["level_touches"], 0)
        self.assertEqual(level_score(metrics, self.config), 0.0)
        self.assertEqual(level_class(metrics, 0.0, self.config), "Yetersiz temas")

    def test_cleanliness_scales_the_score_instead_of_subtracting_from_it(self):
        config = LevelConfig(
            evidence_target_touches=20,
            bounce_cap_atr=2.0,
            cross_cap_per_100=8.0,
            cross_damping=0.6,
        )
        clean = {
            "level_touches": 20,
            "hold_rate_pct": 100.0,
            "median_bounce_atr": 2.0,
            "cross_per_100": 0.0,
        }
        choppy = dict(clean, cross_per_100=8.0)

        self.assertEqual(level_score(clean, config), 100.0)
        # 1 - 0.6 * (8/8) = 0.4 of the clean score, not a flat deduction.
        self.assertAlmostEqual(level_score(choppy, config), 40.0)

    def test_a_choppy_short_average_cannot_outrank_a_clean_long_one(self):
        """Regression: the additive cleanliness term let noise win.

        These two rows are taken from a real BIST daily scan.  The first is a
        10-period average price crosses roughly every third bar; the second is
        SMA200.  Under the original additive score the choppy average came out
        ahead (71.4 against 63.2) and was labelled a strong level.
        """
        config = LevelConfig()
        choppy_short = {
            "level_touches": 21,
            "hold_rate_pct": 71.43,
            "median_bounce_atr": 3.25,
            "cross_per_100": 29.41,
        }
        clean_long = {
            "level_touches": 8,
            "hold_rate_pct": 62.50,
            "median_bounce_atr": 1.94,
            "cross_per_100": 1.96,
        }

        self.assertLess(
            level_score(choppy_short, config), level_score(clean_long, config)
        )
        self.assertNotEqual(
            level_class(choppy_short, level_score(choppy_short, config), config),
            "Guclu seviye",
        )

    def test_class_thresholds_are_configurable(self):
        metrics = {
            "level_touches": 20,
            "hold_rate_pct": 60.0,
            "median_bounce_atr": 2.0,
            "cross_per_100": 2.0,
        }
        score = level_score(metrics, LevelConfig())

        strict = LevelConfig(strong_threshold=95.0, level_threshold=90.0, weak_threshold=85.0)
        loose = LevelConfig(strong_threshold=10.0, level_threshold=5.0, weak_threshold=1.0)

        self.assertEqual(level_class(metrics, score, strict), "Seviye degil")
        self.assertEqual(level_class(metrics, score, loose), "Guclu seviye")

    def test_thresholds_must_be_ordered(self):
        with self.assertRaises(ValueError):
            LevelConfig(strong_threshold=10.0, level_threshold=50.0)

    def test_damping_must_be_a_fraction(self):
        with self.assertRaises(ValueError):
            LevelConfig(cross_damping=1.5)

    def test_density_normalisation_makes_short_and_long_periods_comparable(self):
        short = summarize_outcomes(
            touch_outcomes(
                frame_from([101.0] * 4, [99.0] * 4, [100.0] * 4),
                [Touch(0, pd.Timestamp("2024-01-01"), 1, 100.0, 1.0)],
                1,
                LevelConfig(reaction_bars=2, hold_bars=2),
            ),
            valid_bars=100,
            cross_count=0,
            config=self.config,
        )
        self.assertAlmostEqual(short["touch_density_per_100"], 1.0)


class PlateauTests(unittest.TestCase):
    def setUp(self):
        self.config = LevelConfig(neighbor_ratio=0.25)

    def _frame(self, periods, scores):
        return pd.DataFrame(
            {
                "symbol": ["ASELS"] * len(periods),
                "timeframe": ["1d"] * len(periods),
                "ma_type": ["EMA"] * len(periods),
                "side": ["Destek"] * len(periods),
                "period": periods,
                "level_score": scores,
            }
        )

    def test_isolated_peak_is_flagged_but_a_plateau_is_not(self):
        result = add_plateau_scores(self._frame([20, 21, 22], [80.0, 30.0, 30.0]), self.config)

        peak = result[result["period"] == 20].iloc[0]
        neighbour = result[result["period"] == 22].iloc[0]

        self.assertEqual(int(peak["plateau_neighbors"]), 2)
        self.assertAlmostEqual(float(peak["plateau_ratio"]), 30.0 / 80.0)
        self.assertAlmostEqual(float(neighbour["plateau_ratio"]), 1.0)

    def test_genuine_plateau_scores_near_one(self):
        result = add_plateau_scores(self._frame([20, 21, 22], [70.0, 72.0, 68.0]), self.config)

        self.assertTrue((result["plateau_ratio"] > 0.9).all())

    def test_isolated_period_reports_no_neighbours(self):
        result = add_plateau_scores(self._frame([5, 34, 377], [90.0, 20.0, 20.0]), self.config)

        self.assertTrue((result["plateau_neighbors"] == 0).all())
        self.assertTrue(result["plateau_ratio"].isna().all())

    def test_different_ma_types_are_not_treated_as_neighbours(self):
        frame = self._frame([20, 21], [90.0, 10.0])
        frame.loc[1, "ma_type"] = "SMA"

        result = add_plateau_scores(frame, self.config)

        self.assertTrue((result["plateau_neighbors"] == 0).all())

    def test_empty_frame_still_gains_the_columns(self):
        result = add_plateau_scores(pd.DataFrame(), self.config)
        self.assertIn("plateau_ratio", result.columns)
        self.assertIn("plateau_neighbors", result.columns)


class AdherenceExcessTests(unittest.TestCase):
    def test_excess_is_measured_against_the_symbols_own_median(self):
        frame = pd.DataFrame(
            {
                "symbol": ["ASELS"] * 3,
                "timeframe": ["1d"] * 3,
                "side": ["Destek"] * 3,
                "side_adherence_pct": [80.0, 85.0, 95.0],
            }
        )

        result = add_adherence_excess(frame)

        self.assertAlmostEqual(float(result["adherence_excess_pct"].iloc[0]), -5.0)
        self.assertAlmostEqual(float(result["adherence_excess_pct"].iloc[1]), 0.0)
        self.assertAlmostEqual(float(result["adherence_excess_pct"].iloc[2]), 10.0)

    def test_a_uniformly_trending_symbol_produces_no_excess(self):
        frame = pd.DataFrame(
            {
                "symbol": ["THYAO"] * 4,
                "timeframe": ["1d"] * 4,
                "side": ["Destek"] * 4,
                "side_adherence_pct": [88.0, 88.0, 88.0, 88.0],
            }
        )

        result = add_adherence_excess(frame)

        self.assertTrue(np.allclose(result["adherence_excess_pct"], 0.0))


if __name__ == "__main__":
    unittest.main()
