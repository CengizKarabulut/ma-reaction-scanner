import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scanner.ma_core import (
    _matched_random_scores,
    AnalysisConfig,
    adjust_fdr,
    analyze_ma_universe,
    build_confluence,
    compute_ma,
    detect_independent_touches,
    evaluate_candidate,
    measure_event,
    normalize_ohlcv,
    precompute_forward_outcomes,
    prepare_frame,
    select_panel_levels,
)


def analysis_frame(close, atr_value=1.0):
    close = np.asarray(close, dtype=float)
    index = pd.date_range("2020-01-01", periods=len(close), freq="D")
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.20,
            "Low": close - 0.20,
            "Close": close,
            "Volume": 1_000_000.0,
            "ATR": atr_value,
            "ADX": 20.0,
            "VOL_BIN": 1,
            "SESSION_BIN": 0,
        },
        index=index,
    )
    return frame


def repeated_support_bounces(cycles=50, fail_after=None):
    successful = [105, 105, 104, 103, 101, 100, 102, 104, 105, 104, 105, 105]
    failed = [105, 105, 104, 103, 101, 100, 99, 98, 97, 98, 99, 100]
    values = []
    for cycle in range(cycles):
        values.extend(failed if fail_after is not None and cycle >= fail_after else successful)
    return analysis_frame(values)


class CoreMathTests(unittest.TestCase):
    def test_normalize_and_sma(self):
        raw = pd.DataFrame(
            {
                "open": [10, 11, 12], "high": [11, 12, 13],
                "low": [9, 10, 11], "close": [10, 11, 12],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )
        normalized = normalize_ohlcv(raw)
        self.assertEqual(list(normalized.columns), ["Open", "High", "Low", "Close", "Volume"])
        result = compute_ma("SMA", normalized["Close"], normalized["Volume"], 2)
        self.assertAlmostEqual(result.iloc[-1], 11.5)

    def test_weighted_mas_match_rolling_apply(self):
        index = pd.date_range("2024-01-01", periods=7)
        close = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0], index=index)
        volume = pd.Series(1.0, index=index)

        wma_weights = np.arange(1, 4, dtype=float)
        expected_wma = close.rolling(3, min_periods=3).apply(
            lambda values: float(np.dot(values, wma_weights) / wma_weights.sum()),
            raw=True,
        )
        pd.testing.assert_series_equal(
            compute_ma("WMA", close, volume, 3), expected_wma
        )

        center = 0.85 * (3 - 1)
        width = 3 / 6.0
        x = np.arange(3)
        alma_weights = np.exp(-((x - center) ** 2) / (2 * width * width))
        alma_weights /= alma_weights.sum()
        expected_alma = close.rolling(3, min_periods=3).apply(
            lambda values: float(np.dot(values, alma_weights)), raw=True
        )
        pd.testing.assert_series_equal(
            compute_ma("ALMA", close, volume, 3), expected_alma
        )

    def test_volatility_bins_do_not_change_when_future_data_is_appended(self):
        close = np.linspace(100.0, 104.0, 90)
        base = pd.DataFrame(
            {
                "Open": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000_000.0,
            },
            index=pd.date_range("2024-01-01", periods=len(close), freq="D"),
        )
        shock = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [190.0],
                "Low": [110.0],
                "Close": [150.0],
                "Volume": [1_000_000.0],
            },
            index=[base.index[-1] + pd.Timedelta(days=1)],
        )
        cfg = AnalysisConfig()

        original = prepare_frame(base, cfg)["VOL_BIN"]
        extended_prefix = prepare_frame(pd.concat([base, shock]), cfg)["VOL_BIN"].iloc[:-1]

        pd.testing.assert_series_equal(original, extended_prefix, check_freq=False)

    def test_invalid_ohlc_is_rejected(self):
        raw = pd.DataFrame(
            {"Open": [10], "High": [9], "Low": [8], "Close": [10], "Volume": [1]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        with self.assertRaises(ValueError):
            normalize_ohlcv(raw)

    def test_touches_are_independent_and_directional(self):
        cycle = [105, 104, 103, 100, 102, 104, 105]
        frame = analysis_frame(cycle * 8)
        ma = pd.Series(100.0, index=frame.index)
        cfg = AnalysisConfig(
            horizon=4, separation_atr=2.0, min_events=2, min_segment_events=1,
            null_iterations=19, use_shift_control=False, use_horizontal_control=False,
        )
        events = detect_independent_touches(frame, ma, cfg)
        self.assertGreaterEqual(len(events), 6)
        self.assertTrue(all(event.direction == 1 for event in events))
        self.assertTrue(all(b.position - a.position > cfg.horizon for a, b in zip(events, events[1:])))

    def test_staying_above_ma_is_not_a_breakthrough_or_touch(self):
        frame = analysis_frame([110.0] * 50)
        ma = pd.Series(100.0, index=frame.index)
        cfg = AnalysisConfig(horizon=5, null_iterations=19)
        self.assertEqual(detect_independent_touches(frame, ma, cfg), [])

    def test_same_bar_target_and_stop_is_conservative(self):
        frame = analysis_frame([100.0] * 40)
        frame.loc[frame.index[21], "High"] = 102.0
        frame.loc[frame.index[21], "Low"] = 98.0
        cfg = AnalysisConfig(horizon=5, target_atr=1.0, stop_atr=1.0, null_iterations=19)
        result = measure_event(frame, 20, 1, cfg)
        self.assertIsNotNone(result)
        self.assertEqual(result.first_hit, -1)
        self.assertTrue(result.ambiguous_bar)

    def test_forward_outcomes_match_measure_event_field_by_field(self):
        frame = analysis_frame([105, 104, 103, 100, 102, 104, 106, 103, 99, 101] * 6)
        cfg = AnalysisConfig(horizon=5, target_atr=1.0, stop_atr=1.0, null_iterations=19)
        outcomes = precompute_forward_outcomes(frame, cfg)

        for position in (3, 8, 12, 20, 35):
            for direction in (1, -1):
                direct = measure_event(frame, position, direction, cfg)
                cached = outcomes.measurement(position, direction)
                self.assertIsNotNone(direct)
                self.assertIsNotNone(cached)
                self.assertEqual(direct.position, cached.position)
                self.assertEqual(direct.direction, cached.direction)
                self.assertEqual(direct.first_hit, cached.first_hit)
                self.assertAlmostEqual(direct.fixed_return_atr, cached.fixed_return_atr)
                self.assertAlmostEqual(direct.favorable_atr, cached.favorable_atr)
                self.assertAlmostEqual(direct.adverse_atr, cached.adverse_atr)
                if np.isnan(direct.bars_to_target):
                    self.assertTrue(np.isnan(cached.bars_to_target))
                else:
                    self.assertAlmostEqual(direct.bars_to_target, cached.bars_to_target)
                self.assertEqual(direct.retested, cached.retested)
                self.assertEqual(direct.ambiguous_bar, cached.ambiguous_bar)

    def test_matched_random_scores_are_identical_with_forward_outcomes(self):
        bounce = [105, 105, 104, 103, 101, 100, 102, 104, 105, 104, 105, 105]
        values = []
        for _ in range(30):
            values.extend(bounce)
        frame = analysis_frame(values)
        ma = pd.Series(100.0, index=frame.index)
        cfg = AnalysisConfig(
            horizon=4,
            target_atr=1.0,
            stop_atr=1.0,
            separation_atr=2.0,
            min_events=3,
            min_segment_events=1,
            null_iterations=29,
            use_shift_control=False,
            use_horizontal_control=False,
            random_seed=99,
        )
        events = detect_independent_touches(frame, ma, cfg)
        discovery_end = int(len(frame) * cfg.discovery_fraction)
        outcomes = precompute_forward_outcomes(frame, cfg)

        legacy = _matched_random_scores(
            frame,
            ma,
            events,
            1,
            0,
            discovery_end,
            cfg,
            np.random.default_rng(123),
        )
        cached = _matched_random_scores(
            frame,
            ma,
            events,
            1,
            0,
            discovery_end,
            cfg,
            np.random.default_rng(123),
            outcomes,
        )

        self.assertEqual(legacy, cached)


class StatisticalGateTests(unittest.TestCase):
    def test_fdr_adjustment_and_dependency_penalty(self):
        p = [0.001, 0.01, 0.03, np.nan]
        bh = adjust_fdr(p, "bh")
        by = adjust_fdr(p, "by")
        self.assertTrue(np.all(by[np.isfinite(by)] >= bh[np.isfinite(bh)]))
        self.assertLessEqual(bh[0], bh[1])
        self.assertTrue(np.isnan(bh[-1]))

    def test_repeatable_bounce_beats_matched_random_control(self):
        bounce = [105, 105, 104, 103, 101, 100, 102, 104, 105, 104, 105, 105]
        non_touch_pullback = [106, 105, 104, 103, 102, 102, 102, 102, 102, 102, 103, 106]
        values = []
        for _ in range(60):
            values.extend(bounce)
            values.extend(non_touch_pullback)
        frame = analysis_frame(values)
        ma = pd.Series(100.0, index=frame.index)
        cfg = AnalysisConfig(
            horizon=4, target_atr=1.0, stop_atr=1.0, separation_atr=2.0,
            min_events=8, min_segment_events=2, null_iterations=199,
            use_shift_control=False, use_horizontal_control=False, random_seed=77,
        )
        result = evaluate_candidate(frame, ma, "SMA", 20, 1, cfg, seed=77)
        self.assertGreater(result["discovery_events"], 10)
        self.assertGreater(result["discovery_score"], 0)
        self.assertLessEqual(result["p_random"], 0.10)
        self.assertTrue(result["validation_pass"])
        self.assertTrue(result["holdout_pass"])
        self.assertTrue(np.isfinite(result["holdout_median_fixed_atr_ci_low"]))
        self.assertTrue(np.isfinite(result["holdout_median_fixed_atr_ci_high"]))

    def test_holdout_rejects_a_decay_pattern(self):
        frame = repeated_support_bounces(cycles=60, fail_after=46)
        ma = pd.Series(100.0, index=frame.index)
        cfg = AnalysisConfig(
            horizon=4, min_events=8, min_segment_events=2, null_iterations=99,
            use_shift_control=False, use_horizontal_control=False,
        )
        result = evaluate_candidate(frame, ma, "SMA", 20, 1, cfg, seed=11)
        self.assertTrue(result["validation_pass"])
        self.assertFalse(result["holdout_pass"])

    def test_missing_secondary_control_cannot_certify(self):
        frame = repeated_support_bounces(cycles=12)
        ma = pd.Series(100.0, index=frame.index)
        cfg = AnalysisConfig(
            horizon=4, min_events=3, min_segment_events=1, null_iterations=49,
            use_shift_control=True, use_horizontal_control=True,
        )
        result = evaluate_candidate(frame, ma, "SMA", 100, 1, cfg, seed=5)
        self.assertFalse(result["secondary_controls_pass"])
        if np.isfinite(result["p_random"]):
            self.assertEqual(result["p_value"], 1.0)


class PresentationTests(unittest.TestCase):
    def test_distance_argument_no_longer_skips_evidence_tests(self):
        frame = analysis_frame(np.linspace(100.0, 150.0, 160))
        cfg = AnalysisConfig(
            horizon=4,
            null_iterations=19,
            use_shift_control=False,
            use_horizontal_control=False,
        )
        result = analyze_ma_universe(
            frame,
            "TEST",
            "1d",
            cfg,
            ma_types=("SMA",),
            periods=(20,),
            active_only=True,
            max_evaluated_distance_atr=0.10,
        )

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertTrue(bool(row["active_side"]))
        self.assertFalse(bool(row["screen_skipped"]))
        self.assertNotEqual(row["status"], "distance_skipped")

    def _passing_candidate(self, *, holdout_events=5, holdout_thin=False):
        return {
            "ma_type": "SMA",
            "period": 20,
            "side": "support",
            "discovery_events": 12,
            "discovery_score": 1.0,
            "discovery_hit_rate": 0.75,
            "discovery_wilson_lower": 0.60,
            "discovery_median_fixed_atr": 0.50,
            "validation_score": 1.0,
            "validation_median_fixed_atr": 0.40,
            "validation_pass": True,
            "holdout_events": holdout_events,
            "holdout_score": 1.0,
            "holdout_median_fixed_atr": 0.35,
            "holdout_median_fixed_atr_ci_low": 0.20,
            "holdout_median_fixed_atr_ci_high": 0.50,
            "holdout_wilson_lower": 0.60,
            "holdout_wilson_pass": True,
            "holdout_pass": True,
            "holdout_thin": holdout_thin,
            "p_value": 0.001,
            "p_random": 0.001,
            "p_shift": 0.001,
            "p_horizontal": 0.001,
            "shift_control_pass": True,
            "horizontal_control_pass": True,
            "secondary_controls_pass": True,
            "shift_score_threshold": 0.0,
            "horizontal_score_threshold": 0.0,
            "random_score_threshold": 0.0,
        }

    def test_fast_profile_downgrades_raw_certification(self):
        frame = analysis_frame(np.linspace(100.0, 130.0, 180))
        cfg = AnalysisConfig(
            null_iterations=29,
            use_shift_control=False,
            use_horizontal_control=False,
        )
        with patch("scanner.ma_core.evaluate_candidate", return_value=self._passing_candidate()):
            result = analyze_ma_universe(
                frame, "TEST", "1d", cfg, ma_types=("SMA",), periods=(20,), active_only=True
            )

        row = result.iloc[0]
        self.assertTrue(bool(row["raw_certified"]))
        self.assertTrue(bool(row["low_confidence_fast"]))
        self.assertTrue(bool(row["low_confidence"]))
        self.assertFalse(bool(row["certified"]))
        self.assertEqual(row["status"], "low_confidence_fast")
        self.assertGreater(float(row["sr_strength_score"]), 0.0)
        self.assertIn("holdout_net_median_fixed_atr", result.columns)

    def test_thin_holdout_downgrades_raw_certification(self):
        frame = analysis_frame(np.linspace(100.0, 130.0, 180))
        cfg = AnalysisConfig(null_iterations=499)
        candidate = self._passing_candidate(holdout_events=3, holdout_thin=True)
        with patch("scanner.ma_core.evaluate_candidate", return_value=candidate):
            result = analyze_ma_universe(
                frame, "TEST", "1d", cfg, ma_types=("SMA",), periods=(20,), active_only=True
            )

        row = result.iloc[0]
        self.assertTrue(bool(row["raw_certified"]))
        self.assertTrue(bool(row["certified_thin_holdout"]))
        self.assertTrue(bool(row["low_confidence"]))
        self.assertFalse(bool(row["certified"]))
        self.assertEqual(row["status"], "certified_thin_holdout")

    def test_panel_labels_unverified_rows_as_candidates(self):
        rows = pd.DataFrame(
            [
                {"ticker": "TEST", "timeframe": "1d", "side": "support", "active_side": True,
                 "ma_type": "SMA", "period": 20, "certified": False, "actionable": False,
                 "rank_score": 1.0, "distance_atr": -1.0},
                {"ticker": "TEST", "timeframe": "1d", "side": "support", "active_side": True,
                 "ma_type": "WMA", "period": 34, "certified": False, "actionable": False,
                 "low_confidence": True, "rank_score": 20.0, "distance_atr": -2.0},
                {"ticker": "TEST", "timeframe": "1d", "side": "resistance", "active_side": True,
                 "ma_type": "EMA", "period": 50, "certified": True, "actionable": True,
                 "rank_score": 100.0, "distance_atr": 2.0},
            ]
        )
        panel = select_panel_levels(rows, top_n=3)
        labels = dict(zip(panel["side"], panel["evidence_label"]))
        self.assertIn("CANDIDATE_ONLY", set(panel["evidence_label"]))
        self.assertIn("LOW_CONFIDENCE", set(panel["evidence_label"]))
        self.assertEqual(labels["resistance"], "CERTIFIED")

    def test_confluence_is_context_not_independent_proof(self):
        panel = pd.DataFrame(
            [
                {"ticker": "TEST", "timeframe": "1h", "side": "support", "current_ma": 100.0,
                 "ma_type": "EMA", "period": 20, "certified": True},
                {"ticker": "TEST", "timeframe": "4h", "side": "support", "current_ma": 100.4,
                 "ma_type": "SMA", "period": 50, "certified": False},
            ]
        )
        clusters = build_confluence(panel, tolerance_pct=0.75)
        self.assertEqual(len(clusters), 1)
        self.assertTrue(bool(clusters.iloc[0]["qualified"]))
        self.assertEqual(clusters.iloc[0]["interpretation"], "context_cluster_not_independent_confirmation")


if __name__ == "__main__":
    unittest.main()

