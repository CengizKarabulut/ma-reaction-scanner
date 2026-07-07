import unittest

import numpy as np
import pandas as pd

from scanner.ma_core import (
    AnalysisConfig,
    adjust_fdr,
    build_confluence,
    compute_ma,
    detect_independent_touches,
    evaluate_candidate,
    measure_event,
    normalize_ohlcv,
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
    def test_panel_labels_unverified_rows_as_candidates(self):
        rows = pd.DataFrame(
            [
                {"ticker": "TEST", "timeframe": "1d", "side": "support", "active_side": True,
                 "ma_type": "SMA", "period": 20, "certified": False, "actionable": False,
                 "rank_score": 1.0, "distance_atr": -1.0},
                {"ticker": "TEST", "timeframe": "1d", "side": "resistance", "active_side": True,
                 "ma_type": "EMA", "period": 50, "certified": True, "actionable": True,
                 "rank_score": 100.0, "distance_atr": 2.0},
            ]
        )
        panel = select_panel_levels(rows, top_n=3)
        labels = dict(zip(panel["side"], panel["evidence_label"]))
        self.assertEqual(labels["support"], "CANDIDATE_ONLY")
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

