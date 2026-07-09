import unittest

import pandas as pd

from scanner.asset_reporting import build_instrument_summary


class AssetReportingTests(unittest.TestCase):
    def test_summary_has_one_row_per_instrument_and_two_sides(self):
        rows = []
        for ma_type, period, side, distance, certified in (
            ("SMA", 20, "support", -0.2, False),
            ("EMA", 50, "support", -0.5, True),
            ("WMA", 20, "resistance", 0.3, False),
            ("HMA", 50, "resistance", 0.7, False),
        ):
            rows.append(
                {
                    "ticker": "XU100",
                    "asset_class": "index",
                    "asset_label": "Endeks",
                    "universe": "bist_main_indices",
                    "display_name": "BIST 100",
                    "timeframe": "1d",
                    "current_price": 14000.0,
                    "current_ma": 13900.0 if side == "support" else 14100.0,
                    "ma_type": ma_type,
                    "period": period,
                    "side": side,
                    "active_side": True,
                    "distance_atr": distance,
                    "q_value": 0.04 if certified else 0.50,
                    "certified": certified,
                    "actionable": certified,
                    "rank_score": 100.0 if certified else 1.0,
                }
            )
        summary = build_instrument_summary(pd.DataFrame(rows))
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["asset_label"], "Endeks")
        self.assertEqual(summary.iloc[0]["support_ma"], "EMA")
        self.assertEqual(summary.iloc[0]["resistance_ma"], "WMA")
        self.assertEqual(summary.iloc[0]["tested_level_count"], 4)
        self.assertEqual(summary.iloc[0]["certified_level_count"], 1)
        self.assertAlmostEqual(summary.iloc[0]["certification_rate_pct"], 25.0)

    def test_uncertified_summary_keeps_nearest_current_candidate(self):
        rows = []
        for side, distance, ma_type in (
            ("support", -1.8, "HMA"),
            ("resistance", 0.25, "WMA"),
        ):
            rows.append(
                {
                    "ticker": "THYAO",
                    "asset_class": "stock",
                    "asset_label": "Hisse",
                    "universe": "custom",
                    "display_name": "THYAO",
                    "timeframe": "1d",
                    "current_price": 354.25,
                    "current_ma": 350.0 if side == "support" else 355.0,
                    "ma_type": ma_type,
                    "period": 20,
                    "side": side,
                    "active_side": True,
                    "distance_pct": -1.2 if side == "support" else 0.21,
                    "distance_atr": distance,
                    "q_value": 1.0,
                    "certified": False,
                    "actionable": False,
                    "rank_score": 1.0,
                    "status": "unverified_candidate",
                    "discovery_events": 18,
                    "discovery_pass": False,
                    "validation_pass": True,
                    "holdout_pass": True,
                }
            )
        summary = build_instrument_summary(pd.DataFrame(rows))
        row = summary.iloc[0]
        self.assertEqual(row["nearest_side"], "resistance")
        self.assertEqual(row["nearest_ma"], "WMA")
        self.assertAlmostEqual(row["nearest_abs_distance_atr"], 0.25)
        self.assertEqual(row["nearest_discovery_events"], 18)

    def test_screen_skipped_levels_are_not_counted_as_tested(self):
        base = {
            "ticker": "TEST",
            "asset_class": "stock",
            "asset_label": "Hisse",
            "universe": "custom",
            "display_name": "TEST",
            "timeframe": "1d",
            "current_price": 100.0,
            "current_ma": 99.0,
            "ma_type": "SMA",
            "period": 20,
            "side": "support",
            "active_side": True,
            "distance_atr": -1.0,
            "q_value": 0.5,
            "certified": False,
            "actionable": False,
            "rank_score": 1.0,
            "screen_skipped": False,
        }
        rows = [base, {**base, "period": 50, "screen_skipped": True}]
        summary = build_instrument_summary(pd.DataFrame(rows))
        row = summary.iloc[0]

        self.assertEqual(row["active_level_count"], 2)
        self.assertEqual(row["tested_level_count"], 1)
        self.assertEqual(row["screen_skipped_level_count"], 1)

    def test_low_confidence_summary_is_counted_without_nan_truthiness(self):
        base = {
            "ticker": "TEST",
            "asset_class": "stock",
            "asset_label": "Hisse",
            "universe": "custom",
            "display_name": "TEST",
            "timeframe": "1d",
            "current_price": 100.0,
            "period": 20,
            "active_side": True,
            "q_value": 0.5,
            "certified": False,
            "actionable": False,
            "rank_score": 1.0,
        }
        rows = [
            {
                **base,
                "current_ma": 99.0,
                "ma_type": "SMA",
                "side": "support",
                "distance_atr": -0.2,
                "low_confidence": True,
                "certified_thin_holdout": True,
            },
            {
                **base,
                "current_ma": 101.0,
                "ma_type": "EMA",
                "side": "resistance",
                "distance_atr": 0.3,
                "low_confidence": float("nan"),
                "certified_thin_holdout": float("nan"),
            },
        ]

        row = build_instrument_summary(pd.DataFrame(rows)).iloc[0]

        self.assertEqual(row["certified_level_count"], 0)
        self.assertEqual(row["low_confidence_level_count"], 1)
        self.assertEqual(row["thin_holdout_level_count"], 1)
        self.assertEqual(row["overall_evidence"], "LOW_CONFIDENCE")
        self.assertEqual(row["support_evidence"], "LOW_CONFIDENCE")
        self.assertEqual(row["resistance_evidence"], "CANDIDATE_ONLY")

    def test_same_symbol_in_two_asset_classes_is_not_merged(self):
        base = {
            "ticker": "TEST",
            "universe": "custom",
            "display_name": "TEST",
            "timeframe": "1d",
            "current_price": 100.0,
            "current_ma": 99.0,
            "ma_type": "SMA",
            "period": 20,
            "side": "support",
            "active_side": True,
            "distance_atr": -0.2,
            "q_value": 0.5,
            "certified": False,
            "actionable": False,
            "rank_score": 1.0,
        }
        rows = [
            {**base, "asset_class": "stock", "asset_label": "Hisse"},
            {**base, "asset_class": "index", "asset_label": "Endeks"},
        ]
        summary = build_instrument_summary(pd.DataFrame(rows))
        self.assertEqual(len(summary), 2)


if __name__ == "__main__":
    unittest.main()
