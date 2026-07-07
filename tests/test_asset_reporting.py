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
