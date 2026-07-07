import unittest

import pandas as pd

from scanner.guarded_notifier import build_guarded_table, select_top_instruments


class GuardedNotifierTests(unittest.TestCase):
    def _summary(self):
        return pd.DataFrame(
            [
                {
                    "symbol": "PASEU",
                    "asset_label": "Hisse",
                    "current_price": 120.0,
                    "tested_level_count": 28,
                    "certified_level_count": 4,
                    "actionable_level_count": 2,
                    "certification_rate_pct": 14.285,
                    "avg_holdout_hit_rate_pct": 72.5,
                    "avg_holdout_return_atr": 0.81,
                },
                {
                    "symbol": "PASEU",
                    "asset_label": "Hisse",
                    "current_price": 120.0,
                    "tested_level_count": 28,
                    "certified_level_count": 1,
                    "actionable_level_count": 0,
                    "certification_rate_pct": 3.57,
                    "avg_holdout_hit_rate_pct": 55.0,
                    "avg_holdout_return_atr": 0.2,
                },
                {
                    "symbol": "HALKB",
                    "asset_label": "Hisse",
                    "current_price": 42.65,
                    "tested_level_count": 28,
                    "certified_level_count": 2,
                    "actionable_level_count": 1,
                    "certification_rate_pct": 7.14,
                    "avg_holdout_hit_rate_pct": 68.0,
                    "avg_holdout_return_atr": 0.55,
                },
            ]
        )

    def test_top_table_has_one_row_per_instrument(self):
        top = select_top_instruments(self._summary(), top_n=20)
        self.assertEqual(top["symbol"].tolist(), ["PASEU", "HALKB"])

    def test_table_includes_aggregate_rates(self):
        headers, rows, _, title = build_guarded_table(self._summary(), top_n=20)
        self.assertIn("Oran", headers)
        self.assertIn("Holdout WR", headers)
        self.assertEqual(rows[0][3], "4/28")
        self.assertEqual(rows[0][4], "14.3%")
        self.assertIn("Tek Satır", title)


if __name__ == "__main__":
    unittest.main()
