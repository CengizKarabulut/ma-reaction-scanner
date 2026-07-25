import unittest

import pandas as pd

from scanner.ma_notify import format_single_detail, format_summary


class NotificationTests(unittest.TestCase):
    def test_filtered_row_shows_status_and_reason(self):
        frame = pd.DataFrame(
            [
                {
                    "symbol": "TEST",
                    "best_timeframe": "1d",
                    "best_ma": "SMA20",
                    "current_price": 10.0,
                    "best_ma_value": 10.2,
                    "best_distance_pct": 2.0,
                    "best_distance_atr": 1.5,
                    "filter_status": "Filtre disi",
                    "filter_reasons": "Likidite, Gap",
                }
            ]
        )

        message = format_summary(frame, "Test", top=1)

        self.assertIn("Disi", message)
        self.assertIn("TEST: Likidite, Gap", message)

    def test_large_filtered_summary_stays_within_telegram_limit(self):
        frame = pd.DataFrame(
            [
                {
                    "symbol": f"T{index:04d}",
                    "best_timeframe": "1d",
                    "best_ma": "SMA233",
                    "current_price": 123.45,
                    "best_ma_value": 125.67,
                    "best_distance_pct": 1.8,
                    "best_distance_atr": 2.4,
                    "filter_status": "Filtre disi",
                    "filter_reasons": "Likidite, Gap, " + "X" * 150,
                }
                for index in range(50)
            ]
        )

        message = format_summary(frame, "Uzun Test", top=50)

        self.assertLessEqual(len(message), 4_000)
        self.assertIn("</pre>", message)
        self.assertIn("tam CSV'de", message)

    def test_single_detail_lists_selected_combinations(self):
        frame = pd.DataFrame(
            [
                {"Zaman Dilimi": "1d", "MA": "SMA233", "Taraf": "Destek", "Temas": 18,
                 "Taraf Koruma %": 81.5, "Kazanma %": 64.0, "Medyan R": 0.8, "Edge R": 0.3},
                {"Zaman Dilimi": "1h", "MA": "EMA55", "Taraf": "Direnç", "Temas": 11,
                 "Taraf Koruma %": 72.0, "Kazanma %": 55.0, "Medyan R": 0.4, "Edge R": 0.1},
            ]
        )

        message = format_single_detail(frame, "AVPGY", top=20)

        self.assertIn("SMA233", message)
        self.assertIn("EMA55", message)
        self.assertIn("Kor%", message)
        self.assertLessEqual(len(message), 4_000)
if __name__ == "__main__":
    unittest.main()
