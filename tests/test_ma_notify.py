import unittest

import pandas as pd

from scanner.ma_notify import format_summary


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


if __name__ == "__main__":
    unittest.main()