import unittest

import pandas as pd

from scanner.ma_notify import (
    _single_image_rows,
    build_notification_image,
    format_single_detail,
    format_summary,
    format_watchlist_detail,
)


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
                    "best_touches": 18,
                    "best_side_adherence_pct": 81.5,
                    "best_win_rate_pct": 64.0,
                    "best_median_net_r": 0.8,
                    "best_edge_r": 0.3,
                    "best_compatibility_score": 78.2,
                    "filter_status": "Filtre disi",
                    "filter_reasons": "Likidite, Gap",
                }
            ]
        )

        message = format_summary(frame, "Test", top=1)

        self.assertIn("Disi", message)
        self.assertIn("TEST: Likidite, Gap", message)
        self.assertIn("78.20", message)
        self.assertIn("81.50", message)

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

    def test_watchlist_detail_prefers_compact_zone_block(self):
        frame = pd.DataFrame(
            [
                {
                    "symbol": "ASELS",
                    "timeframe": "1d",
                    "side": "Destek",
                    "zone_low": 309.0,
                    "zone_high": 316.89,
                    "distance_pct": -8.0,
                    "ma_list": "SMA200, EMA200",
                    "level_touches": 10,
                    "hold_rate_pct": 56.0,
                    "confidence": "Orta",
                }
            ]
        )

        message = format_watchlist_detail(frame, "ASELS", top=20)

        self.assertIn("Izleme seti", message)
        self.assertIn("SMA200", message)
        self.assertIn("309.00-316.89", message)
        self.assertIn("<pre>", message)
        self.assertLessEqual(len(message), 4_000)

    def test_watchlist_detail_enforces_budget_with_long_ma_lists(self):
        long_ma_list = ", ".join(f"MA{index}" for index in range(300))
        frame = pd.DataFrame(
            [
                {
                    "symbol": "ASELS",
                    "timeframe": "1d" if index % 2 == 0 else "4h",
                    "side": "Destek" if index % 3 else "Direnc",
                    "zone_low": 300.0 + index,
                    "zone_high": 300.5 + index,
                    "distance_pct": float(index) / 10.0,
                    "ma_list": long_ma_list,
                    "level_touches": 10 + index,
                    "hold_rate_pct": 55.0,
                    "confidence": "Orta",
                }
                for index in range(80)
            ]
        )

        message = format_watchlist_detail(frame, "ASELS", top=0)

        self.assertLessEqual(len(message), 4_000)
        self.assertIn("Telegram limiti", message)
        self.assertIn("... +", message)
        self.assertIn("1d", message)
        self.assertIn("337.00-337.50", message)


    def test_single_image_rows_use_emitted_unicode_headers(self):
        frame = pd.DataFrame(
            [
                {
                    "Zaman Dilimi": "1d",
                    "MA": "SMA233",
                    "Taraf": "Destek",
                    "Temas": 18,
                    "Seviye Skoru": 72.4,
                    "Tutma %": 81.5,
                    "Uzakl?k %": 2.35,
                    "G?ncel Rol": "Aktif",
                }
            ]
        )

        rows = _single_image_rows(frame, top=20)

        self.assertEqual(rows[0]["distance"], "2.4%")
        self.assertEqual(rows[0]["role"], "Aktif")

    def test_watchlist_notification_image_is_png(self):
        frame = pd.DataFrame(
            [
                {
                    "symbol": "ASELS",
                    "timeframe": "1d",
                    "side": "Destek",
                    "zone_low": 309.0,
                    "zone_high": 316.89,
                    "distance_pct": -8.0,
                    "ma_list": "SMA200, EMA200, VWMA200, HMA200, ALMA200",
                    "level_touches": 10,
                    "hold_rate_pct": 56.0,
                    "confidence": "Orta",
                }
            ]
        )

        payload = build_notification_image(frame, "ASELS", top=20, watch_frame=frame)

        self.assertIsNotNone(payload)
        image_bytes, kind = payload
        self.assertEqual(kind, "watchlist")
        self.assertTrue(image_bytes.startswith(b"\x89PNG"))
        self.assertGreater(len(image_bytes), 1_000)


if __name__ == "__main__":
    unittest.main()
