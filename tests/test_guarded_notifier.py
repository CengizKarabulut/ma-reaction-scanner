from pathlib import Path
import tempfile
import unittest

import pandas as pd

import scanner.guarded_notifier as guarded_notifier
from scanner.guarded_notifier import build_guarded_table, select_top_instruments


def _nearest(distance=0.4, side="support", ma="EMA", period=50, strength=50.0):
    return {
        "nearest_timeframe": "1d",
        "nearest_side": side,
        "nearest_ma": ma,
        "nearest_period": period,
        "nearest_level": 100.0,
        "nearest_distance_pct": distance * 2,
        "nearest_abs_distance_atr": distance,
        "nearest_status": "unverified_candidate",
        "nearest_discovery_events": 12,
        "nearest_total_touch_events": 12,
        "nearest_sr_strength_score": strength,
    }


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
                    "low_confidence_level_count": 0,
                    "actionable_level_count": 2,
                    "certification_rate_pct": 14.285,
                    "max_sr_strength_score": 77.0,
                    "avg_holdout_hit_rate_pct": 72.5,
                    "avg_holdout_return_atr": 0.81,
                    "avg_holdout_net_return_atr": 0.55,
                    **_nearest(),
                },
                {
                    "symbol": "PASEU",
                    "asset_label": "Hisse",
                    "current_price": 120.0,
                    "tested_level_count": 28,
                    "certified_level_count": 1,
                    "low_confidence_level_count": 2,
                    "actionable_level_count": 0,
                    "certification_rate_pct": 3.57,
                    "max_sr_strength_score": 70.0,
                    "avg_holdout_hit_rate_pct": 55.0,
                    "avg_holdout_return_atr": 0.2,
                    "avg_holdout_net_return_atr": -0.1,
                    **_nearest(0.2),
                },
                {
                    "symbol": "HALKB",
                    "asset_label": "Hisse",
                    "current_price": 42.65,
                    "tested_level_count": 28,
                    "certified_level_count": 2,
                    "low_confidence_level_count": 1,
                    "actionable_level_count": 1,
                    "certification_rate_pct": 7.14,
                    "max_sr_strength_score": 66.0,
                    "avg_holdout_hit_rate_pct": 68.0,
                    "avg_holdout_return_atr": 0.55,
                    "avg_holdout_net_return_atr": 0.31,
                    **_nearest(0.3),
                },
            ]
        )

    def test_top_table_has_one_row_per_instrument(self):
        top = select_top_instruments(self._summary(), top_n=20)
        self.assertEqual(top["symbol"].tolist(), ["PASEU", "HALKB"])

    def test_certified_table_includes_aggregate_rates(self):
        headers, rows, _, title = build_guarded_table(self._summary(), top_n=20)
        self.assertIn("Oran", headers)
        self.assertIn("Holdout WR", headers)
        self.assertIn("SR Guc", headers)
        self.assertIn("Net ATR", headers)
        self.assertEqual(rows[0][3], "4+0/28")
        self.assertEqual(rows[0][4], "77.0")
        self.assertEqual(rows[0][5], "14.3%")
        self.assertIn("Tek Satır", title)

    def test_uncertified_fallback_is_ranked_by_candidate_proximity(self):
        rows = []
        for symbol, distance in (("AAA", 2.0), ("ZZZ", 0.1)):
            rows.append(
                {
                    "symbol": symbol,
                    "asset_label": "Hisse",
                    "current_price": 100.0,
                    "tested_level_count": 28,
                    "certified_level_count": 0,
                    "low_confidence_level_count": 0,
                    "actionable_level_count": 0,
                    "certification_rate_pct": 0.0,
                    "max_sr_strength_score": 1.0,
                    "avg_holdout_hit_rate_pct": float("nan"),
                    "avg_holdout_return_atr": float("nan"),
                    "avg_holdout_net_return_atr": float("nan"),
                    **_nearest(distance, side="resistance", ma="HMA", period=20, strength=1.0),
                }
            )
        summary = pd.DataFrame(rows)
        top = select_top_instruments(summary, top_n=20)
        self.assertEqual(top["symbol"].tolist(), ["ZZZ", "AAA"])
        headers, table_rows, _, title = build_guarded_table(summary, top_n=20)
        self.assertIn("En Yakın Aday", headers)
        self.assertEqual(table_rows[0][0], "ZZZ")
        self.assertEqual(table_rows[0][2], "1d:HMA20")
        self.assertIn("En Yakın", title)

    def test_uncertified_fallback_excludes_zero_touch_candidates(self):
        rows = []
        for symbol, touches in (("ZERO", 0), ("LIVE", 12)):
            nearest = _nearest(0.1, side="resistance", ma="EMA", period=55, strength=1.0)
            nearest["nearest_discovery_events"] = touches
            nearest["nearest_total_touch_events"] = touches
            rows.append(
                {
                    "symbol": symbol,
                    "asset_label": "Hisse",
                    "current_price": 100.0,
                    "tested_level_count": 28,
                    "certified_level_count": 0,
                    "low_confidence_level_count": 0,
                    "actionable_level_count": 0,
                    "certification_rate_pct": 0.0,
                    "max_sr_strength_score": 1.0,
                    "avg_holdout_hit_rate_pct": float("nan"),
                    "avg_holdout_return_atr": float("nan"),
                    "avg_holdout_net_return_atr": float("nan"),
                    **nearest,
                }
            )

        top = select_top_instruments(pd.DataFrame(rows), top_n=20)

        self.assertEqual(top["symbol"].tolist(), ["LIVE"])

    def test_behavior_tables_are_sent_when_guarded_fallback_is_empty(self):
        nearest = _nearest(0.08, side="resistance", ma="HMA", period=13, strength=0.0)
        nearest["nearest_discovery_events"] = 0
        nearest["nearest_total_touch_events"] = 0
        summary = pd.DataFrame(
            [
                {
                    "symbol": "NETCD",
                    "asset_label": "Hisse",
                    "current_price": 146.90,
                    "tested_level_count": 61,
                    "certified_level_count": 0,
                    "low_confidence_level_count": 0,
                    "actionable_level_count": 0,
                    "certification_rate_pct": 0.0,
                    "max_sr_strength_score": 0.0,
                    "avg_holdout_hit_rate_pct": float("nan"),
                    "avg_holdout_return_atr": float("nan"),
                    "avg_holdout_net_return_atr": float("nan"),
                    **nearest,
                }
            ]
        )
        behavior = pd.DataFrame(
            [
                {
                    "symbol": "NETCD",
                    "current_price": 146.90,
                    "side": "resistance",
                    "ma_label": "1d:HMA13",
                    "current_ma": 147.69,
                    "abs_distance_atr": 0.08,
                    "distance_pct": 0.54,
                    "total_touch_events": 18,
                    "reaction_hit_rate_pct": 61.0,
                    "reaction_median_fixed_atr": 0.42,
                    "reaction_quality_score": 55.0,
                }
            ]
        )

        messages: list[str] = []
        photos: list[tuple[str, str | None]] = []
        original_send_telegram = guarded_notifier.send_telegram
        original_send_photo = guarded_notifier.send_photo
        original_render_table_image = guarded_notifier.render_table_image
        try:
            guarded_notifier.send_telegram = (
                lambda _token, _chat_id, text, **_kwargs: messages.append(text) or True
            )
            guarded_notifier.send_photo = (
                lambda _token, _chat_id, image, caption=None: photos.append((image, caption))
                or True
            )
            guarded_notifier.render_table_image = (
                lambda _headers, _rows, title, **_kwargs: [f"{title}.png"]
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                behavior.to_csv(Path(temp_dir) / "ma_behavior_most_visited.csv", index=False)

                ok = guarded_notifier.send_guarded_summary(
                    "token",
                    "chat",
                    summary,
                    label="Guarded Tek Varlık: NETCD",
                    top_n=1,
                    behavior_dir=temp_dir,
                )
        finally:
            guarded_notifier.send_telegram = original_send_telegram
            guarded_notifier.send_photo = original_send_photo
            guarded_notifier.render_table_image = original_render_table_image

        self.assertTrue(ok)
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            photos,
            [("En Cok Temas Alan Ortalamalar.png", "En Cok Temas Alan Ortalamalar")],
        )

    def test_empty_behavior_notice_is_sent_when_no_guarded_rows_or_behavior_rows(self):
        nearest = _nearest(0.08, side="resistance", ma="HMA", period=13, strength=0.0)
        nearest["nearest_discovery_events"] = 0
        nearest["nearest_total_touch_events"] = 0
        summary = pd.DataFrame(
            [
                {
                    "symbol": "NETCD",
                    "asset_label": "Hisse",
                    "current_price": 146.90,
                    "tested_level_count": 61,
                    "certified_level_count": 0,
                    "low_confidence_level_count": 0,
                    "actionable_level_count": 0,
                    "certification_rate_pct": 0.0,
                    "max_sr_strength_score": 0.0,
                    "avg_holdout_hit_rate_pct": float("nan"),
                    "avg_holdout_return_atr": float("nan"),
                    "avg_holdout_net_return_atr": float("nan"),
                    **nearest,
                }
            ]
        )

        messages: list[str] = []
        photos: list[tuple[str, str | None]] = []
        original_send_telegram = guarded_notifier.send_telegram
        original_send_photo = guarded_notifier.send_photo
        try:
            guarded_notifier.send_telegram = (
                lambda _token, _chat_id, text, **_kwargs: messages.append(text) or True
            )
            guarded_notifier.send_photo = (
                lambda _token, _chat_id, image, caption=None: photos.append((image, caption))
                or True
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                ok = guarded_notifier.send_guarded_summary(
                    "token",
                    "chat",
                    summary,
                    label="Guarded Tek Varlık: NETCD",
                    top_n=1,
                    behavior_dir=temp_dir,
                )
        finally:
            guarded_notifier.send_telegram = original_send_telegram
            guarded_notifier.send_photo = original_send_photo

        self.assertTrue(ok)
        self.assertEqual(photos, [])
        self.assertTrue(any("minimum 10 ham temas" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
