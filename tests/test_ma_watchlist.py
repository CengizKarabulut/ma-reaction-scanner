import unittest

import pandas as pd

from scanner.ma_watchlist import (
    WatchlistConfig,
    apply_quality_gate,
    build_watchlist,
    format_watchlist_text,
)


def row(ma, ma_type, current_ma, distance_atr, **kwargs):
    base = {
        "symbol": "ASELS",
        "timeframe": "1d",
        "side": "Destek",
        "ma": ma,
        "ma_type": ma_type,
        "period": int("".join(c for c in ma if c.isdigit()) or 0),
        "current_ma": current_ma,
        "current_price": 336.25,
        "distance_atr": distance_atr,
        "distance_pct": distance_atr * 2.0,
        "active_side": True,
        "filter_pass": True,
        "level_touches": 10,
        "level_score": 50.0,
        "hold_rate_pct": 70.0,
        "median_bounce_atr": 2.0,
        "plateau_ratio": 1.0,
        "plateau_neighbors": 2,
        "adherence_excess_pct": 5.0,
        "analysis_basis": "nominal",
    }
    base.update(kwargs)
    return base


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.config = WatchlistConfig(min_touches=5, min_level_score=35.0)

    def test_inactive_side_is_dropped(self):
        frame = pd.DataFrame([row("SMA200", "SMA", 309.0, -1.0, active_side=False)])
        self.assertEqual(len(apply_quality_gate(frame, self.config)), 0)

    def test_thin_evidence_is_dropped(self):
        frame = pd.DataFrame([row("SMA200", "SMA", 309.0, -1.0, level_touches=2)])
        self.assertEqual(len(apply_quality_gate(frame, self.config)), 0)

    def test_isolated_peak_is_dropped_but_untestable_row_is_kept(self):
        frame = pd.DataFrame(
            [
                row("SMA200", "SMA", 309.0, -1.0, plateau_ratio=0.2, plateau_neighbors=2),
                row("SMA377", "SMA", 232.0, -3.0, plateau_ratio=float("nan"), plateau_neighbors=0),
            ]
        )
        kept = apply_quality_gate(frame, self.config)
        self.assertEqual(list(kept["ma"]), ["SMA377"])

    def test_absurdly_distant_levels_are_dropped(self):
        frame = pd.DataFrame([row("SMA377", "SMA", 100.0, -20.0)])
        self.assertEqual(len(apply_quality_gate(frame, WatchlistConfig(max_distance_atr=8.0))), 0)


class ClusteringTests(unittest.TestCase):
    def test_nearby_averages_collapse_into_one_zone(self):
        frame = pd.DataFrame(
            [
                row("SMA200", "SMA", 309.00, -1.20),
                row("VWMA200", "VWMA", 309.46, -1.18),
                row("EMA200", "EMA", 316.89, -0.90),
            ]
        )

        watch = build_watchlist(frame, WatchlistConfig(cluster_atr=0.50))

        self.assertEqual(len(watch), 1)
        zone = watch.iloc[0]
        self.assertAlmostEqual(zone["zone_low"], 309.00)
        self.assertAlmostEqual(zone["zone_high"], 316.89)
        self.assertEqual(zone["ma_families"], 3)
        self.assertEqual(zone["confidence"], "Guclu")

    def test_distant_averages_stay_separate(self):
        frame = pd.DataFrame(
            [
                row("SMA200", "SMA", 309.0, -1.0),
                row("SMA377", "SMA", 232.0, -5.0),
            ]
        )

        watch = build_watchlist(frame, WatchlistConfig(cluster_atr=0.50))

        self.assertEqual(len(watch), 2)

    def test_touches_are_aggregated_with_max_not_sum(self):
        frame = pd.DataFrame(
            [
                row("SMA200", "SMA", 309.0, -1.00, level_touches=8),
                row("EMA200", "EMA", 309.5, -0.98, level_touches=12),
            ]
        )

        watch = build_watchlist(frame, WatchlistConfig(cluster_atr=0.50))

        # The same swings produced both counts; adding them would double count.
        self.assertEqual(float(watch.iloc[0]["level_touches"]), 12.0)

    def test_current_price_is_preserved_for_watchlist_images(self):
        frame = pd.DataFrame(
            [
                row("SMA200", "SMA", 309.0, -1.00, current_price=336.25),
                row("EMA200", "EMA", 309.5, -0.98, current_price=336.25),
            ]
        )

        watch = build_watchlist(frame, WatchlistConfig(cluster_atr=0.50))

        self.assertAlmostEqual(float(watch.iloc[0]["current_price"]), 336.25)

    def test_zones_are_ordered_by_proximity_not_by_score(self):
        frame = pd.DataFrame(
            [
                row("SMA377", "SMA", 232.0, -5.0, level_score=90.0),
                row("SMA200", "SMA", 309.0, -1.0, level_score=40.0),
            ]
        )

        watch = build_watchlist(frame, WatchlistConfig(cluster_atr=0.50))

        self.assertEqual(watch.iloc[0]["ma_list"], "SMA200")

    def test_zone_count_is_capped_per_side(self):
        frame = pd.DataFrame(
            [row(f"SMA{p}", "SMA", 300.0 - i * 20, -1.0 - i * 2.0)
             for i, p in enumerate([50, 100, 200, 233, 377])]
        )

        watch = build_watchlist(frame, WatchlistConfig(cluster_atr=0.50, max_zones_per_side=3))

        self.assertEqual(len(watch), 3)

    def test_support_and_resistance_are_capped_independently(self):
        rows = [row(f"SMA{p}", "SMA", 300.0 - i * 20, -1.0 - i * 2.0)
                for i, p in enumerate([50, 100, 200, 233])]
        rows += [row(f"EMA{p}", "EMA", 350.0 + i * 20, 1.0 + i * 2.0, side="Direnç")
                 for i, p in enumerate([50, 100, 200, 233])]

        watch = build_watchlist(pd.DataFrame(rows), WatchlistConfig(max_zones_per_side=2))

        self.assertEqual(len(watch), 4)
        self.assertEqual(watch["side"].value_counts().to_dict(), {"Destek": 2, "Direnç": 2})


class RenderingTests(unittest.TestCase):
    def test_empty_watchlist_says_so_instead_of_printing_nothing(self):
        text = format_watchlist_text(pd.DataFrame(), symbol="ASELS", price=336.25)

        self.assertIn("ASELS", text)
        self.assertIn("seviye yok", text)

    def test_rendered_block_contains_band_and_members(self):
        frame = pd.DataFrame(
            [
                row("SMA200", "SMA", 309.00, -1.20),
                row("EMA200", "EMA", 316.89, -0.90),
            ]
        )
        watch = build_watchlist(frame, WatchlistConfig())

        text = format_watchlist_text(watch, symbol="ASELS", price=336.25, timeframe="1d")

        self.assertIn("309.00-316.89", text)
        self.assertIn("SMA200", text)
        self.assertIn("EMA200", text)

    def test_rendered_rows_include_timeframe_for_multi_timeframe_scans(self):
        watch = pd.DataFrame(
            [
                {
                    "side": "Destek",
                    "timeframe": "1d",
                    "zone_low": 309.0,
                    "zone_high": 316.89,
                    "distance_pct": -8.0,
                    "confidence": "Orta",
                    "level_touches": 10,
                    "hold_rate_pct": 56.0,
                    "ma_list": "SMA200, EMA200",
                },
                {
                    "side": "Destek",
                    "timeframe": "4h",
                    "zone_low": 330.0,
                    "zone_high": 331.0,
                    "distance_pct": -1.0,
                    "confidence": "Guclu",
                    "level_touches": 18,
                    "hold_rate_pct": 72.0,
                    "ma_list": "HMA55",
                },
            ]
        )

        text = format_watchlist_text(watch)

        self.assertIn("1d", text)
        self.assertIn("4h", text)
        self.assertIn("HMA55", text)

    def test_missing_columns_return_an_empty_shaped_frame(self):
        watch = build_watchlist(pd.DataFrame({"symbol": ["X"]}), WatchlistConfig())

        self.assertEqual(len(watch), 0)
        self.assertIn("confidence", watch.columns)


if __name__ == "__main__":
    unittest.main()
