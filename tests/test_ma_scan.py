import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from scanner.ma_engine import MA_TYPES, TIMEFRAMES
from scanner.ma_levels import LevelConfig
from scanner.ma_watchlist import WatchlistConfig
from scanner.ma_scan import (
    build_parser,
    build_single_stock_table,
    main,
    merge_outputs,
    parse_level_config_json,
    parse_ma_types,
    parse_periods,
    parse_timeframes,
)


class ScannerInputTests(unittest.TestCase):
    def test_all_timeframe_preset_contains_eight_options(self):
        self.assertEqual(parse_timeframes("all"), TIMEFRAMES)

    def test_custom_timeframes_preserve_user_selection(self):
        self.assertEqual(parse_timeframes("15m,1h,4h,1d"), ("15m", "1h", "4h", "1d"))

    def test_all_ma_types_are_accepted(self):
        self.assertEqual(parse_ma_types(",".join(MA_TYPES)), MA_TYPES)

    def test_periods_are_fully_user_configurable_and_deduplicated(self):
        self.assertEqual(parse_periods("7,14,28,14"), (7, 14, 28))

    def test_level_config_json_accepts_known_overrides(self):
        self.assertEqual(
            parse_level_config_json('{"evidence_target_touches": 30, "reaction_bars": 12}'),
            {"evidence_target_touches": 30, "reaction_bars": 12},
        )

    def test_level_config_json_coerces_numeric_strings(self):
        self.assertEqual(
            parse_level_config_json('{"evidence_target_touches":"30","break_atr":"0.75"}'),
            {"evidence_target_touches": 30, "break_atr": 0.75},
        )

    def test_level_config_json_accepts_calibration_overrides(self):
        self.assertEqual(
            parse_level_config_json(
                '{"cross_damping":"0.5","strong_threshold":60,'
                '"level_threshold":45,"weak_threshold":30}'
            ),
            {
                "cross_damping": 0.5,
                "strong_threshold": 60.0,
                "level_threshold": 45.0,
                "weak_threshold": 30.0,
            },
        )

    def test_cli_level_defaults_follow_level_config(self):
        args = build_parser().parse_args([])
        defaults = LevelConfig()

        self.assertEqual(args.reaction_bars, defaults.reaction_bars)
        self.assertEqual(args.hold_bars, defaults.hold_bars)
        self.assertEqual(args.break_atr, defaults.break_atr)
        self.assertEqual(args.bounce_cap_atr, defaults.bounce_cap_atr)
        self.assertEqual(args.cross_cap_per_100, defaults.cross_cap_per_100)
        self.assertEqual(args.cross_damping, defaults.cross_damping)
        self.assertEqual(args.evidence_target_touches, defaults.evidence_target_touches)
        self.assertEqual(args.neighbor_ratio, defaults.neighbor_ratio)
        self.assertEqual(args.strong_threshold, defaults.strong_threshold)
        self.assertEqual(args.level_threshold, defaults.level_threshold)
        self.assertEqual(args.weak_threshold, defaults.weak_threshold)

    def test_cli_watchlist_defaults_follow_config(self):
        args = build_parser().parse_args([])
        defaults = WatchlistConfig()

        self.assertEqual(args.watch_cluster_atr, defaults.cluster_atr)
        self.assertEqual(args.watch_min_touches, defaults.min_touches)
        self.assertEqual(args.watch_min_score, defaults.min_level_score)
        self.assertEqual(args.watch_min_plateau, defaults.min_plateau_ratio)
        self.assertEqual(args.watch_max_zones, defaults.max_zones_per_side)
        self.assertEqual(args.watch_max_distance_atr, defaults.max_distance_atr)
        self.assertFalse(args.watch_require_positive_adherence)

    def test_level_config_json_rejects_non_integer_int_fields(self):
        with self.assertRaisesRegex(ValueError, "tam sayi"):
            parse_level_config_json('{"reaction_bars": 12.5}')

    def test_level_config_json_rejects_boolean_values(self):
        with self.assertRaisesRegex(ValueError, "sayisal deger"):
            parse_level_config_json('{"break_atr": true}')

    def test_level_config_json_rejects_non_finite_float_values(self):
        with self.assertRaisesRegex(ValueError, "sonlu sayi"):
            parse_level_config_json('{"break_atr": "nan"}')

    def test_level_config_json_rejects_unknown_overrides(self):
        with self.assertRaisesRegex(ValueError, "bilinmeyen alan"):
            parse_level_config_json('{"foo": 1}')

    def test_concatenated_period_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "commas"):
            parse_periods("58101320212234505589100144200233377")

    def test_merge_rejects_missing_shards(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shard-0"
            shard.mkdir()
            (shard / "ma_detail.csv").write_text("symbol\nASELS\n", encoding="utf-8")
            (shard / "run_config.json").write_text(
                '{"shard_index": 0}', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "Missing shards"):
                merge_outputs(root, root / "output", expected_shards=2)

    def test_main_fails_when_every_request_errors(self):
        instrument = SimpleNamespace(
            symbol="ASELS",
            asset_class="stock",
            market="BIST",
            asset_label="Hisse",
            display_name="ASELS",
            sector="",
            industry="",
            index_memberships=(),
        )
        with TemporaryDirectory() as temporary:
            with (
                patch("scanner.ma_scan.resolve_instruments", return_value=[instrument]),
                patch("scanner.ma_scan.MarketDataProvider") as provider_class,
            ):
                provider_class.return_value.fetch.side_effect = RuntimeError("offline")
                code = main(
                    [
                        "--timeframes",
                        "1d",
                        "--output-dir",
                        str(Path(temporary) / "output"),
                    ]
                )
        self.assertEqual(code, 1)

    def test_main_accepts_valid_empty_analysis(self):
        instrument = SimpleNamespace(
            symbol="ASELS",
            asset_class="stock",
            market="BIST",
            asset_label="Hisse",
            display_name="ASELS",
            sector="",
            industry="",
            index_memberships=(),
        )
        fetched = SimpleNamespace(
            frame=pd.DataFrame(),
            source="test",
            fingerprint="abc",
        )
        with TemporaryDirectory() as temporary:
            with (
                patch("scanner.ma_scan.resolve_instruments", return_value=[instrument]),
                patch("scanner.ma_scan.MarketDataProvider") as provider_class,
                patch("scanner.ma_scan.scan_frame", return_value=pd.DataFrame()),
            ):
                provider_class.return_value.fetch.return_value = fetched
                output = Path(temporary) / "output"
                code = main(
                    [
                        "--timeframes",
                        "1d",
                        "--output-dir",
                        str(output),
                    ]
                )
                self.assertEqual(code, 0)
                self.assertTrue((output / "ma_watchlist.csv").exists())
                self.assertTrue((output / "watchlist.csv").exists())

    def test_single_stock_table_keeps_each_selected_ma_side(self):
        detail = pd.DataFrame(
            [
                {
                    "timeframe": "1d", "ma_type": "SMA", "period": 233,
                    "ma": "SMA233", "side": "Destek", "active_side": True,
                    "trend_state": "Yükselen", "current_price": 52.85,
                    "current_ma": 53.42, "touches": 18,
                    "side_adherence_pct": 81.5, "wrong_side_pct": 18.5,
                    "cross_count": 7, "win_rate_pct": 64.0,
                    "median_net_r": 0.8, "edge_r": 0.3,
                    "positive_periods": 3, "compatibility": "Güçlü uyum",
                    "distance_pct": 1.08, "distance_atr": 0.45,
                    "filter_pass": True, "filter_status": "Uygun",
                    "filter_reasons": "",
                },
                {
                    "timeframe": "1d", "ma_type": "SMA", "period": 233,
                    "ma": "SMA233", "side": "Direnç", "active_side": False,
                    "trend_state": "Yükselen", "current_price": 52.85,
                    "current_ma": 53.42, "touches": 4,
                    "side_adherence_pct": 18.5, "wrong_side_pct": 81.5,
                    "cross_count": 7, "win_rate_pct": 40.0,
                    "median_net_r": -0.2, "edge_r": -0.1,
                    "positive_periods": 1, "compatibility": "İzleme",
                    "distance_pct": 1.08, "distance_atr": 0.45,
                    "filter_pass": True, "filter_status": "Uygun",
                    "filter_reasons": "",
                },
            ]
        )

        table = build_single_stock_table(detail)

        self.assertEqual(len(table), 2)
        self.assertEqual(table.loc[0, "Taraf"], "Destek")
        self.assertEqual(table.loc[0, "Temas"], 18)
        self.assertEqual(table.loc[0, "Taraf Koruma %"], 81.5)
        self.assertEqual(table.loc[0, "Güncel Rol"], "Aktif")
    def test_single_stock_table_marks_requested_combination_without_history(self):
        detail = pd.DataFrame(
            [
                {
                    "timeframe": "1mo", "ma_type": "SMA", "period": 55,
                    "ma": "SMA55", "side": "Destek", "active_side": True,
                    "compatibility": "Uyumlu", "compatibility_score": 70.0,
                    "touches": 14, "positive_periods": 3, "edge_r": 0.3,
                    "median_net_r": 0.4, "distance_atr": 0.2,
                    "filter_pass": True,
                }
            ]
        )

        table = build_single_stock_table(
            detail,
            timeframes=["1mo"],
            ma_types=["SMA"],
            periods=[55, 377],
        )

        missing = table[table["MA"] == "SMA377"].iloc[0]
        self.assertEqual(missing["Taraf"], "Veri yok")
        self.assertEqual(missing["Uyum"], "Yetersiz veri")
        self.assertEqual(missing["Filtre Nedeni"], "Seçilen MA için yeterli geçmiş mum yok")
if __name__ == "__main__":
    unittest.main()
