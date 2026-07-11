import unittest

from scanner.ma_research_cli import DEFAULT_SCAN_PERIODS, _resolve_instruments, build_parser


class ResearchCliBudgetTests(unittest.TestCase):
    def test_default_scan_periods_keep_full_operational_research_list(self):
        args = build_parser().parse_args([])

        parsed_periods = [int(value) for value in args.periods.split(",")]
        self.assertEqual(parsed_periods, list(DEFAULT_SCAN_PERIODS))
        self.assertIn(20, parsed_periods)
        self.assertEqual(args.behavior_min_touches, 10)

    def test_fast_flag_is_backward_compatible_noop(self):
        args = build_parser().parse_args(["--fast", "--periods", "5,8,13,21,34,55"])

        self.assertTrue(args.fast)
        self.assertEqual(args.periods, "5,8,13,21,34,55")


class ResearchCliAssetTests(unittest.TestCase):
    def test_automatic_index_universe_needs_no_symbol_codes(self):
        args = build_parser().parse_args(["--universe", "bist_main_indices"])
        instruments = _resolve_instruments(args)
        self.assertEqual(
            [item.symbol for item in instruments], ["XU030", "XU050", "XU100", "XUTUM"]
        )
        self.assertEqual({item.asset_label for item in instruments}, {"Endeks"})

    def test_all_sector_sentinel_explains_whole_market_choice(self):
        args = build_parser().parse_args(
            [
                "--universe",
                "bist_sector_stocks",
                "--sector",
                "Tümü / uygulanmaz",
            ]
        )
        with self.assertRaisesRegex(ValueError, "bist30/50/100/all_stocks"):
            _resolve_instruments(args)

    def test_custom_symbols_are_typed_and_deduplicated(self):
        args = build_parser().parse_args(
            [
                "--universe",
                "custom",
                "--asset-class",
                "commodity",
                "--market",
                "GLOBAL",
                "--tickers",
                "GC=F,SI=F,GC=F",
            ]
        )
        instruments = _resolve_instruments(args)
        self.assertEqual([item.symbol for item in instruments], ["GC=F", "SI=F"])
        self.assertEqual({item.asset_label for item in instruments}, {"Emtia"})


if __name__ == "__main__":
    unittest.main()
