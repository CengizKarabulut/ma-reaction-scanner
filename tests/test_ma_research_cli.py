import unittest

from scanner.ma_research_cli import _cap_fast_periods, _resolve_instruments, build_parser


class ResearchCliBudgetTests(unittest.TestCase):
    def test_fast_large_scan_prefers_full_operational_period_core(self):
        periods = [5, 8, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200, 233, 377]

        self.assertEqual(_cap_fast_periods(periods, 100, 1), [20, 50, 100, 200])

    def test_fast_large_scan_tops_up_partial_core_periods(self):
        periods = [5, 8, 13, 20, 21, 34, 55, 89]

        self.assertEqual(_cap_fast_periods(periods, 100, 1), [20, 5, 8, 13, 21, 34])

    def test_fast_large_scan_without_core_uses_first_periods(self):
        periods = [5, 8, 13, 21, 34, 55, 89]

        self.assertEqual(_cap_fast_periods(periods, 100, 1), [5, 8, 13, 21, 34, 55])

    def test_fast_small_scan_keeps_research_periods(self):
        periods = [5, 8, 13, 20, 21, 34, 50, 55]

        self.assertEqual(_cap_fast_periods(periods, 1, 1), periods)


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
