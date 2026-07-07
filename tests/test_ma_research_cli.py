import unittest

from scanner.ma_research_cli import _resolve_instruments, build_parser


class ResearchCliAssetTests(unittest.TestCase):
    def test_automatic_index_universe_needs_no_symbol_codes(self):
        args = build_parser().parse_args(["--universe", "bist_main_indices"])
        instruments = _resolve_instruments(args)
        self.assertEqual(
            [item.symbol for item in instruments], ["XU030", "XU050", "XU100", "XUTUM"]
        )
        self.assertEqual({item.asset_label for item in instruments}, {"Endeks"})

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
