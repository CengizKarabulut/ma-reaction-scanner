import unittest

from scanner.asset_universe import build_custom_instruments, resolve_universe
from scanner.bist_classification import resolve_sector_choice
from scanner.stock_metadata import (
    enrich_stock_instruments,
    format_index_memberships,
    invert_index_components,
)


class StockMetadataTests(unittest.TestCase):
    def test_friendly_sector_selects_components_without_code_guessing(self):
        calls = []

        def loader(name):
            calls.append(name)
            return ["GARAN", "AKBNK", "GARAN"]

        instruments = resolve_universe(
            "bist_sector_stocks", sector="Bankacılık", stock_loader=loader
        )
        self.assertEqual(calls, ["BIST_BILES:XBANK"])
        self.assertEqual([item.symbol for item in instruments], ["GARAN", "AKBNK"])
        self.assertEqual({item.sector for item in instruments}, {"Bankacılık"})
        self.assertEqual(instruments[0].index_memberships, ("XBANK",))

    def test_turkish_sector_aliases_are_normalized(self):
        self.assertEqual(resolve_sector_choice("gida").index_symbol, "XGIDA")
        self.assertEqual(resolve_sector_choice("ulaşım").index_symbol, "XULAS")
        self.assertEqual(resolve_sector_choice("XUTEK").label, "Teknoloji")

    def test_memberships_are_inverted_and_enriched_once(self):
        instruments = build_custom_instruments(
            ["GARAN", "THYAO"], asset_class="stock", market="BIST"
        )
        components = {
            "XU030": {"GARAN", "THYAO"},
            "XU100": {"GARAN", "THYAO"},
            "XBANK": {"GARAN"},
            "XUMAL": {"GARAN"},
            "XULAS": {"THYAO"},
        }
        enriched = enrich_stock_instruments(
            instruments,
            index_components=components,
            stock_details={"GARAN": {"name": "Garanti BBVA"}},
        )
        by_symbol = {item.symbol: item for item in enriched}
        self.assertEqual(
            by_symbol["GARAN"].index_memberships,
            ("XU030", "XU100", "XBANK", "XUMAL"),
        )
        self.assertEqual(by_symbol["GARAN"].sector, "Bankacılık / Mali")
        self.assertEqual(by_symbol["GARAN"].display_name, "Garanti BBVA")
        self.assertEqual(by_symbol["THYAO"].sector, "Ulaştırma")

    def test_membership_formatter_keeps_codes_and_names(self):
        text = format_index_memberships(("XU030", "XBANK"))
        self.assertEqual(text, "XU030 (BIST 30); XBANK (BIST Banka)")

    def test_inversion_deduplicates_members(self):
        inverted = invert_index_components({"XU100": ["GARAN", "GARAN"]})
        self.assertEqual(inverted, {"GARAN": ("XU100",)})


if __name__ == "__main__":
    unittest.main()
