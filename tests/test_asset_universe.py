import unittest

from scanner.asset_universe import (
    ASSET_LABELS,
    build_custom_instruments,
    list_universes,
    resolve_universe,
)


class AssetUniverseTests(unittest.TestCase):
    def test_universe_catalog_separates_asset_classes(self):
        catalog = {row["key"]: row for row in list_universes()}
        self.assertEqual(catalog["bist100_stocks"]["asset_label"], "Hisse")
        self.assertEqual(catalog["bist_main_indices"]["asset_label"], "Endeks")
        self.assertEqual(
            catalog["bist_sector_indices"]["asset_label"], "Sektör Endeksi"
        )
        self.assertEqual(catalog["crypto_majors"]["asset_label"], "Kripto")
        self.assertEqual(catalog["commodities_majors"]["asset_label"], "Emtia")

    def test_dynamic_stock_members_are_deduplicated(self):
        instruments = resolve_universe(
            "bist30_stocks", stock_loader=lambda _: ["GARAN", "AKBNK", "GARAN"]
        )
        self.assertEqual([item.symbol for item in instruments], ["GARAN", "AKBNK"])
        self.assertTrue(all(item.asset_class == "stock" for item in instruments))

    def test_indices_are_not_labelled_as_stocks(self):
        instruments = resolve_universe("bist_main_indices")
        self.assertEqual(len(instruments), 4)
        self.assertEqual({item.asset_label for item in instruments}, {"Endeks"})
        self.assertEqual(len({item.symbol for item in instruments}), len(instruments))

    def test_custom_crypto_symbols_keep_provider_format(self):
        instruments = build_custom_instruments(
            ["BTC-USD", "ETH-USD", "BTC-USD"], asset_class="crypto"
        )
        self.assertEqual(
            [item.provider_symbol for item in instruments], ["BTC-USD", "ETH-USD"]
        )
        self.assertEqual(ASSET_LABELS[instruments[0].asset_class], "Kripto")


if __name__ == "__main__":
    unittest.main()
