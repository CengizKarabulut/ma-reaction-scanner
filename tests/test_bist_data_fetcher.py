import unittest
from unittest.mock import patch

from scanner import bist_data_fetcher


class BistDataFetcherTests(unittest.TestCase):
    def test_small_official_sector_indices_do_not_fall_through_to_bad_scrape(self):
        with (
            patch.object(
                bist_data_fetcher,
                "fetch_borsapy",
                return_value=(["TCELL", "TTKOM"], "borsapy.component_symbols"),
            ),
            patch.object(bist_data_fetcher, "fetch_isyatirim") as fallback,
        ):
            result = bist_data_fetcher.fetch_index_components("XILTM", verbose=False)

        self.assertTrue(result["verified"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["source"], "borsapy.component_symbols")
        fallback.assert_not_called()

    def test_weekly_update_catalog_contains_all_friendly_sector_indices(self):
        expected = set(bist_data_fetcher.SECTOR_INDEX_NAMES)
        with (
            patch.object(
                bist_data_fetcher, "update_index", return_value=True
            ) as update,
            patch.object(bist_data_fetcher.time, "sleep"),
        ):
            result = bist_data_fetcher.update_all(verbose=False)
        updated = {call.args[0] for call in update.call_args_list}
        self.assertTrue(expected <= updated)
        self.assertEqual(set(result["failed"]), set())


if __name__ == "__main__":
    unittest.main()
