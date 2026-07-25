import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scanner.ma_engine import MA_TYPES, TIMEFRAMES
from scanner.ma_scan import merge_outputs, parse_ma_types, parse_periods, parse_timeframes


class ScannerInputTests(unittest.TestCase):
    def test_all_timeframe_preset_contains_eight_options(self):
        self.assertEqual(parse_timeframes("all"), TIMEFRAMES)

    def test_custom_timeframes_preserve_user_selection(self):
        self.assertEqual(parse_timeframes("15m,1h,4h,1d"), ("15m", "1h", "4h", "1d"))

    def test_all_ma_types_are_accepted(self):
        self.assertEqual(parse_ma_types(",".join(MA_TYPES)), MA_TYPES)

    def test_periods_are_fully_user_configurable_and_deduplicated(self):
        self.assertEqual(parse_periods("7,14,28,14"), (7, 14, 28))
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


if __name__ == "__main__":
    unittest.main()
