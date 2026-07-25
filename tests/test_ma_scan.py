import unittest

from scanner.ma_engine import MA_TYPES, TIMEFRAMES
from scanner.ma_scan import parse_ma_types, parse_periods, parse_timeframes


class ScannerInputTests(unittest.TestCase):
    def test_all_timeframe_preset_contains_eight_options(self):
        self.assertEqual(parse_timeframes("all"), TIMEFRAMES)

    def test_custom_timeframes_preserve_user_selection(self):
        self.assertEqual(parse_timeframes("15m,1h,4h,1d"), ("15m", "1h", "4h", "1d"))

    def test_all_ma_types_are_accepted(self):
        self.assertEqual(parse_ma_types(",".join(MA_TYPES)), MA_TYPES)

    def test_periods_are_fully_user_configurable_and_deduplicated(self):
        self.assertEqual(parse_periods("7,14,28,14"), (7, 14, 28))


if __name__ == "__main__":
    unittest.main()
