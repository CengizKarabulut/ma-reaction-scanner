import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scanner.ma_data import fingerprint_frame


class DataRegressionTests(unittest.TestCase):
    def test_csv_roundtrip_keeps_canonical_fingerprint(self):
        index = pd.date_range("2025-01-01", periods=20, freq="D")
        close = 100.0 + np.arange(20) / 7.0
        frame = pd.DataFrame({
            "Open": close - 0.123456789012,
            "High": close + 0.987654321098,
            "Low": close - 0.987654321098,
            "Close": close,
            "Volume": 1_000_000.123456789,
        }, index=index)
        expected = fingerprint_frame(frame)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.csv.gz"
            frame.to_csv(path, compression="gzip", float_format="%.10f")
            loaded = pd.read_csv(path, index_col=0, parse_dates=True)
        self.assertEqual(expected, fingerprint_frame(loaded))


if __name__ == "__main__":
    unittest.main()

