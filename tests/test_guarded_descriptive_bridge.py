import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scanner.guarded_descriptive_bridge import (
    HighlightInstrument,
    collect_highlight_instruments,
    write_manifest,
)


class GuardedDescriptiveBridgeTests(unittest.TestCase):
    def test_collects_behavior_highlights_before_guarded_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            pd.DataFrame(
                [
                    {"asset_class": "stock", "symbol": "AAA", "total_touch_events": 12},
                    {"asset_class": "stock", "symbol": "LOW", "total_touch_events": 9},
                ]
            ).to_csv(folder / "ma_behavior_near_price.csv", index=False)
            pd.DataFrame(
                [
                    {"asset_class": "crypto", "symbol": "BTC-USD", "total_touch_events": 22},
                    {"asset_class": "stock", "symbol": "AAA", "total_touch_events": 30},
                ]
            ).to_csv(folder / "ma_behavior_best_reactions.csv", index=False)
            pd.DataFrame(
                [
                    {"asset_class": "index", "symbol": "XU100", "total_touch_events": 18},
                ]
            ).to_csv(folder / "ma_behavior_most_visited.csv", index=False)
            pd.DataFrame(
                [
                    {"asset_class": "stock", "symbol": "CCC"},
                    {"asset_class": "stock", "symbol": "DDD"},
                ]
            ).to_csv(folder / "instrument_summary.csv", index=False)

            selected = collect_highlight_instruments(
                folder / "instrument_summary.csv",
                behavior_dir=folder,
                max_symbols=4,
                behavior_min_touches=10,
            )

        self.assertEqual(
            [(item.asset_class, item.market, item.symbol, item.source) for item in selected],
            [
                ("stock", "BIST", "AAA", "near_price"),
                ("crypto", "GLOBAL", "BTC-USD", "best_reactions"),
                ("index", "BIST", "XU100", "most_visited"),
                ("stock", "BIST", "CCC", "guarded_summary"),
            ],
        )

    def test_manifest_writes_groupable_symbol_files(self):
        instruments = [
            HighlightInstrument("stock", "ASELS", "BIST", "near_price"),
            HighlightInstrument("crypto", "BTC-USD", "GLOBAL", "best_reactions"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            write_manifest(temp_dir, instruments)
            folder = Path(temp_dir)
            manifest = pd.read_csv(folder / "highlighted_instruments.csv")
            symbols = (folder / "highlighted_symbols.txt").read_text(encoding="utf-8")

        self.assertEqual(manifest["symbol"].tolist(), ["ASELS", "BTC-USD"])
        self.assertEqual(symbols, "ASELS,BTC-USD")


if __name__ == "__main__":
    unittest.main()
