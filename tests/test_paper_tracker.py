from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scanner.paper_tracker import (
    advance_watchlist,
    append_watchlist,
    build_watchlist,
    ledger_summary,
)
from scanner.ma_core import TIMEFRAME_CONFIGS


class PaperTrackerTests(unittest.TestCase):
    def test_only_certified_rows_become_watches_by_default(self):
        panel = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "timeframe": "1d",
                    "side": "support",
                    "ma_type": "SMA",
                    "period": 20,
                    "current_ma": 100,
                    "current_price": 102,
                    "q_value": 0.02,
                    "certified": True,
                },
                {
                    "ticker": "TEST",
                    "timeframe": "1d",
                    "side": "resistance",
                    "ma_type": "EMA",
                    "period": 50,
                    "current_ma": 110,
                    "current_price": 102,
                    "q_value": 0.80,
                    "certified": False,
                },
            ]
        )
        ledger = build_watchlist(panel, created_at="2025-01-01")
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger.iloc[0]["state"], "WATCHING")

    def test_active_duplicate_is_not_reenrolled(self):
        panel = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "timeframe": "1d",
                    "side": "support",
                    "ma_type": "SMA",
                    "period": 20,
                    "current_ma": 100,
                    "current_price": 102,
                    "q_value": 0.02,
                    "certified": True,
                }
            ]
        )
        first = build_watchlist(panel, created_at="2025-01-01", cohort_id="pilot-v1")
        duplicate = first.copy()
        duplicate.loc[0, "signal_id"] = "different-signal-id"
        duplicate.loc[0, "created_at"] = "2025-02-01"
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ledger.csv"
            stored = append_watchlist(path, first)
            stored = append_watchlist(path, duplicate)
            self.assertEqual(len(stored), 1)

            stored.loc[0, "state"] = "RESOLVED"
            stored.to_csv(path, index=False)
            stored = append_watchlist(path, duplicate)
            self.assertEqual(len(stored), 2)

    def test_candidate_comparison_rows_keep_their_evidence_label(self):
        panel = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "timeframe": "1d",
                    "side": "support",
                    "ma_type": "SMA",
                    "period": 20,
                    "current_ma": 100,
                    "current_price": 102,
                    "q_value": 0.02,
                    "certified": "True",
                },
                {
                    "ticker": "TEST",
                    "timeframe": "1d",
                    "side": "resistance",
                    "ma_type": "EMA",
                    "period": 50,
                    "current_ma": 110,
                    "current_price": 102,
                    "q_value": 0.80,
                    "certified": "False",
                },
            ]
        )
        ledger = build_watchlist(panel, include_candidates=True, cohort_id="pilot-v1")
        self.assertEqual(
            set(ledger["evidence_status"]), {"CERTIFIED", "CANDIDATE_ONLY"}
        )
        self.assertEqual(set(ledger["cohort_id"]), {"pilot-v1"})
        ledger["state"] = "RESOLVED"
        ledger["outcome"] = ["TARGET", "STOP"]
        ledger["fixed_return_atr"] = [1.0, -1.0]
        summary = ledger_summary(ledger)
        self.assertEqual(
            set(summary["evidence_status"]), {"CERTIFIED", "CANDIDATE_ONLY"}
        )
        legacy = ledger.drop(
            columns=["cohort_id", "evidence_status", "net_fixed_return_atr"]
        )
        legacy_summary = ledger_summary(legacy)
        self.assertEqual(set(legacy_summary["cohort_id"]), {"legacy"})
        self.assertEqual(set(legacy_summary["evidence_status"]), {"UNLABELLED"})

    def test_scan_provenance_freezes_configuration_on_each_watch(self):
        panel = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "timeframe": "1d",
                    "side": "support",
                    "ma_type": "SMA",
                    "period": 20,
                    "current_ma": 100,
                    "current_price": 102,
                    "q_value": 0.02,
                    "certified": True,
                }
            ]
        )
        frozen = replace(TIMEFRAME_CONFIGS["1d"], horizon=3, target_atr=1.25)
        provenance = {
            ("TEST", "1d"): {
                "config": asdict(frozen),
                "fingerprint": "abc123",
                "last_bar": "2025-01-31",
            }
        }
        ledger = build_watchlist(panel, cohort_id="pilot-v1", provenance=provenance)
        stored = json.loads(ledger.iloc[0]["config_json"])
        self.assertEqual(stored["horizon"], 3)
        self.assertEqual(stored["target_atr"], 1.25)
        self.assertEqual(ledger.iloc[0]["source_fingerprint"], "abc123")
        self.assertEqual(ledger.iloc[0]["scan_data_end"], "2025-01-31")

    def test_future_touch_resolves_without_using_old_events(self):
        cycle = [105, 104, 103, 100, 102, 104, 105, 105, 104, 103, 100, 102, 104, 105]
        close = np.asarray(cycle * 12, dtype=float)
        index = pd.date_range("2025-01-01", periods=len(close), freq="D")
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.2,
                "Low": close - 0.2,
                "Close": close,
                "Volume": 1000.0,
            },
            index=index,
        )
        panel = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "timeframe": "1d",
                    "side": "support",
                    "ma_type": "SMA",
                    "period": 20,
                    "current_ma": 100,
                    "current_price": 105,
                    "q_value": 0.01,
                    "certified": True,
                }
            ]
        )
        ledger = build_watchlist(
            panel,
            created_at="2025-01-01",
            watch_after={("TEST", "1d"): str(index[40])},
            roundtrip_cost_bps=25.0,
        )
        constant_ma = pd.Series(100.0, index=index)
        with patch("scanner.paper_tracker.compute_ma", return_value=constant_ma):
            advanced = advance_watchlist(ledger, lambda ticker, tf: frame)
        self.assertEqual(advanced.iloc[0]["state"], "RESOLVED")
        self.assertGreater(pd.Timestamp(advanced.iloc[0]["trigger_time"]), index[40])
        self.assertLess(
            advanced.iloc[0]["net_fixed_return_atr"],
            advanced.iloc[0]["fixed_return_atr"],
        )
        summary = ledger_summary(advanced)
        self.assertEqual(int(summary.iloc[0]["resolved"]), 1)


if __name__ == "__main__":
    unittest.main()
