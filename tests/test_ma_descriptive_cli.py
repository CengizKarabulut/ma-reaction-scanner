import unittest

import numpy as np
import pandas as pd

from scanner.ma_core import AnalysisConfig
from scanner.ma_descriptive_cli import (
    DEFAULT_DESC_PERIODS,
    _format_episode_range,
    _format_event_timestamp,
    _resolve_instruments,
    build_parser,
    crossing_episodes,
    format_report,
    scan_ma_respect,
)


def _synthetic_frame(length: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=length, freq="D")
    x = np.arange(length, dtype=float)
    close = 100.0 + 0.04 * x + 4.0 * np.sin(x / 4.0)
    return pd.DataFrame(
        {
            "Open": close + 0.15 * np.sin(x),
            "High": close + 1.8,
            "Low": close - 1.8,
            "Close": close,
            "Volume": np.full(length, 1_000_000.0),
        },
        index=idx,
    )


class DescriptiveMaCliTests(unittest.TestCase):
    def test_default_periods_match_user_pool_and_remain_editable(self):
        args = build_parser().parse_args(["--ticker", "ASELS"])

        parsed_periods = [int(value) for value in args.periods.split(",")]
        self.assertEqual(parsed_periods, list(DEFAULT_DESC_PERIODS))
        self.assertIn(10, parsed_periods)
        self.assertIn(50, parsed_periods)
        self.assertIn(200, parsed_periods)
        self.assertEqual(len(parsed_periods), 16)
        self.assertEqual(args.ma_types, "SMA,EMA,WMA,VWMA,KAMA,ALMA,HMA")
        self.assertEqual(args.min_visits, 5)
        custom = build_parser().parse_args(["--ticker", "ASELS", "--periods", "5,34,233"])
        self.assertEqual(custom.periods, "5,34,233")

    def test_custom_ticker_list_resolves_multiple_instruments(self):
        args = build_parser().parse_args(
            ["--tickers", "ASELS,THYAO", "--asset-class", "stock", "--market", "BIST"]
        )

        instruments = _resolve_instruments(args)

        self.assertEqual([item.symbol for item in instruments], ["ASELS", "THYAO"])
        self.assertEqual({item.asset_class for item in instruments}, {"stock"})

    def test_crossing_episodes_mark_recovery(self):
        idx = pd.date_range("2025-01-01", periods=6, freq="D")
        state = np.array([False, True, True, False, True, True])
        valid = np.ones(6, dtype=bool)

        episodes = crossing_episodes(idx, state, valid)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].bars, 2)
        self.assertTrue(episodes[0].recovered)
        self.assertFalse(episodes[1].recovered)

    def test_intraday_event_timestamps_keep_hour_and_minute(self):
        ts = pd.Timestamp("2026-07-10 14:30")

        self.assertEqual(_format_event_timestamp(ts, "1h"), "2026-07-10 14:30")
        self.assertEqual(_format_event_timestamp(ts, "4h"), "2026-07-10 14:30")
        self.assertEqual(_format_event_timestamp(ts, "1d"), "2026-07-10")
        self.assertEqual(
            _format_episode_range(ts, pd.Timestamp("2026-07-10 18:30"), "1h"),
            "2026-07-10 14:30→2026-07-10 18:30",
        )

    def test_scan_builds_descriptive_scorecard_without_guard_terms(self):
        cfg = AnalysisConfig(horizon=5, zone_atr=0.8, separation_atr=0.2)
        prepared, scorecard, events, current = scan_ma_respect(
            _synthetic_frame(),
            symbol="ASELS",
            timeframe="1d",
            ma_types=("SMA", "EMA", "HMA"),
            periods=(5, 8, 13),
            side=1,
            config=cfg,
        )

        labels = set(scorecard["MA"].tolist())
        self.assertIn("SMA5", labels)
        self.assertIn("EMA8", labels)
        self.assertIn("HMA13", labels)
        self.assertNotIn("HMA5", labels)
        self.assertFalse(scorecard.empty)
        self.assertFalse(current.empty)
        self.assertIn("ziyaret", scorecard.columns)
        self.assertIn("saygı_skoru", scorecard.columns)
        self.assertEqual(scorecard["ziyaret"].iloc[0], scorecard["ziyaret"].max())

        report = format_report(
            "ASELS",
            "1d",
            prepared,
            scorecard,
            events,
            current,
            min_visits=1,
        )
        forbidden = ("CERTIFIED", "FDR", "holdout", "kanıt kapısı", "sertifika")
        for token in forbidden:
            self.assertNotIn(token, report)

    def test_report_does_not_promote_one_touch_rows(self):
        cfg = AnalysisConfig(horizon=5, zone_atr=0.8, separation_atr=0.2)
        prepared, scorecard, events, current = scan_ma_respect(
            _synthetic_frame(),
            symbol="NETCD",
            timeframe="1d",
            ma_types=("SMA",),
            periods=(5,),
            side=1,
            config=cfg,
        )

        report = format_report(
            "NETCD",
            "1d",
            prepared,
            scorecard,
            events,
            current,
            min_visits=999,
        )

        self.assertIn("999+ temaslı MA bulunamadı", report)
        self.assertIn("Tam liste: ma_respect_scorecard.csv", report)


if __name__ == "__main__":
    unittest.main()
