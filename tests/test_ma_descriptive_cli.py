import unittest

import numpy as np
import pandas as pd

import scanner.ma_descriptive_cli as desc_cli

from scanner.ma_core import AnalysisConfig
from scanner.ma_descriptive_cli import (
    DEFAULT_DESC_PERIODS,
    _format_episode_range,
    _format_event_timestamp,
    build_ma_dna_profile,
    _resolve_instruments,
    build_parser,
    crossing_episodes,
    format_report,
    render_respect_images,
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
        self.assertEqual(args.side, "auto")
        self.assertEqual(args.top, 0)
        self.assertEqual(args.detail_top, 10)
        labeled = build_parser().parse_args(["--ticker", "ASELS", "--label", "Ek Liste"])
        self.assertEqual(labeled.label, "Ek Liste")
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


    def test_intraday_scan_episode_rows_keep_hour_and_minute(self):
        idx = pd.date_range("2026-07-10 09:00", periods=48, freq="h")
        close = np.full(48, 100.0)
        close[10:12] = 90.0
        close[13:15] = 112.0
        close[25:27] = 88.0
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": close + 2.0,
                "Low": close - 2.0,
                "Close": close,
                "Volume": np.full(48, 1_000_000.0),
            },
            index=idx,
        )

        _, _, events, _ = scan_ma_respect(
            frame,
            symbol="ASELS",
            timeframe="1h",
            ma_types=("SMA",),
            periods=(5,),
            side=1,
            config=AnalysisConfig(horizon=3, zone_atr=0.8, separation_atr=0.1),
        )

        ranges = events[events["tarih"].astype(str).str.contains("\u2192", regex=False)]["tarih"].astype(str)
        self.assertTrue(any(":" in value and " " in value for value in ranges), ranges.tolist())

    def test_respect_visual_renderer_returns_png_when_available(self):
        cfg = AnalysisConfig(horizon=5, zone_atr=0.8, separation_atr=0.2)
        _, scorecard, _, current = scan_ma_respect(
            _synthetic_frame(),
            symbol="ASELS",
            timeframe="1d",
            ma_types=("SMA", "EMA", "HMA"),
            periods=(5, 8, 13),
            side=1,
            config=cfg,
        )

        images = render_respect_images(
            scorecard,
            current,
            label="ASELS",
            timeframe="1d",
            top=3,
            detail_top=2,
            min_visits=1,
            sort_by="visits",
            include_symbol=False,
        )
        if not images:
            self.skipTest("matplotlib unavailable")
        self.assertTrue(images[0].startswith(b"\x89PNG"))


    def test_auto_side_scans_support_and_resistance(self):
        idx = pd.date_range("2025-01-01", periods=80, freq="D")
        close = np.linspace(100.0, 150.0, 80)
        close[-1] = 80.0
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": close + 2.0,
                "Low": close - 2.0,
                "Close": close,
                "Volume": np.full(80, 1_000_000.0),
            },
            index=idx,
        )

        _, scorecard, _, current = scan_ma_respect(
            frame,
            symbol="ASELS",
            timeframe="1d",
            ma_types=("SMA",),
            periods=(5,),
            side=0,
            config=AnalysisConfig(horizon=3, zone_atr=0.8, separation_atr=0.1),
        )

        self.assertEqual(set(scorecard["taraf"]), {"Destek", "Direnç"})
        self.assertEqual(set(current["taraf"]), {"Destek", "Direnç"})
        self.assertEqual(scorecard["MA"].nunique(), 1)

    def test_dna_profile_keeps_support_and_resistance_separate(self):
        scorecard = pd.DataFrame(
            [
                {
                    "symbol": "ASELS",
                    "timeframe": "1d",
                    "taraf": "Destek",
                    "MA": "SMA50",
                    "t\u00fcr": "SMA",
                    "periyot": 50,
                    "fiyat": 100.0,
                    "ma_de\u011feri": 96.0,
                    "ziyaret": 12,
                    "tepki_oran\u0131_%": 67.0,
                    "geri_d\u00f6n\u00fc\u015f_%": 80.0,
                    "ort_tepki_ATR": 0.9,
                    "uzakl\u0131k_%": -4.0,
                    "uzakl\u0131k_ATR": -0.8,
                },
                {
                    "symbol": "ASELS",
                    "timeframe": "1d",
                    "taraf": "Diren\u00e7",
                    "MA": "SMA50",
                    "t\u00fcr": "SMA",
                    "periyot": 50,
                    "fiyat": 100.0,
                    "ma_de\u011feri": 104.0,
                    "ziyaret": 11,
                    "tepki_oran\u0131_%": 64.0,
                    "geri_d\u00f6n\u00fc\u015f_%": 73.0,
                    "ort_tepki_ATR": 0.7,
                    "uzakl\u0131k_%": 4.0,
                    "uzakl\u0131k_ATR": 0.8,
                },
            ]
        )

        dna = build_ma_dna_profile(scorecard, min_visits=1)

        self.assertEqual(len(dna), 2)
        self.assertEqual(set(dna["guncel_taraf"]), {"Destek", "Diren\u00e7"})
        self.assertEqual(dna["MA"].nunique(), 1)

    def test_dna_profile_scores_and_report_block_are_created(self):
        cfg = AnalysisConfig(horizon=5, zone_atr=0.8, separation_atr=0.2)
        prepared, scorecard, events, current = scan_ma_respect(
            _synthetic_frame(),
            symbol="ASELS",
            timeframe="1d",
            ma_types=("SMA", "EMA", "HMA"),
            periods=(5, 8, 13),
            side=0,
            config=cfg,
        )

        dna = build_ma_dna_profile(scorecard, min_visits=1)

        self.assertFalse(dna.empty)
        self.assertIn("dna_skoru", dna.columns)
        self.assertIn("guncel_aksiyon_skoru", dna.columns)
        self.assertIn("dna_sinifi", dna.columns)
        self.assertTrue(dna["dna_skoru"].between(0, 100).all())
        self.assertTrue(dna["guncel_aksiyon_skoru"].between(0, 100).all())
        self.assertTrue(set(dna["dna_sinifi"]).issubset({"Ana DNA", "Guclu", "Izleme", "Zayif"}))

        report = format_report(
            "ASELS",
            "1d",
            prepared,
            scorecard,
            events,
            current,
            dna,
            min_visits=1,
        )
        self.assertIn("MA DNA okumasi", report)
        self.assertIn("DNA skoru gecmis karakteri", report)

    def test_universe_dna_block_ranks_globally_before_truncating(self):
        dna = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "MA": "SMA5",
                    "guncel_taraf": "Destek",
                    "temas": 20,
                    "tepki_%": 55.0,
                    "uzak_%": 1.0,
                    "dna_skoru": 40.0,
                    "guncel_aksiyon_skoru": 35.0,
                    "dna_sinifi": "Izleme",
                    "yorum": "dusuk skor",
                },
                {
                    "symbol": "ZZZ",
                    "MA": "HMA34",
                    "guncel_taraf": "Destek",
                    "temas": 25,
                    "tepki_%": 70.0,
                    "uzak_%": 3.0,
                    "dna_skoru": 88.0,
                    "guncel_aksiyon_skoru": 60.0,
                    "dna_sinifi": "Ana DNA",
                    "yorum": "guclu skor",
                },
            ]
        )

        block = "\n".join(desc_cli._format_dna_block(dna, top=1, include_symbol=True))

        self.assertIn("ZZZ | HMA34", block)
        self.assertNotIn("AAA | SMA5", block)

    def test_universe_dna_image_honors_top_zero(self):
        dna = pd.DataFrame(
            [
                {
                    "symbol": f"SYM{i:02d}",
                    "MA": "SMA5",
                    "guncel_taraf": "Destek",
                    "temas": i + 1,
                    "tepki_%": 50.0,
                    "dna_skoru": float(i),
                    "guncel_aksiyon_skoru": float(i) / 2.0,
                    "dna_sinifi": "Izleme",
                }
                for i in range(45)
            ]
        )
        captured: dict[str, int] = {}
        original = desc_cli.render_respect_table_image

        def fake_renderer(headers, rows, **kwargs):
            captured["row_count"] = len(rows)
            return [b"png"]

        try:
            desc_cli.render_respect_table_image = fake_renderer
            images = desc_cli.render_dna_profile_images(
                dna,
                label="BIST",
                timeframe="1d",
                top=0,
                include_symbol=True,
            )
        finally:
            desc_cli.render_respect_table_image = original

        self.assertEqual(images, [b"png"])
        self.assertEqual(captured["row_count"], 45)

    def test_current_table_prefers_touch_strength_over_raw_proximity(self):
        prepared = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.date_range("2026-01-01", periods=1, freq="D"),
        )
        rows = pd.DataFrame(
            [
                {
                    "symbol": "ASELS",
                    "timeframe": "1d",
                    "taraf": "Destek",
                    "MA": "EMA5",
                    "tür": "EMA",
                    "periyot": 5,
                    "fiyat": 100.0,
                    "ma_değeri": 100.1,
                    "üst/alt_bar_%": 52.0,
                    "ziyaret": 6,
                    "tepki_sayısı": 2,
                    "tepki_oranı_%": 33.0,
                    "sarkma_epizodu": 3,
                    "geri_dönen": 1,
                    "geri_dönüş_%": 33.0,
                    "ort_sarkma_bar": 4.0,
                    "en_uzun_sarkma_bar": 8,
                    "ort_tepki_%": 0.2,
                    "ort_tepki_ATR": 0.2,
                    "şu_an": "üstünde 1 bar",
                    "uzaklık_%": 0.1,
                    "uzaklık_ATR": 0.02,
                    "saygı_skoru": 22.0,
                },
                {
                    "symbol": "ASELS",
                    "timeframe": "1d",
                    "taraf": "Destek",
                    "MA": "HMA55",
                    "tür": "HMA",
                    "periyot": 55,
                    "fiyat": 100.0,
                    "ma_değeri": 105.0,
                    "üst/alt_bar_%": 78.0,
                    "ziyaret": 30,
                    "tepki_sayısı": 18,
                    "tepki_oranı_%": 60.0,
                    "sarkma_epizodu": 10,
                    "geri_dönen": 9,
                    "geri_dönüş_%": 90.0,
                    "ort_sarkma_bar": 1.2,
                    "en_uzun_sarkma_bar": 3,
                    "ort_tepki_%": 1.1,
                    "ort_tepki_ATR": 0.9,
                    "şu_an": "altında 2 bar",
                    "uzaklık_%": 5.0,
                    "uzaklık_ATR": 1.0,
                    "saygı_skoru": 82.0,
                },
            ]
        )

        report = format_report(
            "ASELS",
            "1d",
            prepared,
            rows,
            pd.DataFrame(),
            rows,
            top=0,
            detail_top=1,
            min_visits=5,
        )
        current_section = report.split("Güçlü ve izlenebilir ortalamalar", 1)[1]

        self.assertIn("1. HMA55", current_section)
        self.assertNotIn("1. EMA5", current_section)

    def test_report_balances_support_and_resistance_in_top_rows(self):
        prepared = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.date_range("2026-01-01", periods=1, freq="D"),
        )
        rows = pd.DataFrame(
            [
                {
                    "symbol": "ASELS",
                    "timeframe": "1d",
                    "taraf": "Diren\u00e7",
                    "MA": "HMA89",
                    "t\u00fcr": "HMA",
                    "periyot": 89,
                    "fiyat": 100.0,
                    "ma_de\u011feri": 108.0,
                    "\u00fcst/alt_bar_%": 80.0,
                    "ziyaret": 30,
                    "tepki_say\u0131s\u0131": 15,
                    "tepki_oran\u0131_%": 50.0,
                    "sarkma_epizodu": 12,
                    "geri_d\u00f6nen": 10,
                    "geri_d\u00f6n\u00fc\u015f_%": 83.0,
                    "ort_sarkma_bar": 2.0,
                    "en_uzun_sarkma_bar": 5,
                    "ort_tepki_%": 1.2,
                    "ort_tepki_ATR": 0.8,
                    "\u015fu_an": "alt\u0131nda 8 bar",
                    "uzakl\u0131k_%": 8.0,
                    "uzakl\u0131k_ATR": 1.4,
                    "sayg\u0131_skoru": 82.0,
                },
                {
                    "symbol": "ASELS",
                    "timeframe": "1d",
                    "taraf": "Diren\u00e7",
                    "MA": "HMA100",
                    "t\u00fcr": "HMA",
                    "periyot": 100,
                    "fiyat": 100.0,
                    "ma_de\u011feri": 111.0,
                    "\u00fcst/alt_bar_%": 78.0,
                    "ziyaret": 25,
                    "tepki_say\u0131s\u0131": 10,
                    "tepki_oran\u0131_%": 40.0,
                    "sarkma_epizodu": 10,
                    "geri_d\u00f6nen": 8,
                    "geri_d\u00f6n\u00fc\u015f_%": 80.0,
                    "ort_sarkma_bar": 2.4,
                    "en_uzun_sarkma_bar": 7,
                    "ort_tepki_%": 1.0,
                    "ort_tepki_ATR": 0.7,
                    "\u015fu_an": "alt\u0131nda 7 bar",
                    "uzakl\u0131k_%": 11.0,
                    "uzakl\u0131k_ATR": 1.8,
                    "sayg\u0131_skoru": 78.0,
                },
                {
                    "symbol": "ASELS",
                    "timeframe": "1d",
                    "taraf": "Destek",
                    "MA": "SMA200",
                    "t\u00fcr": "SMA",
                    "periyot": 200,
                    "fiyat": 100.0,
                    "ma_de\u011feri": 91.0,
                    "\u00fcst/alt_bar_%": 68.0,
                    "ziyaret": 9,
                    "tepki_say\u0131s\u0131": 6,
                    "tepki_oran\u0131_%": 67.0,
                    "sarkma_epizodu": 5,
                    "geri_d\u00f6nen": 4,
                    "geri_d\u00f6n\u00fc\u015f_%": 80.0,
                    "ort_sarkma_bar": 1.6,
                    "en_uzun_sarkma_bar": 4,
                    "ort_tepki_%": 1.1,
                    "ort_tepki_ATR": 0.9,
                    "\u015fu_an": "\u00fcst\u00fcnde 3 bar",
                    "uzakl\u0131k_%": -9.0,
                    "uzakl\u0131k_ATR": -1.3,
                    "sayg\u0131_skoru": 65.0,
                },
            ]
        )

        report = format_report(
            "ASELS",
            "1d",
            prepared,
            rows,
            pd.DataFrame(),
            rows,
            top=2,
            detail_top=0,
            min_visits=5,
        )
        main_section = report.split("MA DNA okumasi", 1)[0]

        self.assertIn("1. HMA89 | Diren\u00e7 | 30 temas", main_section)
        self.assertIn("2. SMA200 | Destek | 9 temas", main_section)
        self.assertNotIn("2. HMA100", main_section)

    def test_respect_image_rows_use_historical_side_column(self):
        frame = pd.DataFrame(
            [
                {
                    "symbol": "ASELS",
                    "MA": "SMA50",
                    "taraf": "Diren\u00e7",
                    "fiyat": 110.0,
                    "ma_de\u011feri": 100.0,
                    "ziyaret": 12,
                    "tepki_oran\u0131_%": 58.0,
                    "geri_d\u00f6n\u00fc\u015f_%": 75.0,
                    "uzakl\u0131k_%": -9.1,
                    "\u015fu_an": "\u00fcst\u00fcnde 3 bar",
                }
            ]
        )

        headers, rows, _ = desc_cli._respect_image_rows(frame, include_symbol=False)

        self.assertEqual(headers[2], "Taraf")
        self.assertEqual(rows[0][2], "Diren\u00e7")

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
