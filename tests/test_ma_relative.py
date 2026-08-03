import unittest

import numpy as np
import pandas as pd

from scanner.ma_relative import BenchmarkCache, to_relative


def series_frame(closes, index=None, spread=0.01):
    closes = np.asarray(closes, dtype=float)
    index = index if index is not None else pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes * (1.0 + spread),
            "Low": closes * (1.0 - spread),
            "Close": closes,
            "Volume": np.full(len(closes), 1_000.0),
        },
        index=index,
    )


class RelativeSeriesTests(unittest.TestCase):
    def test_rebasing_leaves_the_final_close_untouched(self):
        stock = series_frame([100.0, 110.0, 121.0, 133.1])
        benchmark = series_frame([1_000.0, 1_100.0, 1_150.0, 1_200.0])

        relative = to_relative(stock, benchmark)

        self.assertAlmostEqual(
            float(relative["Close"].iloc[-1]), float(stock["Close"].iloc[-1])
        )

    def test_common_drift_is_removed(self):
        growth = 1.01 ** np.arange(60)
        stock = series_frame(100.0 * growth)
        benchmark = series_frame(1_000.0 * growth)

        relative = to_relative(stock, benchmark)

        # The instrument moved exactly with the market, so nothing is left.
        self.assertTrue(np.allclose(relative["Close"], relative["Close"].iloc[-1]))

    def test_instrument_specific_move_survives(self):
        benchmark = series_frame(1_000.0 * 1.01 ** np.arange(30))
        outperformer = series_frame(100.0 * 1.02 ** np.arange(30))

        relative = to_relative(outperformer, benchmark)

        self.assertGreater(
            float(relative["Close"].iloc[-1]), float(relative["Close"].iloc[0])
        )

    def test_ohlc_ordering_is_preserved(self):
        stock = series_frame([100.0, 96.0, 104.0, 99.0], spread=0.03)
        benchmark = series_frame([1_000.0, 980.0, 1_040.0, 1_010.0])

        relative = to_relative(stock, benchmark)

        self.assertTrue((relative["High"] >= relative[["Open", "Close", "Low"]].max(axis=1)).all())
        self.assertTrue((relative["Low"] <= relative[["Open", "Close", "High"]].min(axis=1)).all())
        self.assertTrue((relative[["Open", "High", "Low", "Close"]] > 0).all().all())

    def test_volume_is_not_rescaled(self):
        stock = series_frame([100.0, 110.0, 120.0])
        benchmark = series_frame([1_000.0, 1_100.0, 1_200.0])

        relative = to_relative(stock, benchmark)

        self.assertTrue(np.allclose(relative["Volume"], stock["Volume"]))

    def test_missing_benchmark_bars_are_carried_forward_never_backfilled(self):
        index = pd.date_range("2024-01-01", periods=4, freq="D")
        stock = series_frame([100.0, 101.0, 102.0, 103.0], index=index)
        benchmark = series_frame([1_000.0, 1_020.0], index=index[[0, 2]])

        relative = to_relative(stock, benchmark, rebase=False)

        # Bar 1 has no benchmark print, so it must reuse bar 0's value, not
        # bar 2's - using a later value would be look-ahead.
        self.assertAlmostEqual(float(relative["Close"].iloc[1]), 101.0 / 1_000.0)
        self.assertAlmostEqual(float(relative["Close"].iloc[2]), 102.0 / 1_020.0)

    def test_leading_bars_without_a_benchmark_are_dropped(self):
        index = pd.date_range("2024-01-01", periods=4, freq="D")
        stock = series_frame([100.0, 101.0, 102.0, 103.0], index=index)
        benchmark = series_frame([1_000.0, 1_010.0], index=index[[2, 3]])

        relative = to_relative(stock, benchmark)

        self.assertEqual(len(relative), 2)
        self.assertEqual(list(relative.index), list(index[[2, 3]]))

    def test_disjoint_series_raise_instead_of_producing_nonsense(self):
        stock = series_frame([100.0, 101.0], index=pd.date_range("2024-01-01", periods=2, freq="D"))
        benchmark = series_frame([1_000.0], index=pd.date_range("2025-01-01", periods=1, freq="D"))

        with self.assertRaises(ValueError):
            to_relative(stock, benchmark)


class _CountingProvider:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def fetch(self, ticker, timeframe, prefer_cache=False, *, asset_class=None, market=None):
        self.calls.append((ticker, timeframe, asset_class, market))

        class _Result:
            pass

        result = _Result()
        result.frame = self.frame
        return result


class _FailingProvider:
    def __init__(self):
        self.calls = 0

    def fetch(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("saglayici hatasi")


class BenchmarkCacheTests(unittest.TestCase):
    def test_benchmark_is_fetched_once_per_timeframe(self):
        provider = _CountingProvider(series_frame([1_000.0, 1_010.0, 1_020.0]))
        cache = BenchmarkCache(provider, "XU100")

        cache.get("1d")
        cache.get("1d")
        cache.get("1wk")

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[0], ("XU100", "1d", "index", "BIST"))

    def test_a_failed_fetch_is_not_retried_for_every_symbol(self):
        provider = _FailingProvider()
        cache = BenchmarkCache(provider, "XU100")

        for _ in range(5):
            with self.assertRaises(RuntimeError):
                cache.get("1d")

        self.assertEqual(provider.calls, 1)


if __name__ == "__main__":
    unittest.main()
