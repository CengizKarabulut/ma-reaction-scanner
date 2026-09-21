"""BIST moving-average reaction research tools.

The public analysis policy has one canonical timeframe set: 1h, 4h, 1d and
1wk.  The legacy engine module still contains historical implementation detail,
but package imports pin its exported ``TIMEFRAMES`` value to this policy so
CLI parsing, summaries and provider validation cannot drift apart.
"""

CORE_TIMEFRAMES = ("1h", "4h", "1d", "1wk")


def _apply_core_timeframe_policy() -> None:
    from . import ma_engine as _ma_engine

    _ma_engine.TIMEFRAMES = CORE_TIMEFRAMES


_apply_core_timeframe_policy()
