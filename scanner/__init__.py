"""BIST moving-average reaction research tools.

The scanner now has one supported timeframe surface: 1h, 4h, 1d and 1wk.
``ma_engine`` historically exposed a broader constant; patching that legacy
export here keeps existing imports compatible while preventing retired periods
from being accepted by ``ma_scan``.
"""

CORE_TIMEFRAMES = ("1h", "4h", "1d", "1wk")

# Compatibility bridge for modules that still import TIMEFRAMES from ma_engine.
from . import ma_engine as _ma_engine

_ma_engine.TIMEFRAMES = CORE_TIMEFRAMES
