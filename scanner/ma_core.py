"""Backward-compatible data-layer import.

The former guarded research core was removed. New code must import from
``scanner.ma_engine`` directly.
"""

from .ma_engine import normalize_ohlcv

__all__ = ["normalize_ohlcv"]
