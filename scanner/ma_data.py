#!/usr/bin/env python3
"""Market-data adapters and reproducible snapshots for the MA research core."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

try:
    from .ma_core import normalize_ohlcv
except ImportError:  # direct script execution
    from ma_core import normalize_ohlcv

try:
    import borsapy as bp
except ImportError:  # pragma: no cover - environment dependent
    bp = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - environment dependent
    yf = None


TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo")
DIRECT_INTERVALS = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "1d": "1d"}
BASE_INTERVAL = {"4h": "1h", "1wk": "1d", "1mo": "1d"}
YF_PERIODS = {"5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d", "1d": "10y"}
BP_PERIODS = {"5m": "5g", "15m": "5g", "30m": "5g", "1h": "1ay"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _symbol_type(symbol: str, asset_class: str | None = None) -> str:
    if asset_class in {"index", "sector_index"}:
        return "bist_index"
    if asset_class in {"crypto", "commodity"}:
        return "external"
    value = symbol.upper().replace(".IS", "")
    if re.match(r"^X[A-Z0-9]{3,4}$", value):
        return "bist_index"
    return "bist"


def _yfinance_symbol(
    ticker: str, asset_class: str | None = None, market: str | None = None
) -> str:
    symbol = ticker.upper()
    if (
        market == "BIST"
        and _symbol_type(symbol, asset_class) in {"bist", "bist_index"}
        and not symbol.endswith(".IS")
    ):
        symbol += ".IS"
    elif (
        asset_class is None
        and _symbol_type(symbol) == "bist"
        and not symbol.endswith(".IS")
    ):
        symbol += ".IS"
    return symbol


def _start_date(years: int = 10) -> str:
    return (
        (datetime.now(timezone.utc) - timedelta(days=365 * years + 90))
        .date()
        .isoformat()
    )


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample with BIST-aware 4h bins and completed week/month labels."""

    df = normalize_ohlcv(frame)
    aggregations = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    if timeframe == "4h":
        # BIST continuous session starts at 10:00 Europe/Istanbul.  Anchoring to
        # 10:00 prevents arbitrary midnight-aligned 4h candles.
        sampled = df.resample(
            "4h", origin="start_day", offset="10h", label="left", closed="left"
        ).agg(aggregations)
    elif timeframe == "1wk":
        sampled = df.resample("W-FRI", label="right", closed="right").agg(aggregations)
    elif timeframe == "1mo":
        sampled = df.resample("ME", label="right", closed="right").agg(aggregations)
    else:
        raise ValueError(f"unsupported derived timeframe: {timeframe}")
    return normalize_ohlcv(sampled.dropna(subset=["Open", "High", "Low", "Close"]))


def _fetch_yfinance(
    ticker: str,
    interval: str,
    asset_class: str | None = None,
    market: str | None = None,
) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    symbol = _yfinance_symbol(ticker, asset_class, market)
    period = YF_PERIODS[interval]
    data = yf.download(
        symbol,
        period=period,
        interval="60m" if interval == "1h" else interval,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} {interval}")
    return normalize_ohlcv(data)


def _bp_history(
    ticker: str,
    interval: str,
    asset_class: str | None = None,
    market: str | None = None,
) -> pd.DataFrame:
    if bp is None:
        raise RuntimeError("borsapy is not installed")
    if market is not None and market != "BIST":
        raise RuntimeError("borsapy adapter only supports BIST instruments")
    if asset_class in {"crypto", "commodity"}:
        raise RuntimeError(f"borsapy does not support asset class: {asset_class}")
    symbol = ticker.upper().replace(".IS", "")
    is_index = _symbol_type(symbol, asset_class) == "bist_index"
    instrument = (
        bp.Index(symbol) if is_index and hasattr(bp, "Index") else bp.Ticker(symbol)
    )
    if interval == "1d":
        try:
            data = instrument.history(start=_start_date(10), interval="1d")
        except TypeError:
            data = instrument.history(period="5y", interval="1d")
    else:
        data = instrument.history(period=BP_PERIODS[interval], interval=interval)
    if data is None or data.empty:
        raise RuntimeError(f"borsapy returned no data for {symbol} {interval}")
    return normalize_ohlcv(data)


@dataclass(frozen=True)
class FetchResult:
    frame: pd.DataFrame
    ticker: str
    timeframe: str
    source: str
    base_interval: str
    snapshot_path: str | None
    fingerprint: str


class MarketDataProvider:
    """Fetch data with explicit provenance and optional immutable snapshots."""

    def __init__(
        self,
        source: str = "auto",
        cache_dir: str | Path = "data_cache",
        use_cache: bool = True,
        snapshot: bool = True,
    ) -> None:
        if source not in {"auto", "borsapy", "yfinance"}:
            raise ValueError("source must be auto, borsapy, or yfinance")
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.snapshot = snapshot

    def _latest_cache(self, ticker: str, timeframe: str) -> Path | None:
        folder = self.cache_dir / _safe_name(ticker.upper()) / timeframe
        if not self.use_cache or not folder.exists():
            return None
        files = sorted(
            folder.glob("*.csv.gz"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return files[0] if files else None

    def _read_cache(self, path: Path, ticker: str, timeframe: str) -> FetchResult:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        df = normalize_ohlcv(frame)
        metadata_path = path.with_suffix("").with_suffix(".json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        return FetchResult(
            frame=df,
            ticker=ticker.upper(),
            timeframe=timeframe,
            source=str(metadata.get("source", "cache")),
            base_interval=str(metadata.get("base_interval", timeframe)),
            snapshot_path=str(path),
            fingerprint=fingerprint_frame(df),
        )

    def fetch(
        self,
        ticker: str,
        timeframe: str,
        prefer_cache: bool = False,
        *,
        asset_class: str | None = None,
        market: str | None = None,
    ) -> FetchResult:
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        cached = self._latest_cache(ticker, timeframe)
        if prefer_cache and cached is not None:
            return self._read_cache(cached, ticker, timeframe)

        base = BASE_INTERVAL.get(timeframe, DIRECT_INTERVALS[timeframe])
        if self.source != "auto":
            sources = [self.source]
        elif market not in {None, "BIST"} or asset_class in {"crypto", "commodity"}:
            sources = ["yfinance"]
        else:
            sources = (
                ["yfinance", "borsapy"] if base != "1d" else ["borsapy", "yfinance"]
            )
        errors = []
        frame = None
        used_source = None
        for source in sources:
            try:
                if source == "borsapy":
                    frame = _bp_history(
                        ticker, base, asset_class=asset_class, market=market
                    )
                else:
                    frame = _fetch_yfinance(
                        ticker,
                        base,
                        asset_class=asset_class,
                        market=market,
                    )
                used_source = source
                break
            except Exception as exc:  # source fallbacks must preserve diagnostics
                errors.append(f"{source}: {exc}")
        if frame is None:
            if cached is not None:
                return self._read_cache(cached, ticker, timeframe)
            raise RuntimeError("; ".join(errors))
        if timeframe in BASE_INTERVAL:
            frame = resample_ohlcv(frame, timeframe)
        else:
            frame = normalize_ohlcv(frame)

        snapshot_path = None
        if self.snapshot:
            snapshot_path = str(
                self._write_snapshot(
                    frame,
                    ticker,
                    timeframe,
                    used_source or "unknown",
                    base,
                    asset_class,
                    market,
                )
            )
        return FetchResult(
            frame=frame,
            ticker=ticker.upper(),
            timeframe=timeframe,
            source=used_source or "unknown",
            base_interval=base,
            snapshot_path=snapshot_path,
            fingerprint=fingerprint_frame(frame),
        )

    def _write_snapshot(
        self,
        frame: pd.DataFrame,
        ticker: str,
        timeframe: str,
        source: str,
        base_interval: str,
        asset_class: str | None,
        market: str | None,
    ) -> Path:
        folder = self.cache_dir / _safe_name(ticker.upper()) / timeframe
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fingerprint = fingerprint_frame(frame)
        path = folder / f"{stamp}_{fingerprint[:12]}.csv.gz"
        if not path.exists():
            frame.to_csv(path, compression="gzip", float_format="%.10f")
            metadata = {
                "ticker": ticker.upper(),
                "timeframe": timeframe,
                "source": source,
                "base_interval": base_interval,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows": len(frame),
                "first_bar": str(frame.index[0]),
                "asset_class": asset_class,
                "market": market,
                "last_bar": str(frame.index[-1]),
                "fingerprint": fingerprint,
            }
            path.with_suffix("").with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return path

    def probe(
        self,
        ticker: str,
        *,
        asset_class: str | None = None,
        market: str | None = None,
    ) -> list[dict[str, object]]:
        rows = []
        for source, fetcher in (
            ("borsapy", _bp_history),
            ("yfinance", _fetch_yfinance),
        ):
            for interval in DIRECT_INTERVALS.values():
                try:
                    frame = fetcher(
                        ticker, interval, asset_class=asset_class, market=market
                    )
                    rows.append(
                        {
                            "source": source,
                            "interval": interval,
                            "ok": True,
                            "rows": len(frame),
                            "first": str(frame.index[0]),
                            "last": str(frame.index[-1]),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "source": source,
                            "interval": interval,
                            "ok": False,
                            "rows": 0,
                            "first": "",
                            "last": "",
                            "error": str(exc),
                        }
                    )
        return rows


def fingerprint_frame(frame: pd.DataFrame) -> str:
    # Vendor floats and CSV round-trips can differ by one ULP. A canonical
    # decimal representation keeps identical market data reproducibly identical
    # without masking economically meaningful price changes.
    canonical = normalize_ohlcv(frame)
    lines: list[str] = []
    for timestamp, row in canonical.iterrows():
        stamp = pd.Timestamp(timestamp).isoformat()
        values = [
            *(
                f"{float(row[column]):.8f}"
                for column in ("Open", "High", "Low", "Close")
            ),
            f"{float(row['Volume']):.3f}",
        ]
        lines.append(",".join((stamp, *values)))
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
