#!/usr/bin/env python3
"""Prospective paper-test ledger for certified dynamic MA levels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from .ma_core import (
        TIMEFRAME_CONFIGS, compute_ma, detect_independent_touches, measure_event, prepare_frame,
    )
    from .ma_data import MarketDataProvider
except ImportError:
    from ma_core import TIMEFRAME_CONFIGS, compute_ma, detect_independent_touches, measure_event, prepare_frame
    from ma_data import MarketDataProvider


LEDGER_COLUMNS = [
    "signal_id", "created_at", "watch_after", "ticker", "timeframe", "side",
    "ma_type", "period", "scan_level", "scan_price", "q_value", "state",
    "trigger_time", "trigger_entry", "trigger_atr", "outcome", "resolved_at",
    "fixed_return_atr", "favorable_atr", "adverse_atr", "bars_to_target",
]


def _identifier(row: pd.Series, created_at: str) -> str:
    payload = "|".join([
        str(row["ticker"]), str(row["timeframe"]), str(row["side"]),
        str(row["ma_type"]), str(int(row["period"])), created_at,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_watchlist(
    panel: pd.DataFrame,
    created_at: str | None = None,
    watch_after: dict[tuple[str, str], str] | None = None,
    include_candidates: bool = False,
) -> pd.DataFrame:
    """Turn panel rows into dynamic-level watches; certified-only by default."""

    if panel is None or panel.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    created = created_at or datetime.now(timezone.utc).isoformat()
    selected = panel.copy() if include_candidates else panel[panel["certified"].astype(bool)].copy()
    rows = []
    for _, row in selected.iterrows():
        after = (watch_after or {}).get(
            (str(row["ticker"]).upper(), str(row["timeframe"])), created
        )
        rows.append({
            "signal_id": _identifier(row, created),
            "created_at": created,
            "watch_after": after,
            "ticker": str(row["ticker"]).upper(),
            "timeframe": str(row["timeframe"]),
            "side": str(row["side"]),
            "ma_type": str(row["ma_type"]),
            "period": int(row["period"]),
            "scan_level": float(row["current_ma"]),
            "scan_price": float(row["current_price"]),
            "q_value": float(row["q_value"]) if np.isfinite(row["q_value"]) else np.nan,
            "state": "WATCHING",
            "trigger_time": "",
            "trigger_entry": np.nan,
            "trigger_atr": np.nan,
            "outcome": "",
            "resolved_at": "",
            "fixed_return_atr": np.nan,
            "favorable_atr": np.nan,
            "adverse_atr": np.nan,
            "bars_to_target": np.nan,
        })
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def append_watchlist(path: str | Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    ledger_path = Path(path)
    existing = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame(columns=LEDGER_COLUMNS)
    combined = pd.concat([existing, new_rows], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates("signal_id", keep="last")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ledger_path, index=False)
    return combined


def _timestamp_after(value: object, boundary: object) -> bool:
    left = pd.Timestamp(value)
    right = pd.Timestamp(boundary)
    if left.tzinfo is not None and right.tzinfo is None:
        right = right.tz_localize(left.tzinfo)
    elif left.tzinfo is None and right.tzinfo is not None:
        right = right.tz_localize(None)
    return bool(left > right)


def advance_watchlist(
    ledger: pd.DataFrame,
    fetch_fn: Callable[[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Trigger and resolve watches only with bars newer than their creation cut."""

    updated = ledger.copy()
    for index, watch in updated.iterrows():
        if watch.get("state") == "RESOLVED":
            continue
        timeframe = str(watch["timeframe"])
        config = TIMEFRAME_CONFIGS[timeframe]
        frame = prepare_frame(fetch_fn(str(watch["ticker"]), timeframe), config)
        ma = compute_ma(str(watch["ma_type"]), frame["Close"], frame["Volume"], int(watch["period"]))
        side = 1 if str(watch["side"]) == "support" else -1
        events = detect_independent_touches(frame, ma, config)

        trigger_event = None
        if watch.get("state") == "WATCHING":
            boundary = watch.get("watch_after") or watch.get("created_at")
            eligible = [
                event for event in events
                if event.direction == side and _timestamp_after(event.timestamp, boundary)
            ]
            if eligible:
                trigger_event = eligible[0]
                updated.at[index, "state"] = "TRIGGERED"
                updated.at[index, "trigger_time"] = str(trigger_event.timestamp)
                updated.at[index, "trigger_entry"] = trigger_event.entry
                updated.at[index, "trigger_atr"] = trigger_event.atr
        else:
            trigger_time = pd.Timestamp(watch["trigger_time"])
            matching = [event for event in events if pd.Timestamp(event.timestamp) == trigger_time]
            trigger_event = matching[0] if matching else None

        if trigger_event is None:
            continue
        if trigger_event.position + config.horizon >= len(frame):
            continue
        measurement = measure_event(frame, trigger_event.position, side, config)
        if measurement is None:
            continue
        outcome = "TARGET" if measurement.first_hit == 1 else (
            "STOP" if measurement.first_hit == -1 else "TIMEOUT"
        )
        updated.at[index, "state"] = "RESOLVED"
        updated.at[index, "outcome"] = outcome
        updated.at[index, "resolved_at"] = str(frame.index[trigger_event.position + config.horizon])
        updated.at[index, "fixed_return_atr"] = measurement.fixed_return_atr
        updated.at[index, "favorable_atr"] = measurement.favorable_atr
        updated.at[index, "adverse_atr"] = measurement.adverse_atr
        updated.at[index, "bars_to_target"] = measurement.bars_to_target
    return updated


def ledger_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    resolved = ledger[ledger["state"] == "RESOLVED"].copy()
    if resolved.empty:
        return pd.DataFrame([{
            "watches": len(ledger), "resolved": 0, "target_rate": np.nan,
            "median_fixed_atr": np.nan,
        }])
    return pd.DataFrame([{
        "watches": len(ledger),
        "resolved": len(resolved),
        "target_rate": float(np.mean(resolved["outcome"] == "TARGET")),
        "median_fixed_atr": float(resolved["fixed_return_atr"].median()),
    }])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--panel", required=True)
    create.add_argument("--ledger", default="paper/ma_watchlist.csv")
    create.add_argument("--include-candidates", action="store_true")
    update = sub.add_parser("update")
    update.add_argument("--ledger", default="paper/ma_watchlist.csv")
    update.add_argument("--source", choices=["auto", "borsapy", "yfinance"], default="auto")
    args = parser.parse_args(argv)

    if args.command == "create":
        panel = pd.read_csv(args.panel)
        rows = build_watchlist(panel, include_candidates=args.include_candidates)
        combined = append_watchlist(args.ledger, rows)
        print(ledger_summary(combined).to_string(index=False))
        return 0

    ledger_path = Path(args.ledger)
    if not ledger_path.exists():
        raise SystemExit(f"Ledger not found: {ledger_path}")
    provider = MarketDataProvider(source=args.source)
    ledger = pd.read_csv(ledger_path)
    advanced = advance_watchlist(ledger, lambda ticker, tf: provider.fetch(ticker, tf).frame)
    advanced.to_csv(ledger_path, index=False)
    print(ledger_summary(advanced).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

