#!/usr/bin/env python3
"""Leakage-aware event-trade validation and random-entry benchmark.

This module is intentionally separate from historical MA certification.  A
candidate is selected on discovery/validation data; trading performance is
measured only on the holdout events, with next-bar entry and explicit costs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor, sqrt
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .ma_core import AnalysisConfig, TouchEvent
except ImportError:  # direct script execution
    from ma_core import AnalysisConfig, TouchEvent


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 5.0
    spread_bps: float = 8.0
    slippage_bps: float = 5.0
    bsmv_rate: float = 0.05
    tick_size: float = 0.01
    settlement_lag_days: int = 2
    short_sale_mode: str = "direction_forecast_only"

    def __post_init__(self) -> None:
        if min(self.commission_bps, self.spread_bps, self.slippage_bps) < 0:
            raise ValueError("costs cannot be negative")
        if self.bsmv_rate < 0 or self.tick_size < 0 or self.settlement_lag_days < 0:
            raise ValueError("BIST cost metadata cannot be negative")
        if self.short_sale_mode not in {"direction_forecast_only", "executable"}:
            raise ValueError("short_sale_mode must be direction_forecast_only or executable")

    @property
    def commission_rate(self) -> float:
        return self.commission_bps / 10_000.0

    @property
    def commission_with_tax_rate(self) -> float:
        return self.commission_rate * (1.0 + self.bsmv_rate)

    @property
    def entry_friction(self) -> float:
        return (self.spread_bps / 2.0 + self.slippage_bps) / 10_000.0

    @property
    def exit_friction(self) -> float:
        return (self.spread_bps / 2.0 + self.slippage_bps) / 10_000.0


def _round_to_tick(price: float, tick_size: float, direction: int) -> float:
    if tick_size <= 0 or not np.isfinite(price):
        return float(price)
    scaled = price / tick_size
    if direction > 0:
        return float(ceil(scaled - 1e-12) * tick_size)
    if direction < 0:
        return float(floor(scaled + 1e-12) * tick_size)
    return float(round(scaled) * tick_size)


def simulate_event_trades(
    df: pd.DataFrame,
    events: Sequence[TouchEvent],
    config: AnalysisConfig,
    side: int,
    start: int,
    end: int,
    costs: CostModel | None = None,
    initial_capital: float = 100_000.0,
    risk_fraction: float = 0.01,
) -> pd.DataFrame:
    """Trade non-overlapping events using information available at bar close.

    Entry occurs at the next bar's open.  If target and stop are both touched in
    one bar, the stop is assumed first.  This deliberately avoids optimistic
    intrabar path assumptions.
    """

    if side not in (1, -1):
        raise ValueError("side must be +1 or -1")
    if not 0 < risk_fraction <= 0.10:
        raise ValueError("risk_fraction must be between 0 and 0.10")
    cost = costs or CostModel()
    equity = float(initial_capital)
    rows = []
    last_exit = start - 1
    for event in sorted(events, key=lambda item: item.position):
        if event.direction != side or event.position < start:
            continue
        entry_pos = event.position + 1
        final_pos = min(event.position + config.horizon, end - 1)
        if entry_pos > final_pos or entry_pos <= last_exit:
            continue
        raw_entry = float(df["Open"].iloc[entry_pos])
        atr_value = float(event.atr)
        if raw_entry <= 0 or not np.isfinite(atr_value) or atr_value <= 0:
            continue
        entry = raw_entry * (1.0 + side * cost.entry_friction)
        entry = _round_to_tick(entry, cost.tick_size, side)
        target = _round_to_tick(
            entry + side * config.target_atr * atr_value,
            cost.tick_size,
            -side,
        )
        stop = _round_to_tick(
            entry - side * config.stop_atr * atr_value,
            cost.tick_size,
            -side,
        )
        estimated_round_trip_cost = entry * (
            2.0 * cost.commission_with_tax_rate
            + cost.entry_friction
            + cost.exit_friction
        )
        risk_per_share = abs(entry - stop) + estimated_round_trip_cost
        risk_budget = equity * risk_fraction
        shares_by_risk = floor(risk_budget / risk_per_share)
        shares_by_capital = floor(equity / entry)
        shares = int(max(0, min(shares_by_risk, shares_by_capital)))
        if shares < 1:
            continue

        exit_reason = "TIME"
        raw_exit = float(df["Close"].iloc[final_pos])
        exit_pos = final_pos
        ambiguous = False
        for pos in range(entry_pos, final_pos + 1):
            high = float(df["High"].iloc[pos])
            low = float(df["Low"].iloc[pos])
            target_hit = high >= target if side == 1 else low <= target
            stop_hit = low <= stop if side == 1 else high >= stop
            if target_hit and stop_hit:
                raw_exit = stop
                exit_reason = "STOP_AMBIGUOUS"
                exit_pos = pos
                ambiguous = True
                break
            if stop_hit:
                raw_exit = stop
                exit_reason = "STOP"
                exit_pos = pos
                break
            if target_hit:
                raw_exit = target
                exit_reason = "TARGET"
                exit_pos = pos
                break
        exit_price = raw_exit * (1.0 - side * cost.exit_friction)
        exit_price = _round_to_tick(exit_price, cost.tick_size, -side)
        gross_pnl = side * (exit_price - entry) * shares
        commissions = (entry + exit_price) * shares * cost.commission_rate
        bsmv = commissions * cost.bsmv_rate
        net_pnl = gross_pnl - commissions - bsmv
        equity_before = equity
        equity += net_pnl
        direction_only_short = side == -1 and cost.short_sale_mode == "direction_forecast_only"
        rows.append({
            "signal_time": event.timestamp,
            "entry_time": df.index[entry_pos],
            "exit_time": df.index[exit_pos],
            "side": "LONG" if side == 1 else "SHORT",
            "execution_mode": (
                "direction_forecast_only" if direction_only_short else "trade_simulation"
            ),
            "execution_note": (
                "direction forecast - BIST short trade is not assumed executable"
                if direction_only_short
                else "cost-adjusted trade simulation"
            ),
            "settlement_lag_days": cost.settlement_lag_days,
            "entry": entry,
            "exit": exit_price,
            "target": target,
            "stop": stop,
            "shares": shares,
            "reason": exit_reason,
            "ambiguous": ambiguous,
            "bars_held": exit_pos - entry_pos + 1,
            "gross_pnl": gross_pnl,
            "commissions": commissions,
            "bsmv": bsmv,
            "costs": (
                commissions
                + bsmv
                + abs(raw_entry - entry) * shares
                + abs(raw_exit - exit_price) * shares
            ),
            "net_pnl": net_pnl,
            "return_on_equity": net_pnl / equity_before,
            "equity": equity,
        })
        last_exit = exit_pos
    return pd.DataFrame(rows)


def trade_statistics(trades: pd.DataFrame, initial_capital: float = 100_000.0) -> dict[str, float]:
    if trades is None or trades.empty:
        return {
            "trades": 0, "total_return": 0.0, "win_rate": np.nan,
            "profit_factor": np.nan, "max_drawdown": 0.0, "trade_sharpe": np.nan,
            "direction_forecast_only_trades": 0,
        }
    pnl = trades["net_pnl"].to_numpy(dtype=float)
    returns = trades["return_on_equity"].to_numpy(dtype=float)
    equity = np.concatenate(([initial_capital], trades["equity"].to_numpy(dtype=float)))
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    sharpe = (
        float(np.mean(returns) / np.std(returns, ddof=1) * sqrt(len(returns)))
        if len(returns) > 1 and np.std(returns, ddof=1) > 0 else np.nan
    )
    direction_only = (
        int((trades["execution_mode"] == "direction_forecast_only").sum())
        if "execution_mode" in trades
        else 0
    )
    return {
        "trades": int(len(trades)),
        "total_return": float(equity[-1] / initial_capital - 1.0),
        "net_pnl": float(pnl.sum()),
        "win_rate": float(np.mean(pnl > 0)),
        "profit_factor": gains / losses if losses > 0 else np.inf,
        "max_drawdown": float(drawdowns.min()),
        "trade_sharpe": sharpe,
        "average_cost": float(trades["costs"].mean()),
        "direction_forecast_only_trades": direction_only,
    }


def random_entry_benchmark(
    df: pd.DataFrame,
    observed_trades: pd.DataFrame,
    config: AnalysisConfig,
    side: int,
    start: int,
    end: int,
    costs: CostModel | None = None,
    iterations: int = 499,
    seed: int = 1729,
    initial_capital: float = 100_000.0,
    risk_fraction: float = 0.01,
) -> dict[str, object]:
    """Compare holdout return with equal-count, non-overlapping random entries."""

    if observed_trades is None or observed_trades.empty:
        return {"p_value": np.nan, "observed_return": 0.0, "null_returns": []}
    count = len(observed_trades)
    candidates = np.arange(max(start + 1, config.atr_period), end - config.horizon - 1)
    if len(candidates) < count:
        return {"p_value": np.nan, "observed_return": 0.0, "null_returns": []}
    rng = np.random.default_rng(seed)
    null_returns = []
    for _ in range(iterations):
        shuffled = rng.permutation(candidates)
        chosen: list[int] = []
        for position in shuffled:
            if all(abs(int(position) - previous) > config.horizon for previous in chosen):
                chosen.append(int(position))
                if len(chosen) == count:
                    break
        if len(chosen) < count:
            continue
        events = [
            TouchEvent(
                position=position,
                timestamp=df.index[position],
                direction=side,
                regime="random",
                atr=float(df["ATR"].iloc[position]),
                entry=float(df["Close"].iloc[position]),
                ma_value=float(df["Close"].iloc[position]),
                volatility_bin=int(df["VOL_BIN"].iloc[position]) if "VOL_BIN" in df else 0,
                session_bin=int(df["SESSION_BIN"].iloc[position]) if "SESSION_BIN" in df else 0,
            )
            for position in sorted(chosen)
            if np.isfinite(df["ATR"].iloc[position])
        ]
        random_trades = simulate_event_trades(
            df, events, config, side, start, end, costs, initial_capital, risk_fraction
        )
        null_returns.append(trade_statistics(random_trades, initial_capital)["total_return"])
    observed_return = trade_statistics(observed_trades, initial_capital)["total_return"]
    valid = np.asarray(null_returns, dtype=float)
    p_value = float((1 + np.sum(valid >= observed_return)) / (len(valid) + 1)) if len(valid) else np.nan
    return {
        "p_value": p_value,
        "observed_return": observed_return,
        "null_median_return": float(np.median(valid)) if len(valid) else np.nan,
        "null_p90_return": float(np.quantile(valid, 0.90)) if len(valid) else np.nan,
        "iterations": int(len(valid)),
        "cost_model": asdict(costs or CostModel()),
        "null_returns": null_returns,
    }

