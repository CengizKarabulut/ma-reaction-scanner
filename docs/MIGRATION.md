# Migration from the legacy scanner

## What remains legacy

The original scanner, backtest, strategy generator, multi-indicator confirmation,
Pine scripts and Telegram notifier remain in the repository for auditability.
They are not consumed by the guarded research core and must not be described as
validated or robust.

Their automatic schedules and `repository_dispatch` triggers are disabled while
the new system accumulates prospective evidence. Manual legacy workflows remain
available for historical comparison only.

## New source of truth

- `scanner/ma_core.py`: MA mathematics, events, outcomes, controls, FDR, holdout,
  level selection and contextual confluence.
- `scanner/ma_data.py`: provider adapters, BIST-aware resampling, snapshots and
  data fingerprints.
- `scanner/ma_research_cli.py`: eight-timeframe research panel.
- `scanner/ma_validation.py`: cost-aware next-bar holdout trade validation and
  random-entry benchmark.
- `scanner/sr_baselines.py`: pivot/VWAP alternatives.
- `tests/`: deterministic unit, synthetic and integration-oriented tests.

## Local commands

Run all tests:

```bash
python -m unittest discover -s tests -v
```

Small candidate scan:

```bash
python -m scanner.ma_research_cli \
  --tickers GARAN,THYAO,AKBNK \
  --timeframes 1d \
  --periods 5,8,13,21,22,34,50,55,89,100,144,200,233,377 \
  --source auto --top 5 --null-iterations 499
```

Full research scan:

```bash
python -m scanner.ma_research_cli \
  --tickers GARAN,THYAO,AKBNK \
  --timeframes 1h,4h,1d,1wk,1mo \
  --source auto --top 5 --null-iterations 999
```

Short intraday histories will frequently be labelled insufficient. That is an
honest data limitation, not a reason to lower the evidence threshold silently.

## Output interpretation

- `all_candidates.csv`: full audit trail, including failed/inactive hypotheses.
- `panel.csv` / `panel.txt`: nearest levels with explicit evidence labels.
- `confluence.csv`: contextual clusters, never independent confirmations.
- `run_metadata.json`: data source, fingerprint, snapshot and failures.
- `data_cache/`: immutable input snapshots, intentionally not committed.

