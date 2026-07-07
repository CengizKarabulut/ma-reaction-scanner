# Prospective paper-test protocol

Historical backtests are not the final judge. The project records certified
dynamic MA watches and waits for future bars that did not exist when the watch
was created.

## Pre-registration

Before enabling a cohort, freeze:

- ticker universe and liquidity rule;
- timeframes and MA universe;
- touch zone, separation, target, stop and horizon by timeframe;
- FDR level and null iterations;
- transaction costs and the minimum number of resolved events;
- the success criterion used to decide whether the system proceeds.

Do not change those values midway through a cohort. Parameter changes start a
new cohort and receive a new identifier.

## Create watches

After a full research scan:

```bash
python -m scanner.paper_tracker create \
  --panel reports/guarded/panel_detail.csv \
  --metadata reports/guarded/run_metadata.json \
  --cohort-id daily-pilot-v1 \
  --ledger paper/ma_watchlist.csv
```

Only certified rows are recorded by default. `--include-candidates` is available
for research comparisons; the ledger permanently labels those rows
`CANDIDATE_ONLY`. Each watch also stores the scan fingerprint and the complete
frozen analysis configuration, so later code changes cannot silently redefine
an in-progress cohort.

## Advance watches

On later dates:

```bash
python -m scanner.paper_tracker update \
  --ledger paper/ma_watchlist.csv \
  --source auto
```

The tracker recomputes the dynamic MA, ignores touches at or before the recorded
cut-off, triggers on the first eligible future touch, and resolves only after the
configured forward horizon is observable.

## Promotion rule

Automatic notifications and scheduled trading-style outputs remain disabled
until a pre-registered cohort has enough resolved events and meets its frozen
target after costs. A high historical score is not a substitute for this step.

