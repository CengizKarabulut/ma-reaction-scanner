# Guarded MA Reaction Research Methodology

## Scope

The research core asks one narrow question:

> After price approaches a moving average from the correct side and touches it,
> is the subsequent directional reaction stronger and more repeatable than
> matched controls, including in unseen time periods?

It does **not** claim that a certified level will hold at the next touch.

## Data contract

- OHLC must be positive and internally ordered.
- Floating-point differences below `1e-12` relative tolerance are clamped;
  larger ordering errors fail the scan.
- yfinance is requested with adjusted prices. The actual adjustment policy and
  corporate-action history must remain in run metadata.
- Every fetched dataset receives a SHA-256 fingerprint and an immutable gzip CSV
  snapshot, so a result can be reproduced.
- Derived 4-hour BIST bars are anchored at 10:00 Europe/Istanbul rather than
  midnight. Weekly bars end Friday; monthly bars use calendar month-end.

## Candidate, evidence and certification

These words are intentionally different:

- **Candidate:** an MA currently below price (support candidate) or above price
  (resistance candidate). Location only.
- **Discovery pass:** historical reactions beat the matched null after FDR.
- **Certified:** discovery pass also remains positive in validation and untouched
  holdout segments, including the configured holdout Wilson lower-bound gate.
- **Low confidence:** a raw pass produced by a fast/underpowered profile, or by a
  holdout segment with only 3-4 events. These rows are downgraded from strict
  certification and labelled `LOW_CONFIDENCE`, `low_confidence_fast`, or
  `certified_thin_holdout`.
- **Actionable:** strictly certified and not farther than the configured ATR distance.

The panel may show several `CANDIDATE_ONLY` or `LOW_CONFIDENCE` levels when
nothing is strictly certified.
It may also show no support or no resistance when no scanned MA exists on that
side. The program never relabels a weak candidate merely to fill a table.

## Independent touch definition

1. Price must first be at least `separation_atr` from the MA on the same side.
2. The candle range then intersects an ATR-sized zone around the MA.
3. Consecutive candles inside the zone count as one event.
4. No new event is opened until the forward horizon of the prior event ends.
5. Approaching from above tests support; approaching from below tests resistance.

This prevents one prolonged interaction from masquerading as many observations.

## Outcomes

Every event is measured in ATR units known at the touch:

- target reached before stop;
- fixed-horizon directional return;
- maximum favourable and adverse excursion (descriptive only);
- bars to target;
- retest and ambiguous-bar rates.

MFE is never used as if a trader exited at the future peak. If target and stop
are both touched in one OHLC bar, the event is conservatively treated as a stop.

## Controls

The discovery statistic must survive:

1. matched random entries with the same direction, ADX regime, volatility bin
   and intraday session bin;
2. lagged/shifted versions of the MA;
3. causal piecewise-horizontal levels built from prior blocks.

The matched random ensemble supplies the empirical p-value. The smaller shift
and horizontal ensembles are secondary gates, not coarse p-values.

Volatility regimes are assigned with fixed ATR/price thresholds from the current
and prior bars rather than full-sample percentile ranks. This avoids a look-ahead
where a future volatility shock could change an older event's matching bin.

## Multiple testing

The live hypothesis family contains the currently relevant side of every MA in
one ticker/timeframe scan. Benjamini-Hochberg is the default because these smooth
MA tests are positively dependent. `fdr_method="by"` is available for a more
conservative arbitrary-dependence audit, but it requires substantially more null
draws. No individual raw p-value is presented as proof after scanning many MAs.

Fast operational scans are intentionally underpowered candidate screens. When
`null_iterations` is below 99, or when shift/horizontal controls are disabled, a
raw pass is downgraded to `LOW_CONFIDENCE` instead of `CERTIFIED` because the
minimum attainable empirical p-value and missing secondary controls are not enough
for strict certification.

## Time separation

Bars are split chronologically:

- 60% discovery;
- 20% validation;
- 20% untouched holdout.

Parameters and candidates are not tuned on holdout results. Holdout must satisfy
minimum event count, positive median fixed-horizon ATR reaction, positive score,
and the same Wilson lower-bound threshold used by the configured discovery gate.
A raw pass with only 3-4 holdout events is kept visible as `certified_thin_holdout`
but is not counted as strict certification. A candidate with too few events in any
required segment is labelled insufficient rather than rescued with a looser fallback.

## Ranking score constants

`rank_score` is a presentation ranking, not a trading expectancy. Strict
certification receives the largest bonus, thin-holdout and fast low-confidence
passes receive smaller bonuses, discovery pass receives a context bonus, and the
remaining quality term rewards positive discovery/validation/holdout scores while
penalizing distance from current price. The constants are deliberately ordinal:
they keep strict certified rows ahead of downgraded evidence and keep mere
candidates behind both. Changing them should be treated as UI ranking sensitivity,
not a change to the statistical gates.

## Cross-timeframe confluence

Nearby levels from multiple timeframes are clustered for context. They are built
from the same underlying price process and often from resampled versions of the
same bars, so they are **not independent votes**. Confluence cannot turn an
uncertified level into statistical evidence.

## Trade validation

`ma_validation.py` enters at the next bar open, handles long and short targets,
uses one exit per event, assumes stop-first on ambiguous bars, applies commission,
spread and slippage, sizes by risk, and compares holdout return with equal-count
random entries. This replaces the legacy repeated-TP1 and missing-short logic.

## Alternative level families

`sr_baselines.py` creates causal rolling pivot and rolling-VWAP levels. They use
only prior bars and pass through the same event/null/holdout schema. If these
alternatives beat MAs, the correct conclusion is that MA is not the best level
proxy for that instrument and timeframe.

## Remaining external validity requirements

Code tests cannot establish future profitability. Before scheduled signals are
enabled, the project still needs:

- provider-specific adjusted/raw verification for BIST corporate actions;
- historical-universe data for true point-in-time membership. Until that exists,
  run summaries and metadata print an explicit survivorship-bias warning because
  BIST universes use currently available membership lists;
- realistic broker-specific costs and short-sale constraints;
- a frozen prospective paper-test protocol with enough completed events;
- periodic monitoring for regime decay.

