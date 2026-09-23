# Quality Momentum

A systematic equity strategy, built and run independently outside work hours on free data.

The strategy does **not** currently beat the S&P 500. This repository is here because the
interesting part is not the return — it is what had to be fixed before the backtest was
worth believing at all.

---

## What "honest" means here

The first version of this backtest looked good. It was wrong in five ways, and each one
flattered the result:

| Problem | What the first version did | What this version does |
|---|---|---|
| Survivorship bias | Used today's index members for all of history | Point-in-time membership from a dated add/remove history (1996 → today); a stock is a candidate only on dates it was actually in the index |
| Look-ahead fundamentals | Screened on ROE / EPS / market cap that are only known *now* | Backtest screen is price-based only. Fundamentals are applied to the **live** signal, where they are known at trade time. A second backtest with today's fundamentals frozen is run alongside — **the gap between the two curves is the size of the look-ahead flattery** |
| Free rebalancing | Re-weighted to 1/N every day at zero cost | Shares are held between rebalances, weights drift, costs charged on real turnover (10 bps) |
| Delistings | Silently dropped | Sold at last printed price at the next rebalance; per-year data coverage reported so residual survivorship is visible |
| Convenient window | 2023–25 only — an AI bull run | From 2007, so 2008–09, 2011, 2015–16, 2018, 2020 and 2022 are all in the sample |

Sharpe uses the 13-week T-bill as the risk-free rate. Share classes are de-duplicated
(GOOG/GOOGL count once).

## The strategy

```
universe    S&P 500, point-in-time
screen      price > $10, avg volume > 2M sh/day, rolling beta 0.8-2.0
            (+ ROE > 10%, EPS > 0, market cap > $10B on the live signal only)
signal      12-1 month total-return momentum
hold        top 20, equal weight, monthly rebalance at the month-end close
hysteresis  keep a holding while it is still in the top 30
cost        10 bps on traded notional
```

## Results

`notebooks/results.ipynb` plots what `run.py` produced — it reads `results/` and needs no network
and no price cache, so it runs from a clean clone.

| run | CAGR | excess vs SPY | Sharpe | max drawdown |
|---|---|---|---|---|
| honest | 11.5% | **+0.64 pp** | 0.49 | −66.7% |
| look-ahead fundamentals (wrong) | 14.0% | **+3.13 pp** | 0.58 | −62.8% |
| S&P 500 (SPY) | 10.9% | — | 0.55 | −55.2% |

Screening on fundamentals that were only knowable later is worth **2.5 points of CAGR a year** —
about four fifths of the apparent edge. The honest run has a worse Sharpe than simply holding the
index, and a deeper drawdown.

The window matters as much as the strategy. Across 2007–2012 the same rules lose to the index by
6.5 points a year; over the 2023-onward window the first backtest used, they win by 19.

## Current status

- Backtested from 2007; forward-tested with real money April–August 2026.
- Does not beat the S&P 500 as it stands: larger drawdowns, occasionally larger upside.
- Residual survivorship bias remains from tickers the free data source no longer serves.
  The coverage report quantifies how much.

## Things tried and dropped

- **Mean reversion on beaten-down sector ETFs** — fit the window it was tuned on, did not
  hold outside it.
- **Ichimoku with the classic fixed periods replaced by fitted ones** — same failure, more
  parameters. Fitting the periods is exactly what made it overfit.
- **Seasonality in the frequency domain** — candidate names with a plausible business cycle,
  read against the existing work on calendar effects, checked for a band that survived
  once one-off shocks and macro regime shifts were separated out. A periodogram always
  hands you peaks; nothing here was repeatable enough to trade.

## Layout

```
qm_data.py       data layer - point-in-time membership, incremental price cache,
                 fundamentals, risk-free rate. Nothing else touches the network.
qm_strategy.py   screen, momentum signal, backtest, live signal, metrics
qm_rebalance.py  signal + holdings -> trade list
run.py           end-to-end entry point
notebooks/       results.ipynb - plots the numbers in results/
results/         backtest output run.py writes (metrics, equity curves, current signal)
data/            bundled S&P 500 membership history (public)
```

## Running it

```bash
pip install -r requirements.txt
python3 run.py
```

Runs from a clean clone — prices are fetched on first run (~10-15 min for ~20 years across
every ticker ever in the index since 2007) and cached incrementally after that.

Not included: the brokerage tracker and its account exports. This repository is the
research code, not the portfolio.
