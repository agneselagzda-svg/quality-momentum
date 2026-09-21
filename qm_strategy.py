"""
qm_strategy.py — Quality Momentum: honest backtest + live signal
=================================================================
Strategy (unchanged from the thesis):
    universe  S&P 500
    screen    price > $10, avg volume > 2M sh/day, beta 0.8–2.0,
              (+ ROE > 10 %, EPS > 0, market cap > $10B when fundamentals exist)
    signal    12-1 month total-return momentum
    hold      top 20, equal weight, monthly rebalance at the month-end close
    hysteresis keep a holding while it is still in the top 30
    cost      10 bps on traded notional

What "honest" means here, vs the original quality_momentum_fix2.py:
    * Point-in-time universe. Membership comes from a dated add/remove
      history (1996 → today), so a stock is only a candidate on dates it was
      actually in the index. The old script used today's list for all history.
    * No look-ahead fundamentals in the backtest. ROE / EPS / market cap are
      only known *today*, so the backtest screen is price-based (price, volume,
      rolling beta). The fundamentals are still applied to the LIVE signal —
      that is legitimate because they are known at trade time. A second
      backtest with today's fundamentals frozen is run for comparison; the
      gap between the two is the size of the look-ahead flattery.
    * Buy-and-hold between rebalances. The old loop re-weighted to 1/N every
      day at zero cost. Here shares are held; weights drift; costs are charged
      on real turnover.
    * Delistings are handled: a holding whose price stops printing is sold at
      its last price on the next rebalance. Data coverage (members with price
      history / members) is reported per year so you can see how much
      survivorship bias remains from tickers Yahoo no longer serves.
    * Sharpe uses the 13-week T-bill as the risk-free rate.
    * One company = one position (GOOG/GOOGL etc. de-duplicated).
    * Longer window (2007 →) so 2008–09, 2011, 2015–16, 2018, 2020 and 2022
      are all in the sample, not just the 2023–25 AI bull run.

Run:  python qm_strategy.py            (needs a filled price cache — see monthly.py)
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

import qm_data as D

# ── parameters ───────────────────────────────────────────────────────────
TOP_N = 20
HYSTERESIS = 30
LOOKBACK_DAYS = 252
SKIP_DAYS = 21
TX_COST_BPS = 10
MIN_PRICE = 10.0
MIN_AVG_VOLUME = 2e6        # shares/day, 63-day average
BETA_RANGE = (0.8, 2.0)     # 252-day rolling beta vs SPY (daily returns)
MIN_MARKET_CAP = 10e9       # live screen only (needs current fundamentals)
MIN_ROE = 0.10              # live screen only
BACKTEST_START = "2007-01-01"
REBALANCE_TO_EQUAL_WEIGHT = True   # False = only trade the names that change

OUT_DIR = os.path.join(D.HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────
def month_end_dates(index: pd.DatetimeIndex, start, end=None) -> pd.DatetimeIndex:
    """Last trading day of each month in `index`, within [start, end]."""
    s = pd.Series(index, index=index)
    me = s.groupby([index.year, index.month]).last()
    me = pd.DatetimeIndex(me.values)
    me = me[me >= pd.Timestamp(start)]
    if end is not None:
        me = me[me <= pd.Timestamp(end)]
    return me


def momentum_matrix(close: pd.DataFrame) -> pd.DataFrame:
    """12-1 momentum for every day: close[t-21] / close[t-252] - 1."""
    return close.shift(SKIP_DAYS) / close.shift(LOOKBACK_DAYS) - 1


def rolling_beta(close: pd.DataFrame, spy: pd.Series, window: int = 252) -> pd.DataFrame:
    r = close.pct_change()
    rs = spy.reindex(close.index).pct_change()
    cov = r.rolling(window, min_periods=int(window * 0.8)).cov(rs)
    var = rs.rolling(window, min_periods=int(window * 0.8)).var()
    return cov.div(var, axis=0)


def screen_at(date, close, avgvol, beta, membership, fundamentals=None,
              require_history=True) -> list[str]:
    """Tickers eligible on `date`. Price-based screen is point-in-time; the
    fundamentals screen (if a dict is given) uses whatever is in the dict."""
    members = D.members_at(membership, date) & set(close.columns)
    if not members:
        return []
    cols = sorted(members)
    px = close.loc[date, cols]
    ok = px.notna() & (px >= MIN_PRICE)
    ok &= avgvol.loc[date, cols].fillna(0) >= MIN_AVG_VOLUME
    b = beta.loc[date, cols]
    ok &= b.notna() & (b >= BETA_RANGE[0]) & (b <= BETA_RANGE[1])
    if require_history:
        loc = close.index.get_loc(date)
        if loc < LOOKBACK_DAYS + 5:
            return []
        hist_ok = close[cols].iloc[loc - LOOKBACK_DAYS - 5: loc + 1].notna().sum() >= LOOKBACK_DAYS * 0.9
        ok &= hist_ok
    elig = [t for t in cols if ok[t]]
    if fundamentals:
        keep = []
        for t in elig:
            f = fundamentals.get(t) or {}
            roe, eps, mc = f.get("returnOnEquity"), f.get("trailingEps"), f.get("marketCap")
            if roe is None or roe < MIN_ROE:
                continue
            if eps is not None and eps <= 0:
                continue
            if mc is not None and mc < MIN_MARKET_CAP:
                continue
            keep.append(t)
        elig = keep
    return D.dedupe_share_classes(elig)


def select_portfolio(ranked: pd.Series, current: set[str]) -> list[str]:
    """Hysteresis: keep current names still in the top HYSTERESIS, then fill
    from the top of the ranking. `ranked` is momentum sorted descending."""
    top_h = list(ranked.index[:HYSTERESIS])
    keep = [t for t in top_h if t in current]
    out = list(keep)
    for t in ranked.index:
        if len(out) >= TOP_N:
            break
        if t not in out:
            out.append(t)
    return out[:TOP_N]


# ── backtest engine ──────────────────────────────────────────────────────
def backtest(close: pd.DataFrame, volume: pd.DataFrame, membership: pd.DataFrame,
             spy: pd.Series, start=BACKTEST_START, end=None, fundamentals=None,
             rebalance_weights=REBALANCE_TO_EQUAL_WEIGHT, label="honest") -> dict:
    close = close.sort_index()
    end = pd.Timestamp(end) if end else close.index[-1]
    idx = close.index
    mom = momentum_matrix(close)
    avgvol = volume.reindex(idx).rolling(63, min_periods=40).mean()
    beta = rolling_beta(close, spy)
    px_ff = close.ffill()                       # last known price for valuation
    last_valid = close.apply(lambda s: s.last_valid_index())

    rebal = month_end_dates(idx, start, end)
    if len(rebal) < 3:
        raise SystemExit("Not enough price history for a backtest.")
    days = idx[(idx >= rebal[0]) & (idx <= end)]

    cash = 1.0
    shares: dict[str, float] = {}
    values, log, coverage = [], [], []
    turnover_total, n_delist = 0.0, 0
    rebal_set = set(rebal)

    for d in days:
        pxd = px_ff.loc[d]
        equity = sum(q * pxd[t] for t, q in shares.items() if pd.notna(pxd[t]))
        value = cash + equity
        if d in rebal_set:
            elig = screen_at(d, close, avgvol, beta, membership, fundamentals)
            members = D.members_at(membership, d)
            coverage.append({"date": d, "members": len(members),
                             "with_prices": len(members & set(close.columns)),
                             "eligible": len(elig)})
            # force-sell anything that has stopped trading
            for t in list(shares):
                lv = last_valid[t]
                if lv is None or lv < d - pd.Timedelta(days=10):
                    q = shares.pop(t)
                    if lv is not None:
                        cash += q * px_ff.loc[lv, t]
                    n_delist += 1
            equity = sum(q * pxd[t] for t, q in shares.items())
            value = cash + equity
            if len(elig) >= TOP_N:
                ranked = mom.loc[d, elig].dropna().sort_values(ascending=False)
                target = select_portfolio(ranked, set(shares))
                sells = [t for t in shares if t not in target]
                buys = [t for t in target if t not in shares]
                traded = 0.0
                for t in sells:
                    traded += shares[t] * pxd[t]
                    cash += shares.pop(t) * pxd[t]
                if rebalance_weights:
                    per = value / len(target)
                    for t in target:
                        cur = shares.get(t, 0.0) * pxd[t]
                        delta = per - cur
                        if abs(delta) > 1e-9:
                            shares[t] = per / pxd[t]
                            cash -= delta
                            traded += abs(delta)
                else:
                    if buys:
                        per = max(cash, 0.0) / len(buys)
                        for t in buys:
                            shares[t] = per / pxd[t]
                            cash -= per
                            traded += per
                cost = traded * TX_COST_BPS / 1e4
                cash -= cost
                turnover_total += traded / value if value else 0
                log.append({"date": d.strftime("%Y-%m-%d"), "n": len(target),
                            "sells": sells, "buys": buys,
                            "holdings": [{"ticker": t, "mom": round(float(ranked.get(t, np.nan)) * 100, 1)}
                                         for t in target]})
                equity = sum(q * pxd[t] for t, q in shares.items())
                value = cash + equity
        values.append((d, value, len(shares)))

    vals = pd.DataFrame(values, columns=["date", "value", "n"]).set_index("date")
    return {"label": label, "values": vals, "log": log, "coverage": pd.DataFrame(coverage),
            "turnover_per_rebalance": turnover_total / max(len(log), 1),
            "n_delist": n_delist, "rebalances": len(log)}


# ── metrics ──────────────────────────────────────────────────────────────
def metrics(values: pd.Series, spy: pd.Series, rf_daily: pd.Series) -> dict:
    v = values.dropna()
    b = spy.reindex(v.index).ffill()
    b = b / b.iloc[0]
    rf = rf_daily.reindex(v.index).fillna(0)
    out = {}
    for name, s in (("strategy", v), ("spy", b)):
        r = s.pct_change().dropna()
        ex = r - rf.reindex(r.index)
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
        vol = r.std() * np.sqrt(252)
        sharpe = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
        down = ex[ex < 0].std() * np.sqrt(252)
        sortino = ex.mean() * 252 / down if down > 0 else 0
        dd = (s / s.cummax() - 1)
        out[name] = {"cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino,
                     "max_dd": dd.min(), "max_dd_date": dd.idxmin().strftime("%Y-%m-%d"),
                     "total": s.iloc[-1] / s.iloc[0] - 1, "years": yrs}
    # calendar years
    ys = v.resample("YE").last()
    yb = b.resample("YE").last()
    first_v, first_b = v.iloc[0], b.iloc[0]
    rows = []
    prev_v, prev_b = first_v, first_b
    for dt in ys.index:
        rows.append({"year": dt.year, "strategy": ys[dt] / prev_v - 1, "spy": yb[dt] / prev_b - 1})
        prev_v, prev_b = ys[dt], yb[dt]
    out["by_year"] = rows
    # rolling 12m relative
    m_v = v.resample("ME").last(); m_b = b.resample("ME").last()
    rel12 = (m_v / m_v.shift(12)) / (m_b / m_b.shift(12)) - 1
    out["worst_12m_vs_spy"] = float(rel12.min()) if rel12.notna().any() else None
    out["worst_12m_vs_spy_date"] = rel12.idxmin().strftime("%Y-%m") if rel12.notna().any() else None
    mrel = m_v.pct_change() - m_b.pct_change()
    out["pct_months_beat_spy"] = float((mrel.dropna() > 0).mean())
    out["excess_cagr"] = out["strategy"]["cagr"] - out["spy"]["cagr"]
    return out


def sub_period_table(values: pd.Series, spy: pd.Series, rf: pd.Series, periods) -> list[dict]:
    rows = []
    for name, a, b in periods:
        s = values.loc[a:b]
        if len(s) < 200:
            continue
        m = metrics(s, spy, rf)
        rows.append({"period": name, "start": s.index[0].strftime("%Y-%m-%d"),
                     "end": s.index[-1].strftime("%Y-%m-%d"),
                     "strategy_cagr": m["strategy"]["cagr"], "spy_cagr": m["spy"]["cagr"],
                     "strategy_sharpe": m["strategy"]["sharpe"], "spy_sharpe": m["spy"]["sharpe"],
                     "strategy_maxdd": m["strategy"]["max_dd"], "spy_maxdd": m["spy"]["max_dd"]})
    return rows


# ── live signal ──────────────────────────────────────────────────────────
def live_signal(close, volume, membership, spy, fundamentals, current_holdings: set[str],
                asof=None) -> tuple[pd.DataFrame, dict]:
    """Ranking as of the last completed month-end (or `asof`). Hysteresis is
    applied against the account's ACTUAL holdings, not a simulated portfolio."""
    idx = close.index
    if asof is None:
        me = month_end_dates(idx, idx[0])
        # only a COMPLETED month counts: the last cached day is a month-end only
        # if the next business day falls in a different month
        if (me[-1] + pd.offsets.BDay(1)).month == me[-1].month:
            me = me[:-1]
        asof = me[-1]
    asof = pd.Timestamp(asof)
    avgvol = volume.reindex(idx).rolling(63, min_periods=40).mean()
    beta = rolling_beta(close, spy)
    mom = momentum_matrix(close)
    elig = screen_at(asof, close, avgvol, beta, membership, fundamentals)
    if fundamentals is None:
        D.log("  [signal] WARNING: no fundamentals — ROE/EPS/market-cap screen skipped")
    ranked = mom.loc[asof, elig].dropna().sort_values(ascending=False)
    target = select_portfolio(ranked, current_holdings)
    sectors = D.sector_map()
    rows = []
    for rank, (t, m) in enumerate(ranked.items(), 1):
        f = (fundamentals or {}).get(t) or {}
        status = ("HOLD" if t in target and t in current_holdings else
                  "BUY" if t in target else
                  "SELL" if t in current_holdings else "")
        rows.append({"rank": rank, "ticker": t, "status": status,
                     "mom_12_1_pct": round(m * 100, 1),
                     "beta_252d": round(float(beta.loc[asof, t]), 2),
                     "avg_vol_M": round(float(avgvol.loc[asof, t]) / 1e6, 1),
                     "roe_pct": None if f.get("returnOnEquity") is None else round(f["returnOnEquity"] * 100, 1),
                     "mcap_B": None if f.get("marketCap") is None else round(f["marketCap"] / 1e9, 1),
                     "sector": sectors.get(t, f.get("sector") or "?"),
                     "name": f.get("name", "")})
    df = pd.DataFrame(rows)
    # holdings that are no longer eligible at all (failed the screen) → SELL
    for t in current_holdings:
        if t not in ranked.index:
            df.loc[len(df)] = {"rank": None, "ticker": t, "status": "SELL", "mom_12_1_pct": None,
                               "beta_252d": None, "avg_vol_M": None, "roe_pct": None,
                               "mcap_B": None, "sector": sectors.get(t, "?"),
                               "name": "(failed screen or left index)"}
    info = {"asof": asof.strftime("%Y-%m-%d"), "eligible": len(elig), "target": target,
            "buys": [t for t in target if t not in current_holdings],
            "sells": [t for t in current_holdings if t not in target],
            "sector_mix": dict(Counter(sectors.get(t, "?") for t in target)),
            "fundamentals_used": fundamentals is not None}
    return df, info


# ── runner ───────────────────────────────────────────────────────────────
def run_backtests(close, volume, membership, spy, fundamentals=None, verbose=True) -> dict:
    rf = D.risk_free_daily(close.index)
    results = {}
    runs = [("honest", None, REBALANCE_TO_EQUAL_WEIGHT),
            ("honest_trade_changes_only", None, False)]
    if fundamentals:
        runs.append(("lookahead_fundamentals", fundamentals, REBALANCE_TO_EQUAL_WEIGHT))
    for label, fund, rw in runs:
        D.log(f"  [backtest] {label} …")
        bt = backtest(close, volume, membership, spy, fundamentals=fund,
                      rebalance_weights=rw, label=label)
        m = metrics(bt["values"]["value"], spy, rf)
        periods = [("2007-2012 (GFC + recovery)", "2007-01-01", "2012-12-31"),
                   ("2013-2019", "2013-01-01", "2019-12-31"),
                   ("2020-2022 (covid + rate shock)", "2020-01-01", "2022-12-31"),
                   ("2023-now (original backtest window)", "2023-01-01", "2099-01-01")]
        m["sub_periods"] = sub_period_table(bt["values"]["value"], spy, rf, periods)
        m["turnover_per_rebalance"] = bt["turnover_per_rebalance"]
        m["delistings_handled"] = bt["n_delist"]
        m["rebalances"] = bt["rebalances"]
        cov = bt["coverage"]
        cov["year"] = cov["date"].dt.year
        m["coverage_by_year"] = (cov.groupby("year")[["members", "with_prices", "eligible"]]
                                 .mean().round(0).astype(int).reset_index().to_dict("records"))
        results[label] = {"metrics": m, "values": bt["values"], "log": bt["log"]}
        if verbose:
            print_metrics(label, m)
    return results


def print_metrics(label: str, m: dict) -> None:
    s, b = m["strategy"], m["spy"]
    print(f"\n{'━' * 64}\n  {label.upper()}   ({s['years']:.1f} years)\n{'━' * 64}")
    print(f"  {'Metric':<22}{'Strategy':>12}{'SPY':>12}")
    for k, fmt in (("cagr", "{:.1%}"), ("vol", "{:.1%}"), ("sharpe", "{:.2f}"),
                   ("sortino", "{:.2f}"), ("max_dd", "{:.1%}"), ("total", "{:.0%}")):
        print(f"  {k:<22}{fmt.format(s[k]):>12}{fmt.format(b[k]):>12}")
    print(f"  {'worst 12m vs SPY':<22}{m['worst_12m_vs_spy']:>12.1%}   ({m['worst_12m_vs_spy_date']})")
    print(f"  {'months beating SPY':<22}{m['pct_months_beat_spy']:>12.0%}")
    print(f"  {'turnover / rebalance':<22}{m['turnover_per_rebalance']:>12.0%}")
    print(f"\n  {'Year':<8}{'Strategy':>10}{'SPY':>10}{'Excess':>10}")
    for r in m["by_year"]:
        print(f"  {r['year']:<8}{r['strategy']:>10.1%}{r['spy']:>10.1%}{r['strategy'] - r['spy']:>10.1%}")
    print("\n  Sub-periods:")
    for r in m["sub_periods"]:
        print(f"  {r['period']:<38} strat {r['strategy_cagr']:>6.1%}  SPY {r['spy_cagr']:>6.1%}"
              f"   Sharpe {r['strategy_sharpe']:.2f} vs {r['spy_sharpe']:.2f}"
              f"   MaxDD {r['strategy_maxdd']:.0%} vs {r['spy_maxdd']:.0%}")
    print("\n  Data coverage (avg per rebalance): members → with prices → pass screen")
    for r in m["coverage_by_year"][::3]:
        print(f"    {r['year']}: {r['members']} → {r['with_prices']} → {r['eligible']}")


def save_results(results: dict, signal_df: pd.DataFrame | None, signal_info: dict | None) -> str:
    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "parameters": {k: globals()[k] for k in
                          ("TOP_N", "HYSTERESIS", "LOOKBACK_DAYS", "SKIP_DAYS", "TX_COST_BPS",
                           "MIN_PRICE", "MIN_AVG_VOLUME", "BETA_RANGE", "MIN_MARKET_CAP",
                           "MIN_ROE", "BACKTEST_START", "REBALANCE_TO_EQUAL_WEIGHT")},
           "backtests": {k: v["metrics"] for k, v in results.items()},
           "signal": signal_info}
    path = os.path.join(OUT_DIR, "backtest_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    for k, v in results.items():
        v["values"].to_csv(os.path.join(OUT_DIR, f"equity_curve_{k}.csv"))
        with open(os.path.join(OUT_DIR, f"holdings_log_{k}.json"), "w") as f:
            json.dump(v["log"], f)
    if signal_df is not None:
        signal_df.to_csv(os.path.join(OUT_DIR, f"signal_{signal_info['asof']}.csv"), index=False)
    return path


if __name__ == "__main__":
    close, volume = D.load_prices()
    if close.empty:
        raise SystemExit("Price cache is empty — run monthly.py first (it fetches prices).")
    membership = D.load_membership()
    spy = close["SPY"]
    fund = D.load_fundamentals(max_age_days=60)
    fund = fund["data"] if fund else None
    res = run_backtests(close, volume, membership, spy, fund)
    sig, info = live_signal(close, volume, membership, spy, fund, set())
    print(f"\nSignal as of {info['asof']}: {info['target']}")
    print("Saved:", save_results(res, sig, info))
