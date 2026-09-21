#!/usr/bin/env python3
"""
run.py - end-to-end entry point.

    python3 run.py            membership -> prices -> backtests -> live signal
    python3 run.py --signal   skip the backtests, just print this month's target

First run downloads ~20 years of daily prices for every ticker that has ever
been in the S&P 500 since 2007 (roughly 10-15 min, cached in data/).
Later runs are incremental.
"""
import sys, time
import pandas as pd
import qm_data as D
import qm_strategy as S


def main(argv):
    signal_only = "--signal" in argv
    t0 = time.time()

    print("[1/4] S&P 500 membership history (point-in-time)")
    D.refresh_membership_files()
    membership = D.load_membership()
    universe = D.all_tickers_since(membership, D.PRICE_START)
    current = D.members_at(membership, pd.Timestamp.today())
    print(f"  {len(universe)} tickers ever in the index since {D.PRICE_START}; {len(current)} today")

    print("[2/4] Prices (yfinance -> data/prices_*.csv)")
    close, volume = D.update_prices(sorted(universe))
    spy = close["SPY"]

    print("[3/4] Fundamentals for the live screen")
    fund = None
    cached = D.load_fundamentals(max_age_days=7)
    if cached:
        fund = cached["data"]
        print(f"  cached from {cached['asof']}")
    else:
        try:
            fund = D.fetch_fundamentals(sorted(current & set(close.columns)))
        except Exception as e:
            print(f"  unavailable ({e}); live screen will be price-based only")

    print("[4/4] " + ("Live signal" if signal_only else "Backtests + live signal"))
    results = {} if signal_only else S.run_backtests(close, volume, membership, spy, fund, verbose=True)
    sig, info = S.live_signal(close, volume, membership, spy, fund, set())
    S.save_results(results, sig, info)

    print(f"\n  Signal as of {info['asof']}: {len(info['target'])} names, "
          f"{info['eligible']} passed the screen"
          + ("" if info["fundamentals_used"] else "  (PRICE-ONLY screen)"))
    top = sig[sig.status.isin(["HOLD", "BUY"])].sort_values("rank")
    print("  " + "  ".join(f"{r.ticker}" for r in top.itertuples()))
    print(f"\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main(sys.argv[1:])
