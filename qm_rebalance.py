"""
qm_rebalance.py — turn the signal + your actual holdings into an order list
===========================================================================
Rules (from the thesis, made explicit):
    * SELL everything the signal says to drop (fell out of the top 30, or
      failed the screen / left the index).
    * BUY each new name to the equal-weight target  = account value / 20.
    * Existing holdings are left alone unless their weight has drifted more
      than REBALANCE_BAND from target — then trim/top-up to target. This is
      the middle ground between "never rebalance" (what the account did:
      weights drifted 3% → 8%) and "rebalance everything monthly" (pointless
      churn on $50 positions).
    * Trades under MIN_TRADE dollars are skipped.
Output: output/trade_list_<asof>.csv + a printed list to type into Fidelity.
"""
from __future__ import annotations

import os

import pandas as pd

import qm_data as D

TOP_N = 20
REBALANCE_BAND = 0.25    # trade a held name only if |weight − target| > 25 % of target
MIN_TRADE = 5.0          # dollars
OUT_DIR = os.path.join(D.HERE, "output")


def trade_list(signal: pd.DataFrame, info: dict, holdings: list[dict], cash: float) -> pd.DataFrame:
    equity = sum(h["value_now"] for h in holdings if pd.notna(h["value_now"]))
    total = equity + cash
    target = total / TOP_N
    by_t = {h["ticker"]: h for h in holdings}
    rows = []
    for t in info["sells"]:
        h = by_t.get(t)
        if h:
            rows.append({"side": "SELL", "ticker": t, "dollars": round(h["value_now"], 2),
                         "shares": round(h["qty"], 4), "note": "sell ALL — dropped by signal"})
    for t in info["buys"]:
        rows.append({"side": "BUY", "ticker": t, "dollars": round(target, 2), "shares": None,
                     "note": "new name — buy to equal weight"})
    for t in info["target"]:
        h = by_t.get(t)
        if not h or t in info["buys"]:
            continue
        dev = (h["value_now"] - target) / target
        if abs(dev) > REBALANCE_BAND:
            delta = target - h["value_now"]
            rows.append({"side": "BUY" if delta > 0 else "SELL", "ticker": t,
                         "dollars": round(abs(delta), 2), "shares": None,
                         "note": f"weight drifted {dev:+.0%} from target — bring back to equal weight"})
    df = pd.DataFrame(rows, columns=["side", "ticker", "dollars", "shares", "note"])
    if df.empty:
        return df
    df = df[df.dollars >= MIN_TRADE].copy()
    # cash check: sells fund buys
    sells = df.loc[df.side == "SELL", "dollars"].sum()
    buys = df.loc[df.side == "BUY", "dollars"].sum()
    avail = cash + sells
    if buys > avail + 0.01:
        scale = avail / buys
        df.loc[df.side == "BUY", "dollars"] = (df.loc[df.side == "BUY", "dollars"] * scale).round(2)
        if scale < 0.98:
            df.loc[df.side == "BUY", "note"] += f" (scaled ×{scale:.2f} to fit available cash)"
    order = {"SELL": 0, "BUY": 1}
    df["o"] = df.side.map(order)
    df = df.sort_values(["o", "dollars"], ascending=[True, False]).drop(columns="o").reset_index(drop=True)
    df.attrs.update({"total": total, "target": target, "cash": cash})
    return df


def print_trades(df: pd.DataFrame, info: dict) -> None:
    print(f"\n{'━' * 64}\n  ORDERS — signal as of {info['asof']}"
          f"   (account ${df.attrs.get('total', 0):,.2f} → target ${df.attrs.get('target', 0):,.2f}/name)\n{'━' * 64}")
    if df.empty:
        print("  Nothing to do — holdings match the signal within the band.")
        return
    for side in ("SELL", "BUY"):
        part = df[df.side == side]
        if part.empty:
            continue
        print(f"\n  {side}  (dollar-based orders; place sells first, wait for them to fill)")
        for _, r in part.iterrows():
            sh = f"  all {r['shares']:.4f} sh" if pd.notna(r["shares"]) else ""
            print(f"    {r['ticker']:<6} ${r['dollars']:>8,.2f}{sh:<18}  {r['note']}")
    print(f"\n  Sector mix of the target: {info['sector_mix']}")


def save_trades(df: pd.DataFrame, info: dict) -> str:
    path = os.path.join(OUT_DIR, f"trade_list_{info['asof']}.csv")
    df.to_csv(path, index=False)
    return path
