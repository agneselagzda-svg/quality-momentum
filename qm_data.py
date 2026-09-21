"""
qm_data.py — data layer for the Quality Momentum project
=========================================================
Everything that touches the network or the disk cache lives here, so the
strategy and tracker code can be run (and tested) fully offline.

* S&P 500 membership history (point-in-time) — bundled CSVs in data/, sourced
  from github.com/fja05680/sp500 (Wikipedia-derived, 1996 → today). Refreshed
  from GitHub when reachable; the bundled copy is used otherwise.
* Daily adjusted close + volume for every ticker that was ever in the index
  since PRICE_START — fetched with yfinance and cached in data/prices_*.csv.
  Fetches are incremental: only new dates / new tickers are requested.
* Current fundamentals (market cap, ROE, EPS, beta) for the live screen —
  cached in data/fundamentals.json, refreshed when older than 7 days.
* 13-week T-bill yield (^IRX) as the risk-free rate for Sharpe.

Nothing here is specific to Fidelity; the tracker has its own parsers.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

PRICE_START = "2005-06-01"          # 18 months before the earliest backtest date
MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "sp500_ticker_start_end.csv")
CURRENT_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv"

MEMBERSHIP_CSV = os.path.join(DATA, "sp500_ticker_start_end.csv")
CURRENT_CSV = os.path.join(DATA, "sp500.csv")
CLOSE_CSV = os.path.join(DATA, "prices_close.csv")
VOLUME_CSV = os.path.join(DATA, "prices_volume.csv")
MISSING_TXT = os.path.join(DATA, "missing_tickers.txt")
FUND_JSON = os.path.join(DATA, "fundamentals.json")
RF_CSV = os.path.join(DATA, "riskfree_irx.csv")

# Tickers that are a second share class of a company already in the universe.
# Key = the class we drop, value = the class we keep. Holding both would be a
# double position in one company (GOOG + GOOGL was 10% of the live account).
SHARE_CLASS_ALIAS = {
    "GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA", "BRK-A": "BRK-B",
    "DISCK": "DISCA", "UA": "UAA", "LEN-B": "LEN", "CMCSK": "CMCSA",
    "LBTYK": "LBTYA", "HEI-A": "HEI", "ZG": "Z", "MOG-B": "MOG-A",
    "LILAK": "LILA", "PBR-A": "PBR", "RDS-B": "RDS-A", "BF-A": "BF-B",
    "WSO-B": "WSO", "GEF-B": "GEF", "CWEN-A": "CWEN", "TAP-A": "TAP",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def yf_symbol(t: str) -> str:
    """Wikipedia/S&P style 'BRK.B' -> Yahoo style 'BRK-B'."""
    return str(t).strip().upper().replace(".", "-")


def _fetch_text(url: str, timeout: int = 20) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8")
    except Exception as e:  # offline / blocked
        log(f"  [data] could not fetch {url.split('/')[-1]}: {e.__class__.__name__}")
        return None


# ── S&P 500 membership ───────────────────────────────────────────────────
def refresh_membership_files() -> None:
    for url, path in ((MEMBERSHIP_URL, MEMBERSHIP_CSV), (CURRENT_URL, CURRENT_CSV)):
        txt = _fetch_text(url)
        if txt and ("ticker" in txt[:200].lower() or "Symbol" in txt[:200]):
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            log(f"  [data] refreshed {os.path.basename(path)}")


def load_membership() -> pd.DataFrame:
    """Columns: ticker (Yahoo style), start, end (NaT = still a member)."""
    if not os.path.exists(MEMBERSHIP_CSV):
        refresh_membership_files()
    if not os.path.exists(MEMBERSHIP_CSV):
        raise SystemExit("No S&P 500 membership file (data/sp500_ticker_start_end.csv) "
                         "and GitHub is unreachable.")
    m = pd.read_csv(MEMBERSHIP_CSV)
    m["ticker"] = m["ticker"].map(yf_symbol)
    m["start"] = pd.to_datetime(m["start_date"])
    m["end"] = pd.to_datetime(m["end_date"])
    return m[["ticker", "start", "end"]]


def members_at(membership: pd.DataFrame, date) -> set[str]:
    d = pd.Timestamp(date)
    ok = (membership["start"] <= d) & (membership["end"].isna() | (membership["end"] > d))
    return set(membership.loc[ok, "ticker"])


def load_current_constituents() -> pd.DataFrame:
    """Current S&P 500 with Security name and GICS Sector (Yahoo-style symbols)."""
    if not os.path.exists(CURRENT_CSV):
        refresh_membership_files()
    if not os.path.exists(CURRENT_CSV):
        return pd.DataFrame(columns=["Symbol", "Security", "GICS Sector"])
    c = pd.read_csv(CURRENT_CSV)
    c["Symbol"] = c["Symbol"].map(yf_symbol)
    return c


def sector_map() -> dict[str, str]:
    c = load_current_constituents()
    return dict(zip(c["Symbol"], c.get("GICS Sector", pd.Series(["Unknown"] * len(c)))))


def dedupe_share_classes(tickers) -> list[str]:
    """Drop secondary share classes so each company appears once."""
    tickers = list(tickers)
    tset = set(tickers)
    out, seen = [], set()
    for t in tickers:
        canon = SHARE_CLASS_ALIAS.get(t, t)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon if canon in tset else t)   # prefer the primary class
    return out


def all_tickers_since(membership: pd.DataFrame, since) -> list[str]:
    s = pd.Timestamp(since)
    ok = membership["end"].isna() | (membership["end"] >= s)
    return sorted(set(membership.loc[ok, "ticker"]))


# ── price cache ──────────────────────────────────────────────────────────
def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, volume) wide frames from the cache; empty frames if none."""
    if not os.path.exists(CLOSE_CSV):
        return pd.DataFrame(), pd.DataFrame()
    close = pd.read_csv(CLOSE_CSV, index_col=0, parse_dates=True)
    vol = pd.read_csv(VOLUME_CSV, index_col=0, parse_dates=True) if os.path.exists(VOLUME_CSV) \
        else pd.DataFrame(index=close.index)
    return close.sort_index(), vol.sort_index()


def _save_prices(close: pd.DataFrame, vol: pd.DataFrame) -> None:
    close.sort_index().to_csv(CLOSE_CSV, float_format="%.4f")
    vol.reindex(close.index).sort_index().to_csv(VOLUME_CSV, float_format="%.0f")


def _yf_download(batch: list[str], start: str, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    data = yf.download(batch, start=start, end=end, auto_adjust=True,
                       progress=False, threads=True, group_by="column")
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        vol = data["Volume"]
    else:  # single ticker
        close = data[["Close"]].rename(columns={"Close": batch[0]})
        vol = data[["Volume"]].rename(columns={"Volume": batch[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    vol.index = close.index
    return close, vol


def update_prices(tickers, start: str = PRICE_START, batch_size: int = 100,
                  force_full: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bring the cache up to date for `tickers` (+SPY). Returns (close, volume).

    Incremental: tickers already cached are refreshed from 10 days before the
    last cached date; tickers never seen are fetched from `start`. Tickers that
    return nothing are remembered in data/missing_tickers.txt so they are not
    re-requested every month (delete that file to retry them).
    """
    tickers = sorted(set(yf_symbol(t) for t in tickers) | {"SPY"})
    close, vol = load_prices()
    missing = set()
    if os.path.exists(MISSING_TXT) and not force_full:
        missing = set(open(MISSING_TXT).read().split())

    have = set(close.columns)
    new = [t for t in tickers if t not in have and t not in missing]
    old = [t for t in tickers if t in have]
    last = close.index.max() if not close.empty else None
    today = pd.Timestamp.today().normalize()

    jobs = []
    if new:
        jobs.append((new, start))
    if old and (last is None or last < today - pd.Timedelta(days=1) or force_full):
        jobs.append((old, start if (force_full or last is None)
                     else (last - pd.Timedelta(days=10)).strftime("%Y-%m-%d")))
    if not jobs:
        log("  [prices] cache is current; nothing to fetch")
        return close, vol

    got_any = False
    for lst, st in jobs:
        for i in range(0, len(lst), batch_size):
            batch = lst[i:i + batch_size]
            log(f"  [prices] fetching {len(batch)} tickers from {st} "
                f"({i + len(batch)}/{len(lst)})")
            try:
                c, v = _yf_download(batch, st)
            except Exception as e:
                log(f"  [prices] batch failed: {e}")
                continue
            if c.empty:
                continue
            got_any = True
            c = c.dropna(how="all", axis=1)
            v = v.reindex(columns=c.columns)
            close = c.combine_first(close) if not close.empty else c
            vol = v.combine_first(vol) if not vol.empty else v
            time.sleep(0.5)
    if not got_any and close.empty:
        raise SystemExit("Could not download any prices (is Yahoo Finance reachable "
                         "from this machine?).")
    still_missing = [t for t in tickers if t not in close.columns]
    with open(MISSING_TXT, "w") as f:
        f.write("\n".join(sorted(set(still_missing) | missing)))
    # drop rows that are entirely NaN (weekends that crept in) and save
    close = close.dropna(how="all")
    vol = vol.reindex(close.index)
    _save_prices(close, vol)
    log(f"  [prices] cache: {close.shape[1]} tickers × {close.shape[0]} days, "
        f"last = {close.index.max():%Y-%m-%d}; {len(still_missing)} tickers unavailable")
    return close, vol


# ── fundamentals for the live screen ─────────────────────────────────────
def load_fundamentals(max_age_days: int = 7) -> dict | None:
    if not os.path.exists(FUND_JSON):
        return None
    age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(FUND_JSON))).days
    if age > max_age_days:
        return None
    with open(FUND_JSON) as f:
        return json.load(f)


def fetch_fundamentals(tickers, max_age_days: int = 7) -> dict:
    """{ticker: {marketCap, returnOnEquity, trailingEps, beta, averageVolume,
    sector, name}} via yfinance .info. Cached for `max_age_days`."""
    cached = load_fundamentals(max_age_days)
    if cached and set(tickers) <= set(cached["data"]):
        log(f"  [fundamentals] using cache from {cached['asof']}")
        return cached["data"]
    import yfinance as yf
    out = dict(cached["data"]) if cached else {}
    todo = [t for t in tickers if t not in out]
    log(f"  [fundamentals] fetching {len(todo)} tickers (slow: ~1s each)")
    for i, t in enumerate(todo, 1):
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        out[t] = {
            "marketCap": info.get("marketCap"),
            "returnOnEquity": info.get("returnOnEquity"),
            "trailingEps": info.get("trailingEps"),
            "beta": info.get("beta"),
            "averageVolume": info.get("averageVolume"),
            "sector": info.get("sector"),
            "name": info.get("shortName", t),
        }
        if i % 25 == 0:
            log(f"    {i}/{len(todo)}")
            time.sleep(0.5)
    with open(FUND_JSON, "w") as f:
        json.dump({"asof": datetime.now().strftime("%Y-%m-%d"), "data": out}, f, indent=1)
    return out


# ── risk-free rate ───────────────────────────────────────────────────────
def risk_free_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free return aligned to `index` from ^IRX (13-week T-bill,
    annualised %). Cached; falls back to 0 if unavailable."""
    rf = None
    if os.path.exists(RF_CSV):
        rf = pd.read_csv(RF_CSV, index_col=0, parse_dates=True).iloc[:, 0]
    need_fetch = rf is None or rf.index.max() < index.max() - pd.Timedelta(days=7)
    if need_fetch:
        try:
            import yfinance as yf
            h = yf.download("^IRX", start=PRICE_START, progress=False, auto_adjust=False)
            s = h["Close"].squeeze()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            rf = s.dropna()
            rf.to_csv(RF_CSV, header=["irx"])
        except Exception as e:
            log(f"  [rf] ^IRX unavailable ({e.__class__.__name__}); "
                f"{'using cache' if rf is not None else 'assuming 0%'}")
    if rf is None:
        return pd.Series(0.0, index=index)
    ann = rf.reindex(index).ffill().bfill() / 100.0
    return (1 + ann) ** (1 / 252) - 1
