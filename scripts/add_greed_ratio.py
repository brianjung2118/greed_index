# ======================================================
# ADD GREED RATIO AND RETURNS TO ATTENTION GREED PANEL
# ======================================================
# Reads attention_greed_panel_daily.csv and:
#   1) Adds greed_ratio = greed_count / attn_volume (0 if either is 0).
#   2) Fetches historical prices (yfinance, .KS then .KQ), computes daily returns,
#      and aligns return at t+1 with row at t (so row at date t has next-day return).
#
# Output:
#   pipeline_output_attention/attention_greed_panel_daily_with_ratio.csv
#   Columns: company_code, dt, attn_volume, greed_count, greed_ratio, returns
#
# Run:
#   python add_greed_ratio.py
# ======================================================

from __future__ import annotations

import contextlib
import io
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ======================================================
# CONFIG
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "pipeline_output_attention" / "attention_greed_panel_daily.csv"
OUTPUT_PATH = BASE_DIR / "pipeline_output_attention" / "attention_greed_panel_daily_with_ratio.csv"

YF_SLEEP_SEC = 0.2  # throttle between yfinance requests

# ======================================================
# PRICE FETCH (Korean stocks: .KS = KOSPI, .KQ = KOSDAQ)
# ======================================================
# yfinance often returns DataFrames with MultiIndex columns (e.g. (Close, ticker)).
# We try .KS first; if it fails or returns empty, we try .KQ.


def _extract_close_series(tmp: pd.DataFrame) -> Optional[pd.Series]:
    """
    Extract close price series from yfinance result. Handles both single-level
    columns and MultiIndex columns (layout A: level0=field, level1=ticker;
    layout B: level0=ticker, level1=field).
    """
    if tmp is None or tmp.empty:
        return None

    close_series = None

    if isinstance(tmp.columns, pd.MultiIndex):
        lv0 = list(tmp.columns.get_level_values(0).unique())
        lv1 = list(tmp.columns.get_level_values(1).unique())
        for f in ["Close", "Adj Close"]:
            # Layout A: field (Close) in level 0
            if f in lv0:
                xs = tmp.xs(f, level=0, axis=1)
                close_series = xs if isinstance(xs, pd.Series) else xs.iloc[:, 0]
                break
            # Layout B: field in level 1
            if f in lv1:
                xs = tmp.xs(f, level=1, axis=1)
                close_series = xs if isinstance(xs, pd.Series) else xs.iloc[:, 0]
                break
        # Fallback: flatten and look for Close by name (e.g. ('Close', '005930.KS') -> first match)
        if close_series is None:
            flat = tmp.copy()
            if isinstance(flat.columns, pd.MultiIndex):
                flat.columns = [c[0] if isinstance(c, tuple) else c for c in flat.columns.tolist()]
            if "Close" in flat.columns:
                close_series = flat["Close"]
            elif "Adj Close" in flat.columns:
                close_series = flat["Adj Close"]
    else:
        if "Close" in tmp.columns:
            close_series = tmp["Close"]
        elif "Adj Close" in tmp.columns:
            close_series = tmp["Adj Close"]

    if close_series is None:
        return None
    return pd.to_numeric(close_series, errors="coerce")


def _download_one(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download one ticker; return DataFrame with dt, close or None."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tmp = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

        if tmp is None or tmp.empty:
            return None

        close_series = _extract_close_series(tmp)
        if close_series is None:
            return None

        # tmp index matches close_series index (same row order); after reset_index order is preserved
        tmp = tmp.reset_index()
        date_col = "Date" if "Date" in tmp.columns else ("Datetime" if "Datetime" in tmp.columns else "index")
        dates = pd.to_datetime(tmp[date_col], errors="coerce").dt.normalize()
        close_vals = close_series.values if hasattr(close_series, "values") else close_series
        out = pd.DataFrame({"dt": dates, "close": close_vals}).dropna(subset=["dt", "close"])
        return out if not out.empty else None
    except Exception:
        return None


def fetch_prices_and_next_day_returns(
    codes: list[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    """
    For each code: try .KS (KOSPI) first; if that fails or is empty, try .KQ (KOSDAQ).
    Build (company_code, dt, close), compute daily return, then shift so that
    at dt=t we have return from t to t+1. Returns DataFrame: company_code, dt, returns.
    """
    start_str = start_dt.strftime("%Y-%m-%d")
    # need one extra day to compute return at last date
    end_plus = end_dt + pd.Timedelta(days=5)
    end_str = end_plus.strftime("%Y-%m-%d")

    frames = []
    for i, code in enumerate(codes):
        code = str(code).zfill(6)
        if (i + 1) % 50 == 0 or i == 0:
            logging.info(f"[PRICES] {i+1}/{len(codes)} {code}...")
        time.sleep(YF_SLEEP_SEC)

        px = None
        for ticker in [f"{code}.KS", f"{code}.KQ"]:  # KOSPI first, then KOSDAQ
            px = _download_one(ticker, start_str, end_str)
            if px is not None and not px.empty:
                break
        if px is None or px.empty:
            continue  # both .KS and .KQ failed for this code

        px["company_code"] = code
        px = px.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")
        px["ret_1d"] = px.groupby("company_code")["close"].pct_change()
        # at dt=t put the return from t to t+1 (i.e. next day's return)
        px["returns"] = px.groupby("company_code")["ret_1d"].shift(-1)
        px = px[["company_code", "dt", "returns"]].dropna(subset=["returns"])
        frames.append(px)

    if not frames:
        logging.warning("[PRICES] No price data retrieved for any code.")
        return pd.DataFrame(columns=["company_code", "dt", "returns"])

    out = pd.concat(frames, ignore_index=True)
    out["dt"] = pd.to_datetime(out["dt"]).dt.normalize()
    return out


# ======================================================
# MAIN
# ======================================================


def main() -> None:
    logging.info(f"Reading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, parse_dates=["dt"], low_memory=False)

    required = {"company_code", "dt", "attn_volume", "greed_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    df["company_code"] = df["company_code"].astype(str).str.zfill(6)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.normalize()
    logging.info(f"Rows: {len(df):,}  unique stocks: {df['company_code'].nunique():,}")

    # 1) greed_ratio
    df["greed_ratio"] = 0.0
    mask = (df["attn_volume"] > 0) & (df["greed_count"] > 0)
    df.loc[mask, "greed_ratio"] = df.loc[mask, "greed_count"] / df.loc[mask, "attn_volume"]

    # 2) returns (t+1 aligned with row at t)
    codes = sorted(df["company_code"].unique().tolist())
    start_dt = df["dt"].min()
    end_dt = df["dt"].max()
    logging.info(f"Fetching prices for {len(codes):,} stocks from {start_dt.date()} to {end_dt.date()} (yfinance)...")

    returns_df = fetch_prices_and_next_day_returns(codes, start_dt, end_dt)
    if returns_df.empty:
        logging.warning("No returns data; adding empty returns column.")
        df["returns"] = float("nan")
    else:
        logging.info(f"[PRICES] got returns for {returns_df['company_code'].nunique():,} stocks, {len(returns_df):,} rows")
        df = df.merge(returns_df, on=["company_code", "dt"], how="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Saved {OUTPUT_PATH}")
    logging.info(f"Columns: {list(df.columns)}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
