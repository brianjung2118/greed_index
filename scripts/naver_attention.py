# ======================================================
# NAVER DISCUSSION ATTENTION (ALL STOCKS) — FULL SCRIPT
# ======================================================
# Purpose:
#   Build DAILY attention metrics for ALL stocks from Naver discussion metadata
#   stored in MySQL table: feeds.dr_post_meta
#
# Outputs (under OUTPUT_DIR):
#   1) attention_panel_daily.csv
#        - company_code x dt panel with:
#          attn_volume, attn_log_volume, attn_abnormal_z
#
# Notes:
#   - "Attention" here = post volume (and its normalised variants).
#   - No prices. No salience. No backtest. Just attention.
# ======================================================

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ======================================================
# CONFIG
# ======================================================

BASE_DIR = "/Users/sanghoonjung/PycharmProjects/PythonProject/rii/greed"

# --- MySQL ---
MYSQL_CONFIG = {
    "host": "fngo-ml-rds-cluster-8-cluster.cluster-ro-c6btgg8fszdb.ap-northeast-2.rds.amazonaws.com",
    "user": "fngoMLAdmin",
    "password": "fngo_2020-for!Knowledge",
    "db": "news",
    "port": 3306,
    "charset": "utf8mb4",
}

# --- Output Paths ---
OUTPUT_DIR = f"{BASE_DIR}/pipeline_output_attention"
PANEL_OUT = f"{OUTPUT_DIR}/attention_panel_daily.csv"

# --- Parameters ---
MIN_POST_DT = "2020-01-01"   # inclusive
VOL_WINDOW = 20              # rolling window length for z-score
VOL_MIN_PERIODS = 10         # warmup
ATTN_Z_COL = "attn_abnormal_z"


# ======================================================
# STEP 1: LOAD DISCUSSION DATA
# ======================================================

def load_discussion_data() -> pd.DataFrame:
    """
    Loads minimal discussion metadata needed for attention:
    - dt (published_at normalised to date)
    - company_code (zero-padded to 6)
    """
    logging.info("[1] Loading discussion data from MySQL")

    conn = pymysql.connect(**MYSQL_CONFIG)

    sql = f"""
        SELECT
            published_at AS dt,
            company_code
        FROM feeds.dr_post_meta
        WHERE published_at >= '{MIN_POST_DT}'
          AND company_code IS NOT NULL
    """

    df = pd.read_sql(sql, conn, parse_dates=["dt"])
    conn.close()

    if df.empty:
        logging.warning("[DISC] No rows returned. Check MIN_POST_DT / table name / permissions.")
        return df

    df["company_code"] = df["company_code"].astype(str).str.zfill(6)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["dt", "company_code"]).copy()

    logging.info(f"[DISC] rows={len(df):,} unique_stocks={df['company_code'].nunique():,}")
    logging.info(f"[DATE CHECK] DISC: min={df['dt'].min().date()} max={df['dt'].max().date()}")

    return df


# ======================================================
# STEP 2: BUILD DAILY ATTENTION PANEL
# ======================================================

def build_attention_panel(df_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Returns panel data: company_code x dt

    Columns:
      - company_code
      - dt
      - attn_volume: daily post count
      - attn_log_volume: log1p(attn_volume)
      - attn_abnormal_z: rolling z-score of attn_log_volume within each stock
    """
    logging.info("[2] Building daily attention panel")

    if df_posts.empty:
        return pd.DataFrame(columns=["company_code", "dt", "attn_volume", "attn_log_volume", ATTN_Z_COL])

    daily = (
        df_posts.groupby(["company_code", "dt"])
        .size()
        .reset_index(name="attn_volume")
        .sort_values(["company_code", "dt"])
        .reset_index(drop=True)
    )

    daily["attn_log_volume"] = np.log1p(daily["attn_volume"]).astype(float)

    def _rolling_z(x: pd.Series) -> pd.Series:
        mu = x.rolling(VOL_WINDOW, min_periods=VOL_MIN_PERIODS).mean()
        sd = x.rolling(VOL_WINDOW, min_periods=VOL_MIN_PERIODS).std()
        z = (x - mu) / sd
        # avoid inf if sd=0
        return z.replace([np.inf, -np.inf], np.nan)

    daily[ATTN_Z_COL] = daily.groupby("company_code")["attn_log_volume"].transform(_rolling_z)

    # keep warm-up removed rows out of the panel output (optional but clean)
    daily = daily.dropna(subset=[ATTN_Z_COL]).copy()

    logging.info(f"[ATTN PANEL] rows={len(daily):,} unique_stocks={daily['company_code'].nunique():,}")
    if not daily.empty:
        logging.info(f"[DATE CHECK] ATTN PANEL: min={daily['dt'].min().date()} max={daily['dt'].max().date()}")

    return daily


# ======================================================
# MAIN
# ======================================================

def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    posts = load_discussion_data()
    panel = build_attention_panel(posts)

    panel.to_csv(PANEL_OUT, index=False)
    logging.info(f"[SAVED] {PANEL_OUT}")

    logging.info("Done.")
    logging.info(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()