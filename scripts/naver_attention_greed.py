# ======================================================
# NAVER DISCUSSION: ATTENTION + GREED COUNTS (DAILY PANEL)
# ======================================================
# Purpose:
#   Same data source as naver_attention (feeds.dr_post_meta), but uses the
#   "title" column to:
#   1) Build daily attention measures (post volume only; no abnormal_z).
#   2) Count how many posts per (company_code, dt) contain any greed phrase
#      from lexicon_greed_final.txt.
#
# Output:
#   pipeline_output_attention/attention_greed_panel_daily.csv
#   Columns: company_code, dt, attn_volume, greed_count
#
# Date range: MIN_POST_DT to MAX_POST_DT (both inclusive)
#
# Run:
#   python naver_attention_greed.py
# ======================================================

from __future__ import annotations

import re
import logging
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ======================================================
# CONFIG
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "pipeline_output_attention"
PANEL_OUT = OUTPUT_DIR / "attention_greed_panel_daily.csv"
LEXICON_PATH = BASE_DIR / "lexicon_greed_final.txt"

MYSQL_HOST = "fngo-ml-rds-cluster-8-cluster.cluster-ro-c6btgg8fszdb.ap-northeast-2.rds.amazonaws.com"
MYSQL_USER = "fngoMLAdmin"
MYSQL_PASSWORD = "fngo_2020-for!Knowledge"
MYSQL_PORT = 3306
MYSQL_DEFAULT_DB = "news"
FULL_TABLE = "feeds.dr_post_meta"

MIN_POST_DT = "2025-12-01"  # inclusive; data from this date
MAX_POST_DT = "2026-02-11"  # inclusive; data through this date
TITLE_COL = "title"


# ======================================================
# TEXT NORMALISATION (same as build_greed_lexicon)
# ======================================================

def normalise_title(s: str) -> str:
    """Normalise for matching: lowercase, keep Korean/English/digits/space."""
    s = str(s).strip().lower()
    s = re.sub(r"[/|·•]", " ", s)
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ======================================================
# LEXICON
# ======================================================

def load_greed_phrases(path: Path) -> list[str]:
    """Load phrases from lexicon_greed_final.txt; skip comments and section headers."""
    if not path.exists():
        raise FileNotFoundError(f"Lexicon not found: {path}")
    phrases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            phrases.append(line)
    return phrases


def build_normalised_phrases(phrases: list[str]) -> list[str]:
    """Return list of normalised phrases for substring matching."""
    return [normalise_title(p) for p in phrases if p]


def title_contains_greed(norm_title: str, norm_phrases: list[str]) -> bool:
    """True if norm_title contains any of norm_phrases as substring."""
    if not norm_title or not norm_phrases:
        return False
    for phrase in norm_phrases:
        if phrase and phrase in norm_title:
            return True
    return False


# ======================================================
# DB (same pattern as build_greed_lexicon / naver_sample)
# ======================================================

def make_engine():
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DEFAULT_DB}?charset=utf8mb4"
    )
    return create_engine(url)


def list_week_buckets(engine, min_dt: str, max_dt: str):
    """Return list of YEARWEEK values from min_dt through max_dt.
    Uses YEARWEEK() for efficient bucketing; each bucket is ~7 days of data.
    """
    sql = text(
        f"""
        SELECT DISTINCT YEARWEEK(published_at) AS yw
        FROM {FULL_TABLE}
        WHERE DATE(published_at) >= :min_dt
          AND DATE(published_at) <= :max_dt
          AND company_code IS NOT NULL
        ORDER BY yw ASC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"min_dt": min_dt, "max_dt": max_dt}).fetchall()
    return [r[0] for r in rows if r and r[0] is not None]


# ======================================================
# FETCH ONE WEEK + BUILD PANEL (used per bucket)
# ======================================================

def fetch_week_posts(engine, bucket: int, min_dt: str, max_dt: str) -> pd.DataFrame:
    """Fetch one week of posts; return cleaned DataFrame."""
    sql = text(
        f"""
        SELECT
            published_at AS dt,
            company_code,
            {TITLE_COL}
        FROM {FULL_TABLE}
        WHERE DATE(published_at) >= :min_dt
          AND DATE(published_at) <= :max_dt
          AND YEARWEEK(published_at) = :bucket
          AND company_code IS NOT NULL
        """
    )
    df = pd.read_sql(
        sql, engine,
        params={"min_dt": min_dt, "max_dt": max_dt, "bucket": bucket},
        parse_dates=["dt"],
    )
    if df.empty:
        return df

    df["company_code"] = df["company_code"].astype(str).str.zfill(6)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["dt", "company_code"]).copy()
    df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)
    return df


def build_attention_greed_panel(
    df_posts: pd.DataFrame,
    norm_phrases: list[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Returns panel: company_code x dt with attn_volume, greed_count.
    """
    df = df_posts.copy()
    df["_title_norm"] = df[TITLE_COL].map(normalise_title)
    df["_greed"] = df["_title_norm"].map(
        lambda t: 1 if title_contains_greed(t, norm_phrases) else 0
    )
    # Attention: total posts per (company_code, dt)
    attn = (
        df.groupby(["company_code", "dt"])
        .size()
        .reset_index(name="attn_volume")
    )
    # Greed: count of posts with greed per (company_code, dt)
    greed = (
        df.groupby(["company_code", "dt"])["_greed"]
        .sum()
        .reset_index(name="greed_count")
    )
    greed["greed_count"] = greed["greed_count"].astype(int)

    panel = attn.merge(greed, on=["company_code", "dt"], how="left")
    panel["greed_count"] = panel["greed_count"].fillna(0).astype(int)
    panel = panel.sort_values(["company_code", "dt"]).reset_index(drop=True)

    if verbose and not panel.empty:
        logging.info(f"[PANEL] rows={len(panel):,} stocks={panel['company_code'].nunique():,} greed={panel['greed_count'].sum():,}")

    return panel


# ======================================================
# MAIN: Incremental fetch by week, process, append to CSV
# ======================================================
# Memory-safe design: Each week is fetched → processed → appended to CSV → discarded.
# Only one week of data in RAM at a time. No pd.concat or accumulation.
# CSV is updated on every non-empty week: mode="w" (first) then mode="a" (append).


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    phrases = load_greed_phrases(LEXICON_PATH)
    norm_phrases = build_normalised_phrases(phrases)
    logging.info(f"[LEXICON] loaded {len(phrases)} phrases from {LEXICON_PATH.name}")

    engine = make_engine()
    buckets = list_week_buckets(engine, MIN_POST_DT, MAX_POST_DT)
    if not buckets:
        logging.warning("No week buckets found. Check MIN_POST_DT / MAX_POST_DT / table / permissions.")
        return

    logging.info("")
    logging.info(f"Fetching {len(buckets)} weeks from {MIN_POST_DT} to {MAX_POST_DT}")
    logging.info(f"Output: {PANEL_OUT}")
    logging.info("")

    total_posts = 0
    total_panel_rows = 0
    first_write = True
    t_start = time.perf_counter()

    for i, bucket in enumerate(buckets):
        pct = 100 * (i + 1) / len(buckets)
        logging.info(f"[{i+1}/{len(buckets)}] ({pct:.0f}%) Week {bucket}: fetching from MySQL...")

        t0 = time.perf_counter()
        df_week = fetch_week_posts(engine, int(bucket), MIN_POST_DT, MAX_POST_DT)
        t_fetch = time.perf_counter() - t0

        if df_week.empty:
            logging.info(f"         Week {bucket}: no data, skipping")
            continue

        total_posts += len(df_week)
        logging.info(f"         Fetched {len(df_week):,} posts in {t_fetch:.1f}s → processing...")

        panel_week = build_attention_greed_panel(df_week, norm_phrases, verbose=False)
        del df_week

        if panel_week.empty:
            continue

        total_panel_rows += len(panel_week)
        d_min, d_max = panel_week["dt"].min().date(), panel_week["dt"].max().date()

        panel_week.to_csv(
            PANEL_OUT,
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )
        first_write = False

        elapsed = time.perf_counter() - t_start
        avg_per_week = elapsed / (i + 1)
        remaining_weeks = len(buckets) - (i + 1)
        eta_sec = avg_per_week * remaining_weeks if remaining_weeks > 0 else 0
        eta_str = f"{eta_sec/60:.0f}m left" if eta_sec >= 60 else f"{eta_sec:.0f}s left"

        logging.info(f"         Saved {len(panel_week):,} panel rows ({d_min}~{d_max}) | total: {total_posts:,} posts, {total_panel_rows:,} panel rows | elapsed: {elapsed/60:.1f}m | {eta_str}")
        logging.info("")
        del panel_week

    elapsed_total = time.perf_counter() - t_start
    logging.info("")
    logging.info(f"[SAVED] {PANEL_OUT}")
    logging.info(f"Done. total_posts={total_posts:,}  total_panel_rows={total_panel_rows:,}  time={elapsed_total/60:.1f}m")


if __name__ == "__main__":
    main()
