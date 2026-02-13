from __future__ import annotations

"""
NAVER DISCUSSION: ATTENTION + GREED + GREED RATIO (FOCUS STOCKS, WEEKLY INCREMENTAL)
====================================================================================

Purpose
-------
Same data source as `naver_attention_greed.py` (feeds.dr_post_meta), but:
- Restricts to a small set of focus stocks (by company_code).
- Uses the full available history (no MIN/MAX post date filter).
- Processes data **one YEARWEEK at a time**, so that each week's posts are:
    - fetched,
    - processed into daily features (attention, greed, greed_ratio),
    - appended to the CSV output,
  before moving on to the next week (memory-safe incremental pipeline).

Focus universe
--------------
- 005930
- 000660
- 005380
- 105560
- 373220
- 080220
- 033100
- 190510
- 064850
- 272110

Output
------
`pipeline_output_attention/attention_greed_panel_daily_focus10.csv`
    Columns:
        - company_code
        - dt
        - attn_volume
        - greed_count
        - greed_ratio

Run
---
From project root:

    python greed/scripts/naver_attention_greed_focus10.py

Requirements
------------
- MySQL access to feeds.dr_post_meta (same as other greed scripts).
- `lexicon_greed_final.txt` present under the greed base directory.
"""

import logging
import re
from pathlib import Path
from typing import List

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ======================================================
# CONFIG
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "pipeline_output_attention"
PANEL_OUT = OUTPUT_DIR / "attention_greed_panel_daily_focus10.csv"
LEXICON_PATH = BASE_DIR / "lexicon_greed_final.txt"

MYSQL_HOST = "fngo-ml-rds-cluster-8-cluster.cluster-ro-c6btgg8fszdb.ap-northeast-2.rds.amazonaws.com"
MYSQL_USER = "fngoMLAdmin"
MYSQL_PASSWORD = "fngo_2020-for!Knowledge"
MYSQL_PORT = 3306
MYSQL_DEFAULT_DB = "news"
FULL_TABLE = "feeds.dr_post_meta"

# Focus universe (zero-padded 6-digit company codes)
FOCUS_CODES: List[str] = [
    "005930",
    "000660",
    "005380",
    "105560",
    "373220",
    "080220",
    "033100",
    "190510",
    "064850",
    "272110",
]

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
# DB HELPERS
# ======================================================

def make_engine():
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DEFAULT_DB}?charset=utf8mb4"
    )
    return create_engine(url)


def list_week_buckets(engine) -> list[int]:
    """
    Return list of YEARWEEK values (e.g. 202401) present in the focus universe.

    No date filter: uses full history for the specified company_code set.
    """
    logging.info("[1] Querying available weeks (YEARWEEK) for focus universe")
    codes_sql = ",".join(f"'{c}'" for c in FOCUS_CODES)
    sql = text(
        f"""
        SELECT DISTINCT YEARWEEK(published_at) AS yw
        FROM {FULL_TABLE}
        WHERE company_code IN ({codes_sql})
          AND company_code IS NOT NULL
        ORDER BY yw ASC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    buckets = [r[0] for r in rows if r and r[0] is not None]
    return buckets


def fetch_week_posts(engine, bucket: int) -> pd.DataFrame:
    """
    Fetch one YEARWEEK of posts for the focus universe.
    Returns DataFrame with columns: dt, company_code, title.
    """
    codes_sql = ",".join(f"'{c}'" for c in FOCUS_CODES)
    sql = text(
        f"""
        SELECT
            published_at AS dt,
            company_code,
            {TITLE_COL}
        FROM {FULL_TABLE}
        WHERE company_code IN ({codes_sql})
          AND company_code IS NOT NULL
          AND YEARWEEK(published_at) = :bucket
        """
    )
    df = pd.read_sql(sql, engine, params={"bucket": int(bucket)}, parse_dates=["dt"])
    if df.empty:
        return df

    df["company_code"] = df["company_code"].astype(str).str.zfill(6)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["dt", "company_code"]).copy()
    df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)
    return df


# ======================================================
# BUILD DAILY PANEL (ATTENTION + GREED + GREED RATIO)
# ======================================================

def build_attention_greed_panel(
    df_posts: pd.DataFrame,
    norm_phrases: list[str],
) -> pd.DataFrame:
    """
    Returns daily panel: company_code x dt with attn_volume, greed_count, greed_ratio.
    """
    if df_posts.empty:
        return pd.DataFrame(columns=["company_code", "dt", "attn_volume", "greed_count", "greed_ratio"])

    logging.info("[2] Normalising titles and flagging greed for current batch")
    df = df_posts.copy()
    df["_title_norm"] = df[TITLE_COL].map(normalise_title)
    df["_greed"] = df["_title_norm"].map(
        lambda t: 1 if title_contains_greed(t, norm_phrases) else 0
    )

    logging.info("[3] Aggregating by (company_code, dt) for current batch")
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

    # greed_ratio = greed_count / attn_volume; 0 if either is 0
    panel["greed_ratio"] = 0.0
    mask = (panel["attn_volume"] > 0) & (panel["greed_count"] > 0)
    panel.loc[mask, "greed_ratio"] = (
        panel.loc[mask, "greed_count"] / panel.loc[mask, "attn_volume"]
    )

    panel = panel.sort_values(["company_code", "dt"]).reset_index(drop=True)

    logging.info(
        f"[PANEL-BATCH] rows={len(panel):,} "
        f"stocks={panel['company_code'].nunique():,} "
        f"dt_min={panel['dt'].min().date()} dt_max={panel['dt'].max().date()}"
    )
    return panel


# ======================================================
# MAIN (WEEKLY INCREMENTAL PIPELINE)
# ======================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    phrases = load_greed_phrases(LEXICON_PATH)
    norm_phrases = build_normalised_phrases(phrases)
    logging.info(f"[LEXICON] loaded {len(phrases)} phrases from {LEXICON_PATH.name}")

    engine = make_engine()
    buckets = list_week_buckets(engine)
    if not buckets:
        logging.warning("No YEARWEEK buckets found for focus universe; nothing to do.")
        return

    n_weeks = len(buckets)
    logging.info(f"[0] Total weeks to process: {n_weeks} (YEARWEEK {buckets[0]} .. {buckets[-1]})")

    first_write = True
    total_posts = 0
    total_panel_rows = 0

    for i, bucket in enumerate(buckets):
        pct = 100 * (i + 1) / n_weeks
        logging.info(f"[1] [{i+1}/{n_weeks}] ({pct:.0f}%) YEARWEEK={bucket} — fetching posts...")

        df_week = fetch_week_posts(engine, bucket)
        if df_week.empty:
            logging.info(f"[1] YEARWEEK={bucket}: no rows for focus universe, skipping")
            continue

        n_week_rows = len(df_week)
        n_week_stocks = df_week["company_code"].nunique()
        total_posts += n_week_rows
        logging.info(
            f"[1] YEARWEEK={bucket}: fetched {n_week_rows:,} posts, "
            f"{n_week_stocks} stocks (cumulative posts={total_posts:,})"
        )

        # Build panel for this week and append to CSV
        panel_week = build_attention_greed_panel(df_week, norm_phrases)
        if panel_week.empty:
            logging.info(f"[2] YEARWEEK={bucket}: no panel rows after processing, skipping save")
            continue

        total_panel_rows += len(panel_week)
        d_min = panel_week["dt"].min().date()
        d_max = panel_week["dt"].max().date()

        panel_week.to_csv(
            PANEL_OUT,
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )
        first_write = False

        logging.info(
            f"[2] YEARWEEK={bucket}: saved {len(panel_week):,} panel rows "
            f"({d_min} ~ {d_max}) | cumulative panel_rows={total_panel_rows:,}"
        )

    if first_write:
        logging.warning("No data written; all weeks were empty or filtered out.")
    else:
        logging.info(f"[SAVED] {PANEL_OUT}")
        logging.info(
            f"Done. total_posts={total_posts:,}  "
            f"total_panel_rows={total_panel_rows:,}"
        )


if __name__ == "__main__":
    main()

