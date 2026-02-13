from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ======================================================
# CONFIG
# ======================================================

BASE_DIR = "/Users/sanghoonjung/PycharmProjects/PythonProject/rii/greed"
OUTPUT_DIR = f"{BASE_DIR}/text_samples"
OUTPUT_PATH = f"{OUTPUT_DIR}/greed_text_sample_1000.csv"

N_SAMPLES = 1000
MIN_POST_DT = "2020-01-01"

MYSQL_HOST = "fngo-ml-rds-cluster-8-cluster.cluster-ro-c6btgg8fszdb.ap-northeast-2.rds.amazonaws.com"
MYSQL_USER = "fngoMLAdmin"
MYSQL_PASSWORD = "fngo_2020-for!Knowledge"
MYSQL_PORT = 3306
MYSQL_DEFAULT_DB = "news"  # for initial connection

SOURCE_DB = "feeds"
SOURCE_TABLE = "dr_post_meta"
FULL_TABLE = f"{SOURCE_DB}.{SOURCE_TABLE}"


def make_engine():
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DEFAULT_DB}?charset=utf8mb4"
    )
    return create_engine(url)


def get_table_columns(engine) -> List[str]:
    q = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :db
          AND table_name = :tbl
        ORDER BY ordinal_position
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"db": SOURCE_DB, "tbl": SOURCE_TABLE}).fetchall()
    return [r[0] for r in rows]


def load_random_posts(n: int) -> pd.DataFrame:
    """Randomly sample n rows with all columns."""

    logging.info(f"[1] Sampling {n} random discussion posts from {FULL_TABLE}")

    engine = make_engine()

    cols = get_table_columns(engine)
    if not cols:
        raise RuntimeError(f"No columns found for {FULL_TABLE}. Check DB/table name and permissions.")

    # We sample ALL columns. This table is meta-only in your schema.
    # We still filter by published_at (known to exist) for recency control.
    sql = text(
        f"""
        SELECT
            *
        FROM {FULL_TABLE}
        WHERE published_at >= :min_dt
        ORDER BY RAND()
        LIMIT {n}
        """
    )

    df = pd.read_sql(sql, engine, params={"min_dt": MIN_POST_DT}, parse_dates=["published_at", "crawled_at"])

    if df.empty:
        logging.warning("No rows returned. Check MIN_POST_DT.")
        return df

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    if "company_code" in df.columns:
        df["company_code"] = df["company_code"].astype(str).str.zfill(6)

    logging.info(f"[SAMPLED] rows={len(df):,}")
    if "published_at" in df.columns:
        logging.info(f"[DATE RANGE] min={df['published_at'].min()} max={df['published_at'].max()}")

    return df


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    df = load_random_posts(N_SAMPLES)

    if df.empty:
        logging.warning("No data saved.")
        return

    # utf-8-sig helps Excel display Korean correctly
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    logging.info(f"[SAVED] {OUTPUT_PATH}")
    logging.info("Done.")


if __name__ == "__main__":
    main()