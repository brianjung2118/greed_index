# discover_greed_terms_from_100k.py
# ------------------------------------------------------------
# Pull ~100k Naver discussion META rows (title + engagement),
# extract candidate unigrams/bigrams/trigrams, and rank them by
# over-representation in high-engagement titles.
#
# Output:
#   {BASE_DIR}/lexicon_discovery_100k/
#     - candidates_unigram.csv
#     - candidates_bigram.csv
#     - candidates_trigram.csv
#     - kwic_top_candidates.csv
#
# Run:
#   python discover_greed_terms_from_100k.py
# ------------------------------------------------------------

from __future__ import annotations

import re
import math
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Iterable, Dict

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(message)s")

# =========================
# CONFIG
# =========================

BASE_DIR = "/Users/sanghoonjung/PycharmProjects/PythonProject/rii/greed"
OUTPUT_DIR = f"{BASE_DIR}/lexicon_discovery_100k"

# Pull settings
TARGET_ROWS = 100_000
STRATIFY_BY = "month"  # "month" or "year"
MAX_BUCKETS = None      # set e.g. 60 to cap to recent 60 months; None = use all buckets since MIN_DT
MIN_DT = "2020-01-01"     # inclusive

# High-engagement definition
HIGH_ENG_PCT = 0.10       # top 10% by engagement score

# How many candidates to keep in outputs
TOP_K_UNI = 800
TOP_K_BI = 800
TOP_K_TRI = 800

# KWIC sampling from top candidates
KWIC_TOP_N = 60           # take top N candidates (from each n-gram level)
KWIC_PER_CAND = 10        # sample titles per candidate

# Database config
MYSQL_HOST = "fngo-ml-rds-cluster-8-cluster.cluster-ro-c6btgg8fszdb.ap-northeast-2.rds.amazonaws.com"
MYSQL_USER = "fngoMLAdmin"
MYSQL_PASSWORD = "fngo_2020-for!Knowledge"
MYSQL_PORT = 3306
MYSQL_DEFAULT_DB = "news"      # for connection
FULL_TABLE = "feeds.dr_post_meta"

# Columns present in your schema (based on your earlier preview)
DT_COL = "published_at"
TITLE_COL = "title"
VIEWS_COL = "views"
COMMENTS_COL = "comment"
CODE_COL = "company_code"


# =========================
# DB
# =========================

def make_engine():
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DEFAULT_DB}?charset=utf8mb4"
    )
    return create_engine(url)



def list_time_buckets(engine) -> List[str]:
    """Return list of time buckets (YYYY-MM or YYYY) present in the table since MIN_DT."""

    if STRATIFY_BY == "year":
        bucket_expr = f"DATE_FORMAT({DT_COL}, '%Y')"
    else:
        bucket_expr = f"DATE_FORMAT({DT_COL}, '%Y-%m')"

    sql = text(
        f"""
        SELECT DISTINCT {bucket_expr} AS bucket
        FROM {FULL_TABLE}
        WHERE {DT_COL} >= :min_dt
          AND {TITLE_COL} IS NOT NULL
        ORDER BY bucket ASC
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(sql, {"min_dt": MIN_DT}).fetchall()

    buckets = [r[0] for r in rows if r and r[0] is not None]

    if MAX_BUCKETS is not None and len(buckets) > MAX_BUCKETS:
        # keep most recent buckets
        buckets = buckets[-MAX_BUCKETS:]

    return buckets


def load_titles_stratified(target_rows: int) -> pd.DataFrame:
    """Loads approximately target_rows rows by sampling evenly across time buckets.

    Strategy:
      - Find available buckets (month or year)
      - Pull `per_bucket` rows per bucket ordered by published_at DESC
      - Concatenate

    This avoids ORDER BY RAND() and avoids 'all recent' bias.
    """

    engine = make_engine()

    buckets = list_time_buckets(engine)
    if not buckets:
        raise RuntimeError("No time buckets found. Check MIN_DT/table/permissions.")

    per_bucket = max(1, int(math.ceil(target_rows / len(buckets))))

    logging.info(
        f"[1] Stratified load: target_rows={target_rows:,}  buckets={len(buckets):,}  per_bucket={per_bucket:,}  by={STRATIFY_BY}"
    )

    # Build bucket filter expressions
    if STRATIFY_BY == "year":
        bucket_filter = f"DATE_FORMAT({DT_COL}, '%Y') = :bucket"
    else:
        bucket_filter = f"DATE_FORMAT({DT_COL}, '%Y-%m') = :bucket"

    sql = text(
        f"""
        SELECT
            {DT_COL},
            {CODE_COL},
            {TITLE_COL},
            {VIEWS_COL},
            {COMMENTS_COL}
        FROM {FULL_TABLE}
        WHERE {DT_COL} >= :min_dt
          AND {TITLE_COL} IS NOT NULL
          AND {bucket_filter}
        ORDER BY {DT_COL} DESC
        LIMIT {per_bucket}
        """
    )

    frames = []
    for b in buckets:
        df_b = pd.read_sql(
            sql,
            engine,
            params={"min_dt": MIN_DT, "bucket": b},
            parse_dates=[DT_COL],
        )
        if df_b.empty:
            continue
        df_b["_bucket"] = b
        frames.append(df_b)

    if not frames:
        raise RuntimeError("No rows returned from stratified queries. Check data availability.")

    df = pd.concat(frames, ignore_index=True)

    # light cleanup
    df[TITLE_COL] = df[TITLE_COL].astype(str)
    df[VIEWS_COL] = pd.to_numeric(df[VIEWS_COL], errors="coerce").fillna(0)
    df[COMMENTS_COL] = pd.to_numeric(df[COMMENTS_COL], errors="coerce").fillna(0)
    df[CODE_COL] = df[CODE_COL].astype(str).str.zfill(6)

    logging.info(f"[OK] loaded rows={len(df):,}  buckets_used={df['_bucket'].nunique():,}  dates={df[DT_COL].min()} ~ {df[DT_COL].max()}")

    # If we overshot, downsample deterministically by engagement rank (keeps 'important' titles)
    if len(df) > target_rows:
        df["_eng"] = engagement_score(df)
        df = df.sort_values("_eng", ascending=False).head(target_rows).reset_index(drop=True)

    return df


# =========================
# TEXT
# =========================

def normalise_title(s: str) -> str:
    """
    Minimal normalisation. Keep Korean/English/nums.
    Keep 'ㅋㅋ', 'ㅎㅎ' etc OUT by default because they are tone markers,
    but this normaliser will strip them since they’re non-alnum.
    If you want to keep them later, we can tweak.
    """
    s = str(s).strip().lower()
    s = re.sub(r"[/|·•]", " ", s)
    # keep korean/english/nums/space
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenise(s: str) -> List[str]:
    """
    Very simple whitespace tokeniser.
    We intentionally keep short titles mostly intact.
    """
    toks = s.split()
    # keep tokens length >= 2 to reduce pure junk like 'ㅋ'
    return [t for t in toks if len(t) >= 2]


def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i+n])


def engagement_score(df: pd.DataFrame) -> pd.Series:
    # log1p views + log1p comments
    return (df[VIEWS_COL].add(1).apply(math.log) + df[COMMENTS_COL].add(1).apply(math.log))


# =========================
# NGRAM STATS
# =========================

def count_ngrams(token_lists: List[List[str]], n: int) -> Counter:
    c = Counter()
    for toks in token_lists:
        for ng in ngrams(toks, n):
            c[ng] += 1
    return c


def build_candidates(df: pd.DataFrame, n: int, top_k: int) -> pd.DataFrame:
    """
    Build n-gram candidates ranked by lift into high-engagement titles.
    """
    df = df.copy()
    df["_title_norm"] = df[TITLE_COL].map(normalise_title)
    df["_tokens"] = df["_title_norm"].map(tokenise)

    df["_eng"] = engagement_score(df)
    cutoff = df["_eng"].quantile(1 - HIGH_ENG_PCT)
    df["_is_high"] = (df["_eng"] >= cutoff).astype(int)

    token_all = df["_tokens"].tolist()
    token_high = df.loc[df["_is_high"] == 1, "_tokens"].tolist()

    c_all = count_ngrams(token_all, n)
    c_high = count_ngrams(token_high, n)

    total_all = sum(c_all.values()) or 1
    total_high = sum(c_high.values()) or 1

    rows = []
    for ng, cnt in c_all.items():
        high_cnt = c_high.get(ng, 0)
        p_all = cnt / total_all
        p_high = high_cnt / total_high
        lift = (p_high / p_all) if p_all > 0 else 0.0

        # skip ultra-rare junk
        if cnt < 3:
            continue

        rows.append({
            "ngram": " ".join(ng),
            "count_all": cnt,
            "count_high": high_cnt,
            "share_all": p_all,
            "share_high": p_high,
            "lift_high_vs_all": lift,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # rank by lift then by count_high then count_all
    out = out.sort_values(
        ["lift_high_vs_all", "count_high", "count_all"],
        ascending=[False, False, False]
    ).head(top_k).reset_index(drop=True)

    return out


def kwic_for_candidates(df: pd.DataFrame, candidates: List[str], per_cand: int) -> pd.DataFrame:
    """
    For each candidate phrase, show sample titles containing it.
    We sort by engagement so you see 'loud' examples first.
    """
    tmp = df.copy()
    tmp["_eng"] = engagement_score(tmp)
    tmp["_title_norm"] = tmp[TITLE_COL].map(normalise_title)

    frames = []
    for phrase in candidates:
        pat = re.escape(phrase)
        hits = tmp[tmp["_title_norm"].str.contains(pat, na=False)].copy()
        if hits.empty:
            continue
        hits = hits.sort_values("_eng", ascending=False).head(per_cand)

        frames.append(pd.DataFrame({
            "candidate": phrase,
            "published_at": hits[DT_COL].values,
            "company_code": hits[CODE_COL].values,
            "views": hits[VIEWS_COL].values,
            "comment": hits[COMMENTS_COL].values,
            "title": hits[TITLE_COL].values,
        }))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================
# MAIN
# =========================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    df = load_titles_stratified(TARGET_ROWS)

    logging.info("[2] Building unigram candidates")
    uni = build_candidates(df, n=1, top_k=TOP_K_UNI)
    uni.to_csv(f"{OUTPUT_DIR}/candidates_unigram.csv", index=False, encoding="utf-8-sig")

    logging.info("[3] Building bigram candidates")
    bi = build_candidates(df, n=2, top_k=TOP_K_BI)
    bi.to_csv(f"{OUTPUT_DIR}/candidates_bigram.csv", index=False, encoding="utf-8-sig")

    logging.info("[4] Building trigram candidates")
    tri = build_candidates(df, n=3, top_k=TOP_K_TRI)
    tri.to_csv(f"{OUTPUT_DIR}/candidates_trigram.csv", index=False, encoding="utf-8-sig")

    logging.info("[5] Building KWIC samples for top candidates (quick inspection)")
    top_uni = uni["ngram"].head(KWIC_TOP_N).tolist() if not uni.empty else []
    top_bi = bi["ngram"].head(KWIC_TOP_N).tolist() if not bi.empty else []
    top_tri = tri["ngram"].head(KWIC_TOP_N).tolist() if not tri.empty else []

    kwic_df = kwic_for_candidates(df, candidates=(top_uni + top_bi + top_tri), per_cand=KWIC_PER_CAND)
    kwic_df.to_csv(f"{OUTPUT_DIR}/kwic_top_candidates.csv", index=False, encoding="utf-8-sig")

    logging.info("Done.")
    logging.info(f"Saved to: {OUTPUT_DIR}")
    logging.info("What to do next:")
    logging.info("  1) Open candidates_bigram.csv + candidates_trigram.csv (sort by lift_high_vs_all).")
    logging.info("  2) Use kwic_top_candidates.csv to confirm meaning + sarcasm + context.")
    logging.info("  3) Add approved terms into your UPSIDE and RISK lexicons.")


if __name__ == "__main__":
    main()