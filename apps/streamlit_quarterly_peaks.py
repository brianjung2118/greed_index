# streamlit_quarterly_peaks.py
# ------------------------------------------------------
# Streamlit app: for a selected focus stock, show quarterly price peaks
# and greed ratio around those peaks (highlighted windows).
#
# Run:
#   streamlit run greed/apps/streamlit_quarterly_peaks.py
#
# Data:
#   greed/prices/{code}.csv — date, close
#   greed/pipeline_output_attention/attention_greed_panel_daily_focus10.csv
# ------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
PRICES_DIR = BASE_DIR / "prices"
FOCUS10_PANEL_PATH = BASE_DIR / "pipeline_output_attention" / "attention_greed_panel_daily_focus10.csv"

FOCUS_STOCKS = [
    "005930", "000660", "005380", "105560", "373220",
    "080220", "033100", "190510", "064850", "272110",
]

DEFAULT_WINDOW_DAYS = 14  # days before and after peak date
YEARS_LOOKBACK = 5  # limit chart and peaks to last N years


# =========================
# DATA LOADING
# =========================

@st.cache_data(show_spinner=False)
def load_price_csv(code: str) -> pd.DataFrame:
    path = PRICES_DIR / f"{code}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")
    df = pd.read_csv(path)
    # Normalize column names (some CSVs may have "Close")
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    # Price: strip commas (e.g. "6,281" -> 6281) then coerce to numeric
    price_series = df["close"].astype(str).str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(price_series, errors="coerce")
    return df[["date", "price"]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_greed_panel() -> pd.DataFrame:
    if not FOCUS10_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Focus10 panel not found: {FOCUS10_PANEL_PATH}\n"
            "Run naver_attention_greed_focus10.py to generate it."
        )
    df = pd.read_csv(FOCUS10_PANEL_PATH, parse_dates=["dt"], low_memory=False)
    df["company_code"] = df["company_code"].astype(str).str.zfill(6)
    df["greed_ratio"] = pd.to_numeric(df["greed_ratio"], errors="coerce")
    return df


def get_greed_for_stock(panel: pd.DataFrame, code: str) -> pd.DataFrame:
    return panel.loc[panel["company_code"] == code, ["dt", "greed_ratio"]].copy()


# =========================
# QUARTERLY PEAKS & WINDOWS
# =========================

def quarterly_peaks_and_windows(
    price_df: pd.DataFrame,
    window_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute peak date and price per quarter, and window start/end per quarter."""
    df = price_df.copy()
    df = df.dropna(subset=["price"])  # only use rows with valid price
    if df.empty:
        empty_peaks = pd.DataFrame(columns=["quarter", "peak_date", "peak_price", "window_start", "window_end"])
        empty_windows = pd.DataFrame(columns=["quarter", "window_start", "window_end", "peak_date", "peak_price"])
        return empty_peaks, empty_windows
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    # Peak per quarter: date and price at max close
    peak_idx = df.groupby("quarter")["price"].idxmax().dropna()
    peaks = df.loc[peak_idx, ["quarter", "date", "price"]].copy()
    peaks = peaks.rename(columns={"date": "peak_date", "price": "peak_price"})

    # Window around each peak
    peaks["window_start"] = peaks["peak_date"] - pd.Timedelta(days=window_days)
    peaks["window_end"] = peaks["peak_date"] + pd.Timedelta(days=window_days)

    # For chart: one row per (quarter, window_start, window_end)
    windows = peaks[["quarter", "window_start", "window_end", "peak_date", "peak_price"]].copy()
    return peaks, windows


def merge_price_and_greed(
    price_df: pd.DataFrame,
    greed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Outer merge on date. Greed has daily (incl. weekends); price has trading days only.
    Rows from greed-only dates will have NaN price; rows from price-only will have NaN greed_ratio."""
    greed_df = greed_df.rename(columns={"dt": "date"})
    greed_df["greed_ratio"] = pd.to_numeric(greed_df["greed_ratio"], errors="coerce")
    merged = price_df.merge(greed_df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


# =========================
# UI
# =========================

def main():
    st.set_page_config(page_title="Quarterly peaks & greed", layout="wide")
    st.title("Quarterly price peaks and greed ratio")

    # Data
    try:
        panel = load_greed_panel()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    code = st.selectbox(
        "Stock (focus 10)",
        options=FOCUS_STOCKS,
        format_func=lambda x: f"{x}",
    )

    try:
        price_df = load_price_csv(code)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # Limit to last N years (use only valid dates)
    if price_df.empty or price_df["date"].isna().all():
        st.warning(f"No valid dates in price file for {code}. Check date format (expected DD/MM/YYYY).")
        return
    max_date = price_df["date"].max()
    cutoff = max_date - pd.Timedelta(days=YEARS_LOOKBACK * 365)
    price_df = price_df.loc[price_df["date"].notna() & (price_df["date"] >= cutoff)].reset_index(drop=True)
    if price_df.empty:
        st.warning(f"No price data in the last {YEARS_LOOKBACK} years for {code}.")
        return

    greed_df = get_greed_for_stock(panel, code)
    window_days = st.slider("Window around peak (days before/after)", 1, 60, DEFAULT_WINDOW_DAYS)

    peaks, windows = quarterly_peaks_and_windows(price_df, window_days)
    merged = merge_price_and_greed(price_df, greed_df)

    # Table: quarters and peaks
    st.subheader("Quarterly peaks")
    display_peaks = peaks[["quarter", "peak_date", "peak_price"]].copy()
    if not display_peaks.empty:
        display_peaks["peak_date"] = pd.to_datetime(display_peaks["peak_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    st.dataframe(display_peaks, use_container_width=True, hide_index=True)

    # Select a quarter: clicking is simulated by choosing a row from a dropdown
    st.subheader("Chart: two weeks before and after peak")
    st.caption("Select a quarter from the table above to see price and greed ratio around that peak date.")

    if peaks.empty:
        st.info("No quarterly peaks in the selected period. Change the stock or window.")
        return

    # Build dropdown options: one per table row (select row = pick this quarter)
    options_labels = [
        f"{row['quarter']} — peak {row['peak_date'].strftime('%Y-%m-%d')} (price {row['peak_price']:,.0f})"
        for _, row in peaks.iterrows()
    ]
    selected_label = st.selectbox(
        "Select quarter to view chart (two weeks before and after peak)",
        options=options_labels,
        index=0,
    )
    selected_idx = options_labels.index(selected_label)
    selected_peak = peaks.iloc[selected_idx]
    window_start = selected_peak["window_start"]
    window_end = selected_peak["window_end"]
    peak_date = selected_peak["peak_date"]

    # Filter merged data to this window only
    chart_df = merged.copy()
    chart_df["price"] = pd.to_numeric(chart_df["price"], errors="coerce")
    chart_df["greed_ratio"] = pd.to_numeric(chart_df["greed_ratio"], errors="coerce")
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df = chart_df[(chart_df["date"] >= window_start) & (chart_df["date"] <= window_end)].copy()
    chart_df["date_str"] = chart_df["date"].dt.strftime("%Y-%m-%d")

    if chart_df.empty:
        st.warning("No data in this window. Try another quarter.")
        return

    price_min = chart_df["price"].min(skipna=True)
    price_max = chart_df["price"].max(skipna=True)
    greed_max = chart_df["greed_ratio"].max(skipna=True)
    if pd.isna(price_min):
        price_min = 0.0
    if pd.isna(price_max):
        price_max = price_min + 1.0
    if pd.isna(greed_max) or greed_max == 0:
        greed_max = 1.0
    price_range = max(price_max - price_min, price_max * 0.01, 1.0)
    price_domain = (price_min - price_range * 0.05, price_max + price_range * 0.05)

    # Single peak date for the vertical rule
    peak_rule_df = pd.DataFrame({"peak_date": [peak_date]})

    # Overlay: price (left y-axis) + greed ratio (right y-axis) + peak rule
    price_chart_df = chart_df.loc[chart_df["price"].notna()].copy()
    price_line = alt.Chart(price_chart_df).mark_line(stroke="steelblue", strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("price:Q", title="Price", scale=alt.Scale(domain=list(price_domain))),
        tooltip=["date_str:N", "price:Q", "greed_ratio:Q"],
    )
    greed_line = alt.Chart(chart_df).mark_line(stroke="green", strokeWidth=2, strokeDash=[4, 2]).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("greed_ratio:Q", title="Greed ratio", scale=alt.Scale(domain=[0, greed_max])),
        tooltip=["date_str:N", "price:Q", "greed_ratio:Q"],
    )
    peak_rule = alt.Chart(peak_rule_df).mark_rule(stroke="red", strokeWidth=2, strokeDash=[4, 2]).encode(
        x=alt.X("peak_date:T"),
    )
    combined = (price_line + greed_line + peak_rule).resolve_scale(y="independent").properties(
        height=350,
        title=f"Price & greed ratio — {selected_peak['quarter']} (peak {peak_date.strftime('%Y-%m-%d')})",
    )
    st.altair_chart(combined, use_container_width=True)


if __name__ == "__main__":
    main()
