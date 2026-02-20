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
# Paths are relative to this script: BASE_DIR = repo root (folder containing apps/).
# Works locally and on Streamlit Cloud — no hardcoded machine paths.

BASE_DIR = Path(__file__).resolve().parent.parent
PRICES_DIR = BASE_DIR / "prices"
FUNDAMENTALS_DIR = BASE_DIR / "fundamentals"
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


@st.cache_data(show_spinner=False)
def load_fundamentals_csv(code: str) -> pd.DataFrame | None:
    """Load fundamentals CSV; return DataFrame with date and 'income' (operating_income or net_income)."""
    path = FUNDAMENTALS_DIR / f"{code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    # Use operating_income or net_income, whichever exists (prefer operating_income)
    if "operating_income" in df.columns:
        income_col = "operating_income"
    elif "net_income" in df.columns:
        income_col = "net_income"
    else:
        return None
    df["income"] = pd.to_numeric(df[income_col], errors="coerce")
    return df[["date", "income"]].sort_values("date").reset_index(drop=True)


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
    price_full = price_df.copy()
    price_df = price_df.loc[price_df["date"].notna() & (price_df["date"] >= cutoff)].reset_index(drop=True)
    if price_df.empty:
        st.warning(f"No price data in the last {YEARS_LOOKBACK} years for {code}.")
        return

    greed_df = get_greed_for_stock(panel, code)
    full_merged = merge_price_and_greed(price_full, greed_df)
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

    # -------------------------------------------------------------------------
    # Full history: price vs greed ratio overlay (normalized 0–1 for co-movement)
    # -------------------------------------------------------------------------
    st.subheader("Price vs greed ratio — full history (normalized)")
    st.caption("All dates where both price and greed ratio exist. Both series scaled to 0–1 to compare co-movements.")

    greed_roll_options_overlay = {
        "Daily (no rolling)": 1,
        "1 week (7 days)": 7,
        "1 month (30 days)": 30,
        "Quarterly (90 days)": 90,
    }
    greed_roll_label_overlay = st.selectbox(
        "Greed ratio rolling window (for chart below)",
        options=list(greed_roll_options_overlay.keys()),
        index=1,
        key="greed_roll_overlay",
    )
    greed_roll_days_overlay = greed_roll_options_overlay[greed_roll_label_overlay]

    ts_df = full_merged.copy()
    ts_df["price"] = pd.to_numeric(ts_df["price"], errors="coerce")
    ts_df["greed_ratio"] = pd.to_numeric(ts_df["greed_ratio"], errors="coerce")
    ts_df = ts_df.dropna(subset=["date", "price", "greed_ratio"]).sort_values("date").reset_index(drop=True)
    if not ts_df.empty:
        ts_df["greed_ratio_rolled"] = (
            ts_df["greed_ratio"].rolling(window=greed_roll_days_overlay, min_periods=1).mean()
        )
    p_min = ts_df["price"].min() if not ts_df.empty else 0.0
    p_max = ts_df["price"].max() if not ts_df.empty else 1.0
    g_min = ts_df["greed_ratio_rolled"].min() if not ts_df.empty and "greed_ratio_rolled" in ts_df.columns else 0.0
    g_max = ts_df["greed_ratio_rolled"].max() if not ts_df.empty and "greed_ratio_rolled" in ts_df.columns else 1.0
    p_range = (p_max - p_min) if (p_max - p_min) > 0 else 1.0
    g_range = (g_max - g_min) if (g_max - g_min) > 0 else 1.0
    ts_df["price_norm"] = (ts_df["price"] - p_min) / p_range
    ts_df["greed_norm"] = (ts_df["greed_ratio_rolled"] - g_min) / g_range
    ts_df["date_str"] = ts_df["date"].dt.strftime("%Y-%m-%d")

    if ts_df.empty or (ts_df["price_norm"].notna().sum() == 0 and ts_df["greed_norm"].notna().sum() == 0):
        st.info("No overlapping dates with both price and greed ratio.")
    else:
        price_ts = alt.Chart(ts_df).mark_line(stroke="steelblue", strokeWidth=2).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("price_norm:Q", title="Normalized (0–1)", scale=alt.Scale(domain=[0, 1])),
            tooltip=["date_str:N", "price:Q", "greed_ratio_rolled:Q"],
        )
        greed_ts = alt.Chart(ts_df).mark_line(stroke="green", strokeWidth=2, strokeDash=[4, 2]).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("greed_norm:Q", title="Normalized (0–1)", scale=alt.Scale(domain=[0, 1])),
            tooltip=["date_str:N", "price:Q", "greed_ratio_rolled:Q"],
        )
        overlay = (price_ts + greed_ts).properties(
            height=350,
            title=f"Price (blue) vs greed ratio (green, {greed_roll_label_overlay}) — {code} — full history",
        )
        st.altair_chart(overlay, use_container_width=True)

    # -------------------------------------------------------------------------
    # Chart 2: Greed ratio & income (operating_income or net_income) time series
    # -------------------------------------------------------------------------
    st.subheader("Greed ratio & income (time series)")
    st.caption("Only dates where both greed ratio and income exist. Both series normalized to 0–1 so movements are comparable.")

    fund_df = load_fundamentals_csv(code)
    if fund_df is None or fund_df.empty:
        st.info("No fundamentals data for this stock. Add a CSV in `fundamentals/{code}.csv` with date, close, and operating_income or net_income.")
    else:
        # Rolling window for greed ratio (dropdown)
        greed_roll_options = {
            "Daily (no rolling)": 1,
            "1 week (7 days)": 7,
            "1 month (30 days)": 30,
            "Quarterly (90 days)": 90,
        }
        greed_roll_label = st.selectbox(
            "Greed ratio rolling window",
            options=list(greed_roll_options.keys()),
            index=1,
        )
        greed_roll_days = greed_roll_options[greed_roll_label]

        # Same 5-year window as the rest of the page
        fund_df = fund_df[(fund_df["date"] >= cutoff) & (fund_df["date"] <= max_date)].copy()
        greed_ts = get_greed_for_stock(panel, code)
        greed_ts = greed_ts.rename(columns={"dt": "date"})
        greed_ts["greed_ratio"] = pd.to_numeric(greed_ts["greed_ratio"], errors="coerce")
        merge_gi = fund_df.merge(greed_ts, on="date", how="inner")
        merge_gi = merge_gi.dropna(subset=["greed_ratio", "income"])
        merge_gi = merge_gi.sort_values("date").reset_index(drop=True)
        if merge_gi.empty:
            st.info("No overlapping dates with both greed ratio and income in the last 5 years.")
        else:
            # Apply rolling mean to greed ratio (min_periods=1 keeps early points, smooths the rest)
            merge_gi["greed_ratio_rolled"] = (
                merge_gi["greed_ratio"].rolling(window=greed_roll_days, min_periods=1).mean()
            )
            merge_gi["date_str"] = merge_gi["date"].dt.strftime("%Y-%m-%d")
            # Normalize both to 0–1 (use rolled greed for the green line)
            g_min, g_max = merge_gi["greed_ratio_rolled"].min(), merge_gi["greed_ratio_rolled"].max()
            g_range = g_max - g_min
            merge_gi["greed_norm"] = (merge_gi["greed_ratio_rolled"] - g_min) / g_range if g_range > 0 else 0.5
            i_min, i_max = merge_gi["income"].min(), merge_gi["income"].max()
            i_range = i_max - i_min
            merge_gi["income_norm"] = (merge_gi["income"] - i_min) / i_range if i_range > 0 else 0.5

            x_axis = alt.Axis(title="Date", format="%b %Y")
            greed_ts_line = alt.Chart(merge_gi).mark_line(stroke="green", strokeWidth=2, strokeDash=[4, 2]).encode(
                x=alt.X("date:T", axis=x_axis),
                y=alt.Y("greed_norm:Q", title="Normalized (0–1)", scale=alt.Scale(domain=[0, 1])),
                tooltip=["date_str:N", "greed_ratio_rolled:Q", "income:Q"],
            )
            income_line = alt.Chart(merge_gi).mark_line(stroke="steelblue", strokeWidth=2).encode(
                x=alt.X("date:T", axis=x_axis),
                y=alt.Y("income_norm:Q", title="Normalized (0–1)", scale=alt.Scale(domain=[0, 1])),
                tooltip=["date_str:N", "greed_ratio_rolled:Q", "income:Q"],
            )
            chart_gi = (greed_ts_line + income_line).properties(
                height=350,
                title=f"Greed ratio & income (greed: {greed_roll_label}; green = greed, blue = income)",
            )
            st.altair_chart(chart_gi, use_container_width=True)


if __name__ == "__main__":
    main()
