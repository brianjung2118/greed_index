# streamlit_viewer.py
# ------------------------------------------------------
# Streamlit app to visualise attention and greed counts
# from attention_greed_panel_daily.csv
#
# Run:
#   streamlit run greed/apps/streamlit_viewer.py
#
# Expected file:
#   greed/pipeline_output_attention/attention_greed_panel_daily_with_ratio.csv
#   (Run naver_attention_greed.py then add_greed_ratio.py to generate.)
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
DATA_PATH = BASE_DIR / "pipeline_output_attention" / "attention_greed_panel_daily_with_ratio.csv"

# =========================
# HELPERS
# =========================

@st.cache_data(show_spinner=False)
def load_attention_greed_panel(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find attention_greed_panel_daily_with_ratio.csv at:\n{csv_path}\n"
            "Run naver_attention_greed.py then add_greed_ratio.py to generate the data."
        )

    df = pd.read_csv(csv_path, parse_dates=["dt"], low_memory=False)

    required = {"company_code", "dt", "attn_volume", "greed_count", "greed_ratio"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if "returns" not in df.columns:
        df["returns"] = float("nan")

    df["company_code"] = df["company_code"].astype(str).str.zfill(6)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.dropna(subset=["dt", "company_code"]).copy()

    return df


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Attention & Greed Viewer", layout="wide")
st.title("📈 Naver Discussion — Attention & Greed Time Series")
st.caption("Select a stock to view daily attention counts, greed counts, and greed ratio.")

# Load data
try:
    df = load_attention_greed_panel(DATA_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")

all_codes = sorted(df["company_code"].unique().tolist())
default_code = all_codes[0] if all_codes else None

code = st.sidebar.selectbox("Company code", options=all_codes, index=0 if default_code else 0)

# Filter stock
stock = df.loc[df["company_code"] == code].sort_values("dt").copy()
stock_all = stock.copy()

if stock.empty:
    st.warning("No rows for that stock code.")
    st.stop()

min_dt = stock["dt"].min().date()
max_dt = stock["dt"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_dt, max_dt),
    min_value=min_dt,
    max_value=max_dt,
)

# Normalise date_range return types (Streamlit sometimes returns single date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_dt, max_dt

mask = (stock["dt"].dt.date >= start_date) & (stock["dt"].dt.date <= end_date)
stock = stock.loc[mask].copy()

# Chart 1: Raw attention counts
st.subheader(f"📊 {code} — Raw Attention Counts (post volume)")
attn_series = stock[["dt", "attn_volume"]].dropna().copy()
if not attn_series.empty:
    attn_chart = (
        alt.Chart(attn_series)
        .mark_line(color="#4CC9F0", strokeWidth=2)
        .encode(
            x=alt.X("dt:T", title="Date"),
            y=alt.Y("attn_volume:Q", title="Attention volume (posts)"),
            tooltip=[
                alt.Tooltip("dt:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("attn_volume:Q", title="Attention volume"),
            ],
        )
        .interactive()
        .properties(height=350)
    )
    st.altair_chart(attn_chart, use_container_width=True)
else:
    st.info("No attention data in this date range.")

st.divider()

# Chart 2: Raw greed counts
st.subheader(f"🟠 {code} — Raw Greed Counts (posts with greed phrases)")
greed_series = stock[["dt", "greed_count"]].dropna().copy()
if not greed_series.empty:
    greed_chart = (
        alt.Chart(greed_series)
        .mark_line(color="#F72585", strokeWidth=2)
        .encode(
            x=alt.X("dt:T", title="Date"),
            y=alt.Y("greed_count:Q", title="Greed count"),
            tooltip=[
                alt.Tooltip("dt:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("greed_count:Q", title="Greed count"),
            ],
        )
        .interactive()
        .properties(height=350)
    )
    st.altair_chart(greed_chart, use_container_width=True)
else:
    st.info("No greed data in this date range.")

st.divider()

# Chart 3: Greed ratio (optional overlay: daily returns)
st.subheader(f"📉 {code} — Greed Ratio (greed_count / attn_volume)")
ratio_series = stock[["dt", "greed_ratio", "returns"]].copy()
ratio_series["returns"] = pd.to_numeric(ratio_series["returns"], errors="coerce")
# Weekends / non-trading days have no return; use 0 so the overlay line has no gaps
ratio_series["returns"] = ratio_series["returns"].fillna(0)

overlay_returns = st.checkbox("Overlay daily returns", value=False, key="overlay_returns")

# Return shift: base "returns" is t→t+1; positive shift = past, negative = future
RETURN_SHIFT_OPTIONS = {
    "t-3": 3,
    "t-2": 2,
    "t-1": 1,
    "t+1": 0,
    "t+2": -1,
    "t+3": -2,
    "t+4": -3,
}
return_shift_label = st.selectbox(
    "Returns lag/lead (when overlay on)",
    options=list(RETURN_SHIFT_OPTIONS.keys()),
    index=3,  # default t+1
    key="return_shift",
)
return_shift_periods = RETURN_SHIFT_OPTIONS[return_shift_label]

if not ratio_series.empty:
    if overlay_returns:
        # Apply time shift: shift(1) = at t show return from t-1; shift(-1) = at t show return from t+1
        returns_shifted = ratio_series["returns"].shift(return_shift_periods).fillna(0)
        ratio_plot = ratio_series[["dt", "greed_ratio"]].copy()
        ratio_plot["returns_shifted"] = returns_shifted

        legend_domain = ["Greed ratio", f"Daily return ({return_shift_label})"]
        legend_range = ["#9B5DE5", "#00F5D4"]
        color_enc = alt.Color(
            "series:N",
            scale=alt.Scale(domain=legend_domain, range=legend_range),
            legend=alt.Legend(title=None),
        )
        df_ratio = ratio_plot[["dt", "greed_ratio"]].copy()
        df_ratio["series"] = "Greed ratio"
        df_returns = ratio_plot[["dt", "returns_shifted"]].copy()
        df_returns = df_returns.rename(columns={"returns_shifted": "returns"})
        df_returns["series"] = f"Daily return ({return_shift_label})"
        line_ratio = (
            alt.Chart(df_ratio)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("dt:T", title="Date"),
                y=alt.Y("greed_ratio:Q", title="Greed ratio", scale=alt.Scale(zero=False)),
                color=color_enc,
                tooltip=[
                    alt.Tooltip("dt:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip("greed_ratio:Q", title="Greed ratio", format=".4f"),
                ],
            )
        )
        line_returns = (
            alt.Chart(df_returns)
            .mark_line(strokeWidth=1.5, opacity=0.9)
            .encode(
                x=alt.X("dt:T", title="Date"),
                y=alt.Y("returns:Q", title=f"Daily return ({return_shift_label})", scale=alt.Scale(zero=True)),
                color=color_enc,
                tooltip=[
                    alt.Tooltip("dt:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip("returns:Q", title=f"Return ({return_shift_label})", format=".4f"),
                ],
            )
        )
        ratio_chart = (
            alt.layer(line_ratio, line_returns)
            .resolve_scale(y="independent")
            .interactive()
            .properties(height=350)
        )
    else:
        ratio_chart = (
            alt.Chart(ratio_series.dropna(subset=["greed_ratio"]))
            .mark_line(color="#9B5DE5", strokeWidth=2)
            .encode(
                x=alt.X("dt:T", title="Date"),
                y=alt.Y("greed_ratio:Q", title="Greed ratio"),
                tooltip=[
                    alt.Tooltip("dt:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip("greed_ratio:Q", title="Greed ratio", format=".4f"),
                ],
            )
            .interactive()
            .properties(height=350)
        )
    st.altair_chart(ratio_chart, use_container_width=True)
else:
    st.info("No greed ratio data in this date range.")

# Sidebar: quick stats
st.sidebar.divider()
st.sidebar.subheader("Quick stats")
latest_row = stock_all.dropna(subset=["dt"]).sort_values("dt").tail(1)
if latest_row.empty:
    st.sidebar.info("No data available.")
else:
    lr = latest_row.iloc[0]
    st.sidebar.write(f"**Latest** ({lr['dt'].date()})")
    st.sidebar.metric("attn_volume", f"{int(lr['attn_volume']):,}")
    st.sidebar.metric("greed_count", f"{int(lr['greed_count']):,}")
    st.sidebar.metric("greed_ratio", f"{float(lr['greed_ratio']):.4f}")
    if "returns" in lr and pd.notna(lr.get("returns")):
        st.sidebar.metric("returns (t+1)", f"{float(lr['returns']):.4f}")

st.sidebar.metric("Rows in range", f"{len(stock):,}")
st.sidebar.metric("Date range", f"{start_date} ~ {end_date}")

# Data table expander
with st.expander("Show data table"):
    cols = ["dt", "company_code", "attn_volume", "greed_count", "greed_ratio"]
    if "returns" in stock.columns:
        cols.append("returns")
    st.dataframe(
        stock[cols].sort_values("dt", ascending=False),
        use_container_width=True,
        height=350,
    )

st.divider()
st.caption(f"Data source: {DATA_PATH}")