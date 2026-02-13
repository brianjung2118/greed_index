# Greed — Quarterly peaks & greed ratio

Streamlit app: view quarterly price peaks for focus stocks and compare price vs. greed ratio in a two-week window around each peak.

## Run locally

```bash
cd greed
pip install -r requirements.txt
streamlit run apps/streamlit_quarterly_peaks.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. **New app** → choose this repo.
4. **Main file path:** `apps/streamlit_quarterly_peaks.py`
5. Deploy (use the same branch and root as your repo).

## Data

- **Prices:** `prices/{code}.csv` (date, close) for the 10 focus stocks.
- **Greed panel:** `pipeline_output_attention/attention_greed_panel_daily_focus10.csv`

Ensure these files exist in the repo (or in the same paths when running).
