"""
dashboard.py — SE3 Price Forecast Dashboard (Module 4)

Streamlit app with four pages:
  📈 Forecast       — next 24h price forecast with confidence band
  📊 Performance    — model metrics, MAE by hour
  🔍 Backtesting    — actuals vs forecast for any date range
  🗄️  Data explorer — raw prices, weather, generation

Usage
-----
  streamlit run dashboard.py

Config (via .env or environment variables)
-----
  API_URL=http://localhost:8000   (default)
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

BLUE   = "#2563eb"
BAND   = "rgba(99,153,255,0.15)"
ACTUAL = "#111111"
GREEN  = "#10b981"
AMBER  = "#f59e0b"
RED    = "#ef4444"
GRAY   = "#888888"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "SE3 Electricity Forecast",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Shared styles ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    [data-testid="stSidebar"] { min-width: 220px; max-width: 220px; }
    .metric-card {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; }
    .metric-sub   { font-size: 12px; color: #aaa; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(endpoint: str, params: dict | None = None) -> dict | None:
    """Call the FastAPI backend. Returns None on error."""
    try:
        r = requests.get(f"{API_URL}/{endpoint.lstrip('/')}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at **{API_URL}**. Is `python api.py` running?")
    except requests.exceptions.Timeout:
        st.error("API request timed out.")
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = e.response.text or str(e)
        st.error(f"API error {e.response.status_code}: {detail}")
    except requests.exceptions.JSONDecodeError:
        st.error("API returned an empty or invalid response.")
    return None


@st.cache_data(ttl=300, show_spinner=False)   # cache 5 min
def fetch_forecast() -> dict | None:
    return api_get("forecast")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_metrics() -> dict | None:
    return api_get("metrics")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(from_date: str, to_date: str) -> dict | None:
    return api_get("history", {"from_date": from_date, "to_date": to_date})


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(from_date: str, to_date: str) -> dict | None:
    return api_get("prices", {"from_date": from_date, "to_date": to_date})


@st.cache_data(ttl=300, show_spinner=False)
def fetch_weather(from_date: str, to_date: str) -> dict | None:
    return api_get("weather", {"from_date": from_date, "to_date": to_date})


@st.cache_data(ttl=300, show_spinner=False)
def fetch_generation(from_date: str, to_date: str) -> dict | None:
    return api_get("generation", {"from_date": from_date, "to_date": to_date})


def records_to_df(records: list[dict], ts_col: str = "timestamp") -> pd.DataFrame:
    df = pd.DataFrame(records)
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
    return df


def metric_card(label: str, value: str, sub: str = "", color: str = BLUE) -> None:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def plotly_layout(fig: go.Figure, title: str, y_label: str = "EUR/MWh") -> go.Figure:
    fig.update_layout(
        title      = title,
        xaxis_title= "Time",
        yaxis_title= y_label,
        yaxis      = dict(rangemode="tozero"),
        legend     = dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode  = "x unified",
        template   = "plotly_white",
        height     = 420,
        margin     = dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ SE3 Forecast")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📈 Forecast", "📊 Performance", "🔍 Backtesting", "🗄️ Data explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Health check
    health = api_get("health")
    if health and health.get("status") == "ok":
        st.success("API connected", icon="✅")
        st.caption(f"Model loaded: {'yes' if health['model_loaded'] else 'no'}")
    else:
        st.error("API offline", icon="🔴")
        st.caption(f"Expected at: {API_URL}")

    st.markdown("---")
    if st.button("🔄 Refresh data", width='stretch'):
        st.cache_data.clear()
        st.rerun()


# ── Page: Forecast ────────────────────────────────────────────────────────────

if page == "📈 Forecast":
    st.title("📈 SE3 Day-Ahead Price Forecast")

    with st.spinner("Fetching forecast..."):
        data = fetch_forecast()

    if not data:
        st.stop()

    df = records_to_df(data["hours"])
    df.index = pd.to_datetime(df.index)

    # Key numbers
    st.markdown(f"**Generated:** {pd.to_datetime(data['generated_at']).strftime('%Y-%m-%d %H:%M')}  "
                f"&nbsp;|&nbsp; "
                f"**Horizon:** {pd.to_datetime(data['forecast_from']).strftime('%d %b %H:%M')} → "
                f"{pd.to_datetime(data['forecast_to']).strftime('%d %b %H:%M')}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Avg forecast", f"{df['p50'].mean():.1f}", "EUR/MWh")
    with col2:
        metric_card("Peak (p50)", f"{df['p50'].max():.1f}",
                    f"at {df['p50'].idxmax().strftime('%H:%M')}", AMBER)
    with col3:
        metric_card("Overnight low", f"{df.loc[df.index.hour.isin([0,1,2,3,4,5]), 'p50'].min():.1f}",
                    "EUR/MWh  00–05h", BLUE)
    with col4:
        metric_card("Uncertainty band",
                    f"±{((df['p95'] - df['p05']) / 2).mean():.1f}",
                    "avg half-width EUR/MWh", GRAY)

    # Forecast chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = list(df.index) + list(df.index[::-1]),
        y    = list(df["p95"]) + list(df["p05"][::-1]),
        fill = "toself", fillcolor=BAND,
        line = dict(color="rgba(0,0,0,0)"),
        name = "90% confidence band (q5–q95)",
    ))
    fig.add_trace(go.Scatter(
        x    = df.index, y=df["p50"],
        name = "Forecast (median)",
        line = dict(color=BLUE, width=2.5),
    ))
    plotly_layout(fig, "SE3 Next 24h Price Forecast")
    st.plotly_chart(fig, width='stretch')

    # Hourly table
    with st.expander("📋 Hourly forecast table"):
        table = df.copy()
        table.index = table.index.strftime("%Y-%m-%d %H:%M")
        table.columns = ["Lower (q5)", "Median (q50)", "Upper (q95)"]
        st.dataframe(table.round(2), width='stretch')


# ── Page: Performance ─────────────────────────────────────────────────────────

elif page == "📊 Performance":
    st.title("📊 Model Performance")

    with st.spinner("Loading metrics..."):
        m = fetch_metrics()

    if not m:
        st.info("No metrics found. Run `python ml.py --train` first.")
        st.stop()

    st.caption(f"Test set: **{m['test_from']}** → **{m['test_to']}**")

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("MAE", f"{m['mae']:.2f}", "EUR/MWh overall")
    with c2:
        metric_card("RMSE", f"{m['rmse']:.2f}", "EUR/MWh overall")
    with c3:
        mape_color = GREEN if m["mape"] < 20 else AMBER if m["mape"] < 40 else RED
        metric_card("MAPE", f"{m['mape']:.1f}%", "excl. near-zero hours", mape_color)
    with c4:
        cov_color = GREEN if m["coverage_q5_q95"] >= 85 else AMBER
        metric_card("PI Coverage", f"{m['coverage_q5_q95']:.1f}%",
                    "q5–q95 ideal ~90%", cov_color)
    with c5:
        metric_card("Spike MAE",
                    f"{m['spike_mae']:.1f}" if m["spike_mae"] else "N/A",
                    f"n={m['n_spikes']} hours >100 EUR/MWh", RED)

    st.markdown("---")
    c_left, c_right = st.columns(2)

    # MAE by hour
    with c_left:
        st.subheader("MAE by hour of day")
        mae_hour = pd.Series(m["mae_by_hour"]).sort_index()
        mae_hour.index = mae_hour.index.astype(int)

        fig = go.Figure()
        colors = [
            RED if h in [8, 9, 17, 18, 19, 20] else
            BLUE if h in [0, 1, 2, 3, 4, 5, 23] else GRAY
            for h in mae_hour.index
        ]
        fig.add_trace(go.Bar(
            x     = mae_hour.index,
            y     = mae_hour.values,
            marker_color = colors,
            name  = "MAE",
        ))
        fig.update_layout(
            xaxis        = dict(tickmode="linear", tick0=0, dtick=1),
            yaxis_title  = "MAE (EUR/MWh)",
            template     = "plotly_white",
            height       = 320,
            margin       = dict(l=0, r=0, t=10, b=0),
            showlegend   = False,
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("🔴 Peak hours  🔵 Night hours  ⬛ Daytime")

    # Night vs peak breakdown
    with c_right:
        st.subheader("Error breakdown")
        breakdown = pd.DataFrame({
            "Period": ["Overall", "Night (23–05h)", "Peak (07–09, 17–20h)"],
            "MAE (EUR/MWh)": [m["mae"], m["night_mae"], m["peak_mae"]],
        })
        fig2 = go.Figure(go.Bar(
            x              = breakdown["MAE (EUR/MWh)"],
            y              = breakdown["Period"],
            orientation    = "h",
            marker_color   = [GRAY, BLUE, RED],
        ))
        fig2.update_layout(
            xaxis_title  = "MAE (EUR/MWh)",
            template     = "plotly_white",
            height       = 200,
            margin       = dict(l=0, r=0, t=10, b=0),
            showlegend   = False,
        )
        st.plotly_chart(fig2, width='stretch')

        st.markdown("---")
        st.subheader("Summary")
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | MAE | **{m['mae']:.2f}** EUR/MWh |
        | RMSE | **{m['rmse']:.2f}** EUR/MWh |
        | MAPE | **{m['mape']:.1f}%** |
        | Coverage (q5–q95) | **{m['coverage_q5_q95']:.1f}%** |
        | Night MAE | **{m['night_mae']:.2f}** EUR/MWh |
        | Peak MAE | **{m['peak_mae']:.2f}** EUR/MWh |
        | Test period | {m['test_from']} → {m['test_to']} |
        """)


# ── Page: Backtesting ─────────────────────────────────────────────────────────

elif page == "🔍 Backtesting":
    st.title("🔍 Backtesting — Actual vs Forecast")

    col1, col2 = st.columns(2)
    with col1:
        from_date = st.date_input("From", value=date.today() - timedelta(days=14))
    with col2:
        to_date   = st.date_input("To",   value=date.today())

    if from_date >= to_date:
        st.error("From date must be before To date.")
        st.stop()

    with st.spinner("Loading history..."):
        data = fetch_history(str(from_date), str(to_date))

    if not data:
        st.stop()

    df = records_to_df(data["records"])
    df.index = pd.to_datetime(df.index)

    has_fc = df["p50"].notna()
    n_fc   = has_fc.sum()

    # Summary row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Hours", f"{data['n_hours']:,}", f"{from_date} → {to_date}")
    with c2:
        metric_card("Hours with forecast", f"{n_fc:,}",
                    f"{100*n_fc/len(df):.0f}% coverage")
    with c3:
        mae_color = GREEN if (data["mae"] or 99) < 20 else AMBER
        metric_card("MAE", f"{data['mae']:.2f}" if data["mae"] else "—",
                    "EUR/MWh", mae_color)
    with c4:
        if has_fc.any():
            coverage = ((df.loc[has_fc, "actual"] >= df.loc[has_fc, "p05"]) &
                        (df.loc[has_fc, "actual"] <= df.loc[has_fc, "p95"])).mean() * 100
            metric_card("PI Coverage", f"{coverage:.1f}%", "q5–q95", GREEN if coverage >= 85 else AMBER)
        else:
            metric_card("PI Coverage", "—", "no forecast data")

    st.markdown("---")

    # Actual vs forecast chart
    fig = go.Figure()
    if has_fc.any():
        fc_df = df[has_fc]
        fig.add_trace(go.Scatter(
            x    = list(fc_df.index) + list(fc_df.index[::-1]),
            y    = list(fc_df["p95"]) + list(fc_df["p05"][::-1]),
            fill = "toself", fillcolor=BAND,
            line = dict(color="rgba(0,0,0,0)"),
            name = "90% band",
        ))
        fig.add_trace(go.Scatter(
            x    = fc_df.index, y=fc_df["p50"],
            name = "Forecast (median)",
            line = dict(color=BLUE, width=1.8),
        ))
    fig.add_trace(go.Scatter(
        x    = df.index, y=df["actual"],
        name = "Actual",
        line = dict(color=ACTUAL, width=1.5),
    ))
    plotly_layout(fig, "Actual vs Forecast")
    st.plotly_chart(fig, width='stretch')

    if has_fc.any():
        col_left, col_right = st.columns(2)

        # Rolling MAE (daily)
        with col_left:
            st.subheader("Daily MAE trend")
            daily_mae = (df.loc[has_fc, "abs_error"]
                         .resample("D").mean()
                         .dropna())
            fig2 = go.Figure(go.Scatter(
                x    = daily_mae.index,
                y    = daily_mae.values,
                fill = "toself",
                fillcolor = "rgba(37,99,235,0.1)",
                line = dict(color=BLUE, width=1.5),
                name = "Daily MAE",
            ))
            fig2.update_layout(
                yaxis_title = "MAE (EUR/MWh)",
                template    = "plotly_white",
                height      = 280,
                margin      = dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig2, width='stretch')

        # Error distribution
        with col_right:
            st.subheader("Error distribution")
            errors = df.loc[has_fc, "error"].dropna()
            fig3 = go.Figure(go.Histogram(
                x         = errors,
                nbinsx    = 40,
                marker    = dict(color=BLUE, opacity=0.7),
                name      = "Forecast error",
            ))
            fig3.add_vline(x=0, line_color=RED, line_dash="dash", line_width=1)
            fig3.update_layout(
                xaxis_title = "Error (EUR/MWh) — forecast minus actual",
                yaxis_title = "Count",
                template    = "plotly_white",
                height      = 280,
                margin      = dict(l=0, r=0, t=10, b=0),
                showlegend  = False,
            )
            st.plotly_chart(fig3, width='stretch')

        # Scatter: predicted vs actual
        st.subheader("Predicted vs actual scatter")
        fc_df = df[has_fc].dropna(subset=["actual", "p50"])
        perfect = [fc_df["actual"].min(), fc_df["actual"].max()]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x    = perfect, y=perfect,
            mode = "lines",
            line = dict(color=RED, dash="dash", width=1),
            name = "Perfect forecast",
        ))
        fig4.add_trace(go.Scatter(
            x      = fc_df["actual"],
            y      = fc_df["p50"],
            mode   = "markers",
            marker = dict(color=BLUE, size=3, opacity=0.5),
            name   = "Predicted vs actual",
        ))
        fig4.update_layout(
            xaxis_title = "Actual (EUR/MWh)",
            yaxis_title = "Forecast p50 (EUR/MWh)",
            template    = "plotly_white",
            height      = 380,
            margin      = dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig4, width='stretch')


# ── Page: Data explorer ───────────────────────────────────────────────────────

elif page == "🗄️ Data explorer":
    st.title("🗄️ Data Explorer")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        from_date = st.date_input("From", value=date.today() - timedelta(days=7))
    with col2:
        to_date   = st.date_input("To",   value=date.today())
    with col3:
        dataset   = st.selectbox("Dataset", ["Prices", "Weather", "Generation"])

    if from_date >= to_date:
        st.error("From date must be before To date.")
        st.stop()

    fs, ts = str(from_date), str(to_date)

    with st.spinner(f"Loading {dataset.lower()}..."):
        if dataset == "Prices":
            raw = fetch_prices(fs, ts)
        elif dataset == "Weather":
            raw = fetch_weather(fs, ts)
        else:
            raw = fetch_generation(fs, ts)

    if not raw:
        st.stop()

    df = records_to_df(raw["records"])
    df.index = pd.to_datetime(df.index)

    st.caption(f"**{raw['n_hours']:,} hours** · {from_date} → {to_date}")

    # ── Prices ────────────────────────────────────────────────────────────────
    if dataset == "Prices":
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Mean", f"{df['price_eur_mwh'].mean():.2f}", "EUR/MWh")
        with c2: metric_card("Max",  f"{df['price_eur_mwh'].max():.2f}",  "EUR/MWh", RED)
        with c3: metric_card("Min",  f"{df['price_eur_mwh'].min():.2f}",  "EUR/MWh", GREEN)
        with c4: metric_card("Std",  f"{df['price_eur_mwh'].std():.2f}",  "EUR/MWh", GRAY)

        fig = go.Figure(go.Scatter(
            x    = df.index, y=df["price_eur_mwh"],
            name = "Price", fill="tozeroy",
            line = dict(color=BLUE, width=1.2),
            fillcolor = "rgba(37,99,235,0.08)",
        ))
        plotly_layout(fig, "SE3 Day-Ahead Prices")
        st.plotly_chart(fig, width='stretch')

    # ── Weather ───────────────────────────────────────────────────────────────
    elif dataset == "Weather":
        tab1, tab2, tab3 = st.tabs(["Wind", "Temperature", "Cloud & Solar"])

        with tab1:
            fig = go.Figure()
            if "windspeed_100m" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["windspeed_100m"],
                    name="100m (hub height)", line=dict(color=BLUE, width=1.5)
                ))
            if "windspeed_10m" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["windspeed_10m"],
                    name="10m", line=dict(color=GRAY, width=1, dash="dot")
                ))
            plotly_layout(fig, "Wind speed", "m/s")
            st.plotly_chart(fig, width='stretch')

        with tab2:
            fig = go.Figure(go.Scatter(
                x=df.index, y=df["temperature"],
                name="Temperature", fill="tozeroy",
                line=dict(color=AMBER, width=1.5),
                fillcolor="rgba(245,158,11,0.08)",
            ))
            plotly_layout(fig, "Temperature", "°C")
            st.plotly_chart(fig, width='stretch')

        with tab3:
            fig = go.Figure()
            if "cloudcover" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["cloudcover"],
                    name="Cloud cover (%)", line=dict(color=GRAY, width=1.5)
                ))
            if "solar_radiation" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["solar_radiation"],
                    name="Solar radiation (W/m²)",
                    line=dict(color=AMBER, width=1.5), yaxis="y2"
                ))
            fig.update_layout(
                yaxis2=dict(title="W/m²", overlaying="y", side="right"),
                template="plotly_white", height=420,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, width='stretch')

    # ── Generation ────────────────────────────────────────────────────────────
    elif dataset == "Generation":
        tab1, tab2 = st.tabs(["Generation mix", "Load"])

        with tab1:
            fig = go.Figure()
            if "wind_gen_mw" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["wind_gen_mw"],
                    name="Wind (MW)", line=dict(color=GREEN, width=1.5)
                ))
            if "nuclear_gen_mw" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["nuclear_gen_mw"],
                    name="Nuclear (MW)", line=dict(color=AMBER, width=1.5)
                ))
            plotly_layout(fig, "Generation", "MW")
            st.plotly_chart(fig, width='stretch')

        with tab2:
            if "load_mw" in df.columns:
                fig = go.Figure(go.Scatter(
                    x=df.index, y=df["load_mw"],
                    name="Load (MW)", fill="tozeroy",
                    line=dict(color=RED, width=1.5),
                    fillcolor="rgba(239,68,68,0.08)",
                ))
                plotly_layout(fig, "Total load SE3", "MW")
                st.plotly_chart(fig, width='stretch')

    # Raw data table
    with st.expander("📋 Raw data table"):
        st.dataframe(df.round(2), width='stretch')
        csv = df.round(2).reset_index().to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            data     = csv,
            file_name= f"se3_{dataset.lower()}_{from_date}_{to_date}.csv",
            mime     = "text/csv",
        )
