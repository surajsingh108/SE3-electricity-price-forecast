"""
dashboard.py — SE3 Price Forecast Dashboard

Reads directly from DuckDB (via GCS volume mount at /app/data/).
No API service required.

Pages:
  📈 Forecast     — next 24h price forecast with confidence band
  📊 Performance  — model metrics
  📉 Backtesting  — actuals vs forecast for any date range
  🗃️  Data        — raw prices, weather, generation
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DB_PATH    = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")
MODEL_DIR  = Path(os.environ.get("MODEL_DIR", "model"))

BLUE  = "#2563eb"
BAND  = "rgba(99,153,255,0.15)"
ACTUAL= "#111111"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED   = "#ef4444"
GRAY  = "#888888"

st.set_page_config(
    page_title="SE3 Electricity Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { min-width: 220px; max-width: 220px; }
    .metric-card {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px; padding: 16px 20px; text-align: center;
    }
    .metric-label { font-size: 13px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; }
    .metric-sub   { font-size: 12px; color: #aaa; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_conn():
    """Persistent DuckDB connection. Re-opens if DB doesn't exist yet."""
    if not Path(DB_PATH).exists():
        st.error(f"Database not found at {DB_PATH}. "
                 "The pipeline job may not have run yet.")
        st.stop()
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_forecast() -> pd.DataFrame | None:
    """Load the most recent forecast from the forecasts table."""
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        df = conn.execute("""
            SELECT timestamp, p05, p50, p95
            FROM forecasts
            WHERE generated_at = (SELECT MAX(generated_at) FROM forecasts)
            ORDER BY timestamp
        """).df()
        conn.close()
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Europe/Stockholm")
        return df.set_index("timestamp")
    except Exception as e:
        st.error(f"Error loading forecast: {e}")
        return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_metrics() -> dict | None:
    """Load metrics.json from model directory."""
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(from_date: str, to_date: str) -> pd.DataFrame:
    conn = duckdb.connect(DB_PATH, read_only=True)
    df = conn.execute("""
        SELECT timestamp, price_eur_mwh FROM prices
        WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp
    """, [from_date, to_date]).df()
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Europe/Stockholm")
    return df.set_index("timestamp")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_weather(from_date: str, to_date: str) -> pd.DataFrame:
    conn = duckdb.connect(DB_PATH, read_only=True)
    df = conn.execute("""
        SELECT * FROM weather
        WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp
    """, [from_date, to_date]).df()
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Europe/Stockholm")
    return df.set_index("timestamp")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_generation(from_date: str, to_date: str) -> pd.DataFrame:
    conn = duckdb.connect(DB_PATH, read_only=True)
    df = conn.execute("""
        SELECT g.timestamp, g.wind_gen_mw, g.load_mw,
               ng.nuclear_gen_mw
        FROM generation g
        LEFT JOIN nuclear_gen ng ON g.timestamp = ng.timestamp
        WHERE g.timestamp >= ? AND g.timestamp <= ?
        ORDER BY g.timestamp
    """, [from_date, to_date]).df()
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Europe/Stockholm")
    return df.set_index("timestamp")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(from_date: str, to_date: str) -> pd.DataFrame:
    """Load actuals vs latest forecast for the date range."""
    conn = duckdb.connect(DB_PATH, read_only=True)
    df = conn.execute("""
        SELECT p.timestamp, p.price_eur_mwh AS actual,
               f.p05, f.p50, f.p95
        FROM prices p
        LEFT JOIN (
            SELECT timestamp, p05, p50, p95
            FROM forecasts
            WHERE generated_at = (
                SELECT MAX(generated_at) FROM forecasts
                WHERE generated_at <= p.timestamp
            )
        ) f ON p.timestamp = f.timestamp
        WHERE p.timestamp >= ? AND p.timestamp <= ?
        ORDER BY p.timestamp
    """, [from_date, to_date]).df()
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Europe/Stockholm")
    df = df.set_index("timestamp")
    df["error"] = df["p50"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_weather_range(from_date: str, to_date: str) -> pd.DataFrame:
    """Load weather and forecast data for the date range."""
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        tables = conn.execute("SHOW TABLES").df()["name"].tolist()
        has_fcst = "weather_forecast" in tables

        query = """
            SELECT
                w.timestamp,
                w.windspeed_100m,
                w.temperature,
                {} as fcst_wind_100m
            FROM weather w
            {}
            WHERE w.timestamp >= ? AND w.timestamp <= ?
            ORDER BY w.timestamp
        """.format(
            "wf.fcst_wind_100m" if has_fcst else "NULL",
            "LEFT JOIN weather_forecast wf ON w.timestamp = wf.timestamp"
            if has_fcst else ""
        )

        df_w = conn.execute(query, [from_date, to_date]).df()
        conn.close()
        if df_w.empty:
            return pd.DataFrame()
        df_w["timestamp"] = pd.to_datetime(
            df_w["timestamp"], utc=True
        ).dt.tz_convert("Europe/Stockholm")
        return df_w.set_index("timestamp")
    except Exception as e:
        st.warning(f"Could not load weather overlays: {e}")
        return pd.DataFrame()


def metric_card(label, value, sub="", color=BLUE):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def plotly_layout(fig, title, y_label="EUR/MWh"):
    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title=y_label,
        yaxis=dict(rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified", template="plotly_white",
        height=420, margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ SE3 Forecast")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📈 Forecast", "📊 Performance", "📉 Backtesting", "🗃️ Data", "💬 Ask SE3"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # DB status
    db_exists = Path(DB_PATH).exists()
    if db_exists:
        st.success("Database connected", icon="✅")
    else:
        st.error(f"No database at {DB_PATH}", icon="🔴")

    # Metrics summary
    m = fetch_metrics()
    if m:
        st.caption(f"Model MAE: **{m['mae']:.2f}** EUR/MWh")

    st.markdown("---")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-source-181717?logo=github&style=flat)]"
        "(https://github.com/surajsingh108/SE3-electricity-price-forecast)"
    )
    st.caption("Built by Suraj Singh")


# ── Page: Forecast ────────────────────────────────────────────────────────────

if page == "📈 Forecast":
    st.title("📈 SE3 Day-Ahead Price Forecast")

    with st.spinner("Loading forecast..."):
        df = fetch_forecast()

    if df is None:
        st.info("No forecast found. The forecast job may not have run yet.")
        st.stop()
    else:
        tz  = pytz.timezone("Europe/Stockholm")
        now = pd.Timestamp.now(tz=tz).floor("h")
        df  = df[df.index >= now]

        if df.empty:
            st.warning("Forecast has expired. Waiting for next forecast job run.")
            st.stop()
        else:
            def highlight_now(row):
                if row.name == now:
                    return ["background-color: #1a3a4a"] * len(row)
                return [""] * len(row)

            col1, col2, col3, col4 = st.columns(4)
            with col1: metric_card("Avg forecast",  f"{df['p50'].mean():.1f}", "EUR/MWh")
            with col2: metric_card("Peak (p50)",    f"{df['p50'].max():.1f}",
                                   f"at {df['p50'].idxmax().strftime('%H:%M')}", AMBER)
            with col3:
                night = df.loc[df.index.hour.isin([0,1,2,3,4,5]), "p50"]
                metric_card("Overnight low", f"{night.min():.1f}" if not night.empty else "—",
                            "EUR/MWh  00–05h", BLUE)
            with col4: metric_card("Uncertainty",
                                   f"±{((df['p95']-df['p05'])/2).mean():.1f}",
                                   "avg half-width EUR/MWh", GRAY)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(df.index)+list(df.index[::-1]),
                y=list(df["p95"])+list(df["p05"][::-1]),
                fill="toself", fillcolor=BAND,
                line=dict(color="rgba(0,0,0,0)"),
                name="90% confidence band",
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df["p50"],
                name="Forecast (median)",
                line=dict(color=BLUE, width=2.5),
            ))
            plotly_layout(fig, "SE3 Next 24h Price Forecast")
            fig.update_xaxes(range=[now, now + pd.Timedelta(hours=24)])
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Hourly forecast table"):
                table = df.copy()
                table.columns = ["Lower (q5)", "Median (q50)", "Upper (q95)"]
                st.dataframe(table.round(2).style.apply(highlight_now, axis=1),
                             use_container_width=True)


# ── Page: Performance ─────────────────────────────────────────────────────────

elif page == "📊 Performance":
    st.title("📊 Model Performance")
    m = fetch_metrics()
    if not m:
        st.info("No metrics found. Run the train job first.")
        st.stop()

    st.caption(f"Test set: **{m['test_from']}** → **{m['test_to']}**")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card("MAE",  f"{m['mae']:.2f}",  "EUR/MWh")
    with c2: metric_card("RMSE", f"{m['rmse']:.2f}", "EUR/MWh")
    with c3:
        mape_color = GREEN if m["mape"]<20 else AMBER if m["mape"]<40 else RED
        metric_card("MAPE", f"{m['mape']:.1f}%", "excl. near-zero hours", mape_color)
    with c4:
        cov_color = GREEN if m["coverage_q5_q95"]>=85 else AMBER
        metric_card("PI Coverage", f"{m['coverage_q5_q95']:.1f}%",
                    "q5–q95 ideal ~90%", cov_color)
    with c5:
        metric_card("Spike MAE",
                    f"{m['spike_mae']:.1f}" if m.get("spike_mae") else "N/A",
                    f"n={m.get('n_spikes',0)} hours >100 EUR/MWh", RED)

    st.markdown("---")
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("MAE by hour of day")
        mae_hour = pd.Series(m["mae_by_hour"]).sort_index()
        mae_hour.index = mae_hour.index.astype(int)
        colors = [RED if h in [8,9,17,18,19,20] else
                  BLUE if h in [0,1,2,3,4,5,23] else GRAY
                  for h in mae_hour.index]
        fig = go.Figure(go.Bar(x=mae_hour.index, y=mae_hour.values,
                               marker_color=colors))
        fig.update_layout(xaxis=dict(tickmode="linear"), yaxis_title="MAE (EUR/MWh)",
                          template="plotly_white", height=320,
                          margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Peak hours  🔵 Night hours  ⬛ Daytime")

    with c_right:
        st.subheader("Error breakdown")
        breakdown = pd.DataFrame({
            "Period": ["Overall","Night (23–05h)","Peak (07–09, 17–20h)"],
            "MAE (EUR/MWh)": [m["mae"], m["night_mae"], m["peak_mae"]],
        })
        fig2 = go.Figure(go.Bar(
            x=breakdown["MAE (EUR/MWh)"], y=breakdown["Period"],
            orientation="h", marker_color=[GRAY, BLUE, RED]))
        fig2.update_layout(xaxis_title="MAE (EUR/MWh)", template="plotly_white",
                           height=200, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


# ── Page: Backtesting ─────────────────────────────────────────────────────────

elif page == "📉 Backtesting":
    st.title("📉 Backtesting — Actual vs Forecast")
    col1, col2 = st.columns(2)
    with col1: from_date = st.date_input("From", value=date.today()-timedelta(days=14))
    with col2: to_date   = st.date_input("To",   value=date.today())
    if from_date >= to_date:
        st.error("From date must be before To date.")
        st.stop()

    overlay = st.multiselect(
        "Overlay on chart",
        ["Wind speed (100m)", "Wind Forecast (Open-Meteo)",
         "Forecast error (actual – p50)", "Temperature"],
        default=[]
    )

    with st.spinner("Loading history..."):
        df = fetch_history(str(from_date), str(to_date))
        df_weather = fetch_weather_range(str(from_date), str(to_date))

    if df.empty:
        st.info("No data for this range.")
        st.stop()

    has_fc = df["p50"].notna()
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Hours", f"{len(df):,}", f"{from_date} → {to_date}")
    with c2: metric_card("Hours with forecast", f"{has_fc.sum():,}",
                         f"{100*has_fc.mean():.0f}% coverage")
    with c3:
        mae = df.loc[has_fc, "abs_error"].mean() if has_fc.any() else None
        metric_card("MAE", f"{mae:.2f}" if mae else "—", "EUR/MWh",
                    GREEN if (mae or 99)<20 else AMBER)
    with c4:
        if has_fc.any():
            cov = ((df.loc[has_fc,"actual"]>=df.loc[has_fc,"p05"]) &
                   (df.loc[has_fc,"actual"]<=df.loc[has_fc,"p95"])).mean()*100
            metric_card("PI Coverage", f"{cov:.1f}%", "q5–q95",
                        GREEN if cov>=85 else AMBER)
        else:
            metric_card("PI Coverage", "—", "no forecast data")

    st.markdown("---")

    from plotly.subplots import make_subplots

    needs_secondary = any(o in overlay for o in [
        "Wind speed (100m)", "Wind Forecast (Open-Meteo)", "Temperature"
    ])

    fig = make_subplots(specs=[[{"secondary_y": needs_secondary}]])

    # Confidence band
    if has_fc.any():
        fc_df = df[has_fc]
        fig.add_trace(go.Scatter(
            x=list(fc_df.index) + list(fc_df.index[::-1]),
            y=list(fc_df["p95"]) + list(fc_df["p05"][::-1]),
            fill="toself", fillcolor=BAND,
            line=dict(color="rgba(0,0,0,0)"),
            name="90% band"),
            secondary_y=False)
        fig.add_trace(go.Scatter(
            x=fc_df.index, y=fc_df["p50"],
            name="Forecast (median)",
            line=dict(color=BLUE, width=1.8)),
            secondary_y=False)

    # Actual price
    fig.add_trace(go.Scatter(
        x=df.index, y=df["actual"],
        name="Actual",
        line=dict(color=ACTUAL, width=1.5)),
        secondary_y=False)

    # Forecast error bars
    if "Forecast error (actual – p50)" in overlay and has_fc.any():
        errors = fc_df["actual"] - fc_df["p50"]
        fig.add_trace(go.Bar(
            x=fc_df.index,
            y=errors,
            name="Error (actual – p50)",
            marker_color=[
                "rgba(239,68,68,0.5)" if v > 0
                else "rgba(59,130,246,0.5)"
                for v in errors
            ],
            opacity=0.6),
            secondary_y=False)

    # Wind speed (Open-Meteo actual)
    if "Wind speed (100m)" in overlay and not df_weather.empty:
        if "windspeed_100m" in df_weather.columns:
            fig.add_trace(go.Scatter(
                x=df_weather.index,
                y=df_weather["windspeed_100m"],
                name="Wind speed 100m (m/s)",
                line=dict(color="rgba(16,185,129,0.9)",
                          width=1.5, dash="dot")),
                secondary_y=True)

    # Wind forecast (Open-Meteo)
    if "Wind Forecast (Open-Meteo)" in overlay and not df_weather.empty:
        if "fcst_wind_100m" in df_weather.columns:
            fig.add_trace(go.Scatter(
                x=df_weather.index,
                y=df_weather["fcst_wind_100m"],
                name="Wind Forecast (Open-Meteo, m/s)",
                line=dict(color="rgba(52,211,153,0.9)",
                          width=1.5, dash="dash")),
                secondary_y=True)

    # Temperature
    if "Temperature" in overlay and not df_weather.empty:
        if "temperature" in df_weather.columns:
            fig.add_trace(go.Scatter(
                x=df_weather.index,
                y=df_weather["temperature"],
                name="Temperature (°C)",
                line=dict(color="rgba(251,146,60,0.9)",
                          width=1.5, dash="dot")),
                secondary_y=True)

    fig.update_yaxes(title_text="Price (EUR/MWh)", secondary_y=False)
    if needs_secondary:
        fig.update_yaxes(
            title_text="Wind (m/s) / Temp (°C)",
            secondary_y=True,
            showgrid=False
        )
    plotly_layout(fig, "Actual vs Forecast")
    st.plotly_chart(fig, use_container_width=True)

    if has_fc.any():
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Daily MAE trend")
            daily_mae = df.loc[has_fc,"abs_error"].resample("D").mean().dropna()
            fig2 = go.Figure(go.Scatter(x=daily_mae.index, y=daily_mae.values,
                fill="toself", fillcolor="rgba(37,99,235,0.1)",
                line=dict(color=BLUE, width=1.5)))
            fig2.update_layout(yaxis_title="MAE (EUR/MWh)", template="plotly_white",
                               height=280, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)
        with col_r:
            st.subheader("Error distribution")
            errors = df.loc[has_fc,"error"].dropna()
            fig3 = go.Figure(go.Histogram(x=errors, nbinsx=40,
                marker=dict(color=BLUE, opacity=0.7)))
            fig3.add_vline(x=0, line_color=RED, line_dash="dash")
            fig3.update_layout(xaxis_title="Error (EUR/MWh)",
                               yaxis_title="Count", template="plotly_white",
                               height=280, margin=dict(l=0,r=0,t=10,b=0),
                               showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)


# ── Page: Data ────────────────────────────────────────────────────────────────

elif page == "🗃️ Data":
    st.title("🗃️ Data Explorer")
    col1,col2,col3 = st.columns([2,2,1])
    with col1: from_date = st.date_input("From", value=date.today()-timedelta(days=7))
    with col2: to_date   = st.date_input("To",   value=date.today())
    with col3: dataset   = st.selectbox("Dataset", ["Prices","Weather","Generation"])
    if from_date >= to_date:
        st.error("From date must be before To date.")
        st.stop()
    fs, ts = str(from_date), str(to_date)

    with st.spinner(f"Loading {dataset.lower()}..."):
        if dataset == "Prices":        df = fetch_prices(fs, ts)
        elif dataset == "Weather":     df = fetch_weather(fs, ts)
        else:                          df = fetch_generation(fs, ts)

    if df.empty:
        st.info("No data for this range.")
        st.stop()

    st.caption(f"**{len(df):,} hours** · {from_date} → {to_date}")

    if dataset == "Prices":
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("Mean", f"{df['price_eur_mwh'].mean():.2f}", "EUR/MWh")
        with c2: metric_card("Max",  f"{df['price_eur_mwh'].max():.2f}",  "EUR/MWh", RED)
        with c3: metric_card("Min",  f"{df['price_eur_mwh'].min():.2f}",  "EUR/MWh", GREEN)
        with c4: metric_card("Std",  f"{df['price_eur_mwh'].std():.2f}",  "EUR/MWh", GRAY)
        fig = go.Figure(go.Scatter(x=df.index, y=df["price_eur_mwh"],
            name="Price", fill="tozeroy",
            line=dict(color=BLUE, width=1.2),
            fillcolor="rgba(37,99,235,0.08)"))
        plotly_layout(fig, "SE3 Day-Ahead Prices")
        st.plotly_chart(fig, use_container_width=True)

    elif dataset == "Weather":
        tab1,tab2,tab3 = st.tabs(["Wind","Temperature","Cloud & Solar"])
        with tab1:
            fig = go.Figure()
            if "windspeed_100m" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["windspeed_100m"],
                    name="100m", line=dict(color=BLUE, width=1.5)))
            if "windspeed_10m" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["windspeed_10m"],
                    name="10m", line=dict(color=GRAY, width=1, dash="dot")))
            plotly_layout(fig, "Wind speed", "m/s")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = go.Figure(go.Scatter(x=df.index, y=df["temperature"],
                name="Temperature", fill="tozeroy",
                line=dict(color=AMBER, width=1.5),
                fillcolor="rgba(245,158,11,0.08)"))
            plotly_layout(fig, "Temperature", "°C")
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            fig = go.Figure()
            if "cloudcover" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["cloudcover"],
                    name="Cloud cover (%)", line=dict(color=GRAY, width=1.5)))
            if "solar_radiation" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["solar_radiation"],
                    name="Solar radiation (W/m²)",
                    line=dict(color=AMBER, width=1.5), yaxis="y2"))
            fig.update_layout(
                yaxis2=dict(title="W/m²", overlaying="y", side="right"),
                template="plotly_white", height=420,
                margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

    elif dataset == "Generation":
        tab1,tab2 = st.tabs(["Generation mix","Load"])
        with tab1:
            fig = go.Figure()
            if "wind_gen_mw" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["wind_gen_mw"],
                    name="Wind (MW)", line=dict(color=GREEN, width=1.5)))
            if "nuclear_gen_mw" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["nuclear_gen_mw"],
                    name="Nuclear (MW)", line=dict(color=AMBER, width=1.5)))
            plotly_layout(fig, "Generation", "MW")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            if "load_mw" in df.columns:
                fig = go.Figure(go.Scatter(x=df.index, y=df["load_mw"],
                    name="Load (MW)", fill="tozeroy",
                    line=dict(color=RED, width=1.5),
                    fillcolor="rgba(239,68,68,0.08)"))
                plotly_layout(fig, "Total load SE3", "MW")
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Raw data table"):
        st.dataframe(df.round(2), use_container_width=True)
        csv = df.round(2).reset_index().to_csv(index=False)
        st.download_button("⬇️ Download CSV", data=csv,
            file_name=f"se3_{dataset.lower()}_{from_date}_{to_date}.csv",
            mime="text/csv")


# ── Page: Ask SE3 ─────────────────────────────────────────────────────────────

elif page == "💬 Ask SE3":
    st.title("💬 Ask SE3")

    import httpx
    st.subheader("Ask anything about SE3 electricity prices")

    API_URL = os.environ.get("API_URL", "http://localhost:8000")

    if "se3_question" not in st.session_state:
        st.session_state.se3_question = ""

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Why is SE3 expensive right now?"):
            st.session_state.se3_question = "Why is SE3 electricity expensive right now?"
    with col2:
        if st.button("What will prices look like tomorrow?"):
            st.session_state.se3_question = "What will SE3 prices look like tomorrow?"
    with col3:
        if st.button("Should I run appliances now or wait?"):
            st.session_state.se3_question = "Should I run my appliances now or wait for cheaper prices?"
    with col4:
        if st.button("Why is the price high?"):
            st.session_state.se3_question = "Why is the SE3 electricity price high right now?"

    with st.form(key="ask_form"):
        question = st.text_input("Or type your own question:", value=st.session_state.se3_question)
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question:
        st.session_state.se3_question = question
        with st.spinner("Fetching live data and reasoning..."):
            try:
                r = httpx.post(f"{API_URL}/ask", json={"question": question}, timeout=30)
                r.raise_for_status()
                data = r.json()
                st.markdown(f"### Answer\n{data['answer']}")
                conf = data.get("confidence", 0)
                st.progress(conf, text=f"Confidence: {int(conf * 100)}%")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if data.get("current_price_eur_mwh") is not None:
                        st.metric("Current Price", f"{data['current_price_eur_mwh']:.1f} EUR/MWh")
                with col_b:
                    if data.get("forecast_p50_next_hour") is not None:
                        st.metric("Forecast (next hour)", f"{data['forecast_p50_next_hour']:.1f} EUR/MWh")
                with col_c:
                    if data.get("forecast_delta") is not None:
                        st.metric("Forecast vs Current", f"{data['forecast_delta']:+.1f} EUR/MWh")
                with st.expander("Data sources used"):
                    st.write("**Tools called:**", ", ".join(data.get("tools_called", [])))
                    if data.get("tools_failed"):
                        st.warning(f"Tools that failed: {', '.join(data['tools_failed'])}")
                    st.write("**Sources:**", ", ".join(data.get("sources", [])))
            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:13px;padding:8px'>"
    "Built by <a href='https://github.com/surajsingh108' target='_blank' "
    "style='color:#2563eb'>Suraj Singh</a> &nbsp;·&nbsp; "
    "Data: <a href='https://transparency.entsoe.eu' target='_blank' "
    "style='color:#2563eb'>ENTSO-E</a> + "
    "<a href='https://open-meteo.com' target='_blank' "
    "style='color:#2563eb'>Open-Meteo</a></div>",
    unsafe_allow_html=True,
)
