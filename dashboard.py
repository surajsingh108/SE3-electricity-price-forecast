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
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DB_PATH   = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))

# ── Color palette ─────────────────────────────────────────────────────────────
PRIMARY  = "#00d4aa"               # teal — SE3 brand / electricity
FORECAST = "#3742fa"               # blue — forecast lines
DANGER   = "#ff4757"               # red — spikes / high prices
WARNING  = "#ffa502"               # amber — stress
NEUTRAL  = "#747d8c"               # gray — muted / secondary
BAND     = "rgba(0,212,170,0.12)"  # teal confidence band
ACTUAL   = "#0f1117"               # near-black — actual price lines

# Aliases kept so any remaining chart code doesn't need mass renaming
BLUE  = FORECAST
GREEN = PRIMARY
AMBER = WARNING
RED   = DANGER
GRAY  = NEUTRAL

st.set_page_config(
    page_title="SE3 Electricity Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #1e2530 !important;
    min-width: 220px;
    max-width: 220px;
}
[data-testid="stSidebar"] * { color: #a8b2c1; }
[data-testid="stSidebar"] a { color: #636e72 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-size: 13px !important;
    color: #a8b2c1 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1a1f2e !important;
    border: 1px solid #1e2530 !important;
    color: #a8b2c1 !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    padding: 4px 10px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #00d4aa !important;
    color: #00d4aa !important;
}

/* ── Main content ──────────────────────────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1400px;
}

/* ── KPI cards (left-border accent, terminal style) ────────────────── */
.kpi-card {
    border-left: 3px solid #00d4aa;
    padding: 14px 18px;
    background: #ffffff;
    border-radius: 4px;
    border-top: 0.5px solid #e8ecf0;
    border-right: 0.5px solid #e8ecf0;
    border-bottom: 0.5px solid #e8ecf0;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #747d8c;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 400;
    color: #0f1117;
    line-height: 1;
}
.kpi-sub {
    font-size: 11px;
    color: #a0aab4;
    margin-top: 5px;
}

/* ── Buttons (Ask SE3 quick questions) ─────────────────────────────── */
.stButton > button {
    font-size: 12px !important;
    padding: 5px 14px !important;
    border-radius: 4px !important;
    border: 1px solid #e0e4ea !important;
    background: white !important;
    color: #636e72 !important;
    font-weight: 400 !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    border-color: #3742fa !important;
    color: #3742fa !important;
    background: white !important;
}
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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_imbalance_actuals(hours_back: int = 168) -> pd.DataFrame:
    """Load last N hours of imbalance actuals from imbalance_prices table."""
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        tables = conn.execute("SHOW TABLES").df()["name"].tolist()
        if "imbalance_prices" not in tables:
            conn.close()
            return pd.DataFrame()
        cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(hours=hours_back)
        df = conn.execute("""
            SELECT timestamp, imbl_price, direction, reg_spread,
                   up_reg_price, down_reg_price
            FROM imbalance_prices
            WHERE timestamp >= ?
            ORDER BY timestamp
        """, [str(cutoff)]).df()
        conn.close()
        if df.empty:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Europe/Stockholm")
        return df.set_index("timestamp")
    except Exception as e:
        st.warning(f"Could not load imbalance actuals: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_imbalance_forecast() -> pd.DataFrame:
    """Load most recent imbalance + spike forecast from imbalance_forecasts table."""
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        tables = conn.execute("SHOW TABLES").df()["name"].tolist()
        if "imbalance_forecasts" not in tables:
            conn.close()
            return pd.DataFrame()
        df = conn.execute("""
            SELECT timestamp, p05, p50, p95, spike_proba, regime
            FROM imbalance_forecasts
            WHERE generated_at = (SELECT MAX(generated_at) FROM imbalance_forecasts)
            ORDER BY timestamp
        """).df()
        conn.close()
        if df.empty:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Europe/Stockholm")
        return df.set_index("timestamp")
    except Exception as e:
        st.warning(f"Could not load imbalance forecast: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_imbalance_backtest_actuals(from_date, to_date):
    """
    Returns actuals for [from_date, to_date+1day) as a DataFrame
    indexed by timestamp in Europe/Stockholm.
    Columns: imbl_price, direction, reg_spread
    """
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        # Fetch all rows in window — let pandas handle the join later
        df = conn.execute("""
            SELECT timestamp, imbl_price, direction, reg_spread
            FROM imbalance_prices
            WHERE CAST(timestamp AS DATE) >= CAST(? AS DATE)
              AND CAST(timestamp AS DATE) <  CAST(? AS DATE) + INTERVAL 1 DAY
            ORDER BY timestamp
        """, [str(from_date), str(to_date)]).df()
        conn.close()
        if df.empty:
            return None
        # Normalize timezone to Europe/Stockholm
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)\
                            .dt.tz_convert("Europe/Stockholm")
        df = df.set_index("timestamp")
        return df
    except Exception as e:
        st.warning(f"Could not load imbalance actuals: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_imbalance_backtest_forecasts(from_date, to_date):
    """
    Returns forecasts whose TIMESTAMP falls in [from_date, to_date+1day).
    Filter is on timestamp (the period the forecast is for),
    NOT on generated_at (when the forecast was made).

    When multiple forecasts exist for the same timestamp
    (e.g. overlapping hindcasts), keep the one with the
    largest generated_at <= timestamp — i.e. the most recent
    forecast that was made BEFORE that period actually happened.
    Done in pandas, not SQL, so it's transparent.
    """
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        df = conn.execute("""
            SELECT timestamp, generated_at, p05, p50, p95,
                   spike_proba, regime
            FROM imbalance_forecasts
            WHERE CAST(timestamp AS DATE) >= CAST(? AS DATE)
              AND CAST(timestamp AS DATE) <  CAST(? AS DATE) + INTERVAL 1 DAY
            ORDER BY timestamp, generated_at
        """, [str(from_date), str(to_date)]).df()
        conn.close()
        if df.empty:
            return None

        # Normalize timezones to Europe/Stockholm
        df["timestamp"]    = pd.to_datetime(df["timestamp"],    utc=True)\
                              .dt.tz_convert("Europe/Stockholm")
        df["generated_at"] = pd.to_datetime(df["generated_at"], utc=True)\
                              .dt.tz_convert("Europe/Stockholm")

        # As-of dedup: keep the most recent forecast made BEFORE
        # the timestamp (no peeking into the future)
        df = df[df["generated_at"] <= df["timestamp"]]
        if df.empty:
            return None
        df = df.sort_values(["timestamp", "generated_at"])
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df = df.set_index("timestamp")
        return df
    except Exception as e:
        st.warning(f"Could not load imbalance forecasts: {e}")
        return None


# ── UI helpers ────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, sub: str = "",
             accent: str = "#00d4aa", danger: bool = False) -> None:
    border_color = "#ff4757" if danger else accent
    value_color  = "#ff4757" if danger else "#0f1117"
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color:{border_color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{value_color}">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def chart_layout(fig, title: str = "", y_label: str = "EUR/MWh",
                 height: int = 400, show_title: bool = False):
    title_cfg = dict(
        text=title,
        font=dict(size=13, color="#0f1117", family="system-ui, sans-serif"),
        x=0, xanchor="left", pad=dict(l=0),
    ) if show_title and title else None
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#fafbfc",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#636e72"),
        title=title_cfg if title_cfg is not None else dict(text=""),
        xaxis=dict(
            gridcolor="#edf0f4", gridwidth=0.5,
            linecolor="#e0e4ea", linewidth=1,
            tickfont=dict(size=10, color="#747d8c"),
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=10, color="#747d8c")),
            gridcolor="#edf0f4", gridwidth=0.5,
            linecolor="#e0e4ea", linewidth=1,
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#747d8c"),
            rangemode="normal",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=10, color="#636e72"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0f1117", font_color="white",
            font_size=11, bordercolor="#0f1117",
        ),
        height=height,
        margin=dict(l=12, r=12, t=40 if (show_title and title) else 12, b=12),
    )
    return fig


def page_header(section: str, title: str, subtitle: str = "") -> None:
    sub_html = (f'<div style="font-size:13px;color:#747d8c;margin-top:4px;">'
                f'{subtitle}</div>') if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <div style="font-size:10px;font-weight:600;letter-spacing:1.5px;
                    text-transform:uppercase;color:#00d4aa;margin-bottom:4px;">{section}</div>
        <div style="font-size:22px;font-weight:600;color:#0f1117;
                    letter-spacing:-0.3px;line-height:1.2;">{title}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(
        f'<div style="font-size:10px;font-weight:600;letter-spacing:1.2px;'
        f'text-transform:uppercase;color:#a0aab4;margin:1.5rem 0 0.75rem;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def _divider() -> None:
    st.markdown(
        '<hr style="border:none;border-top:1px solid #e8ecf0;margin:1.5rem 0;">',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;display:flex;align-items:center;gap:8px;">
        <span style="font-size:20px;">⚡</span>
        <span style="color:white;font-weight:600;font-size:15px;
                     letter-spacing:0.3px;">SE3 Forecast</span>
    </div>
    <div style="font-size:10px;color:#636e72;letter-spacing:1.5px;
                text-transform:uppercase;margin-bottom:1rem;
                padding-bottom:0.75rem;border-bottom:1px solid #1e2530;">
        Stockholm · Mälardalen
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "📈 Forecast",
            "⚡ Imbalance & Spike",
            "📊 Performance",
            "📉 Backtesting",
            "🗃️ Data",
            "💬 Ask SE3",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div style="border-top:1px solid #1e2530;margin:0.75rem 0;"></div>',
        unsafe_allow_html=True,
    )

    db_exists = Path(DB_PATH).exists()
    status_color = "#00d4aa" if db_exists else "#ff4757"
    status_text  = "DB connected" if db_exists else "DB not found"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
        <div style="width:6px;height:6px;border-radius:50%;
                    background:{status_color};flex-shrink:0;"></div>
        <span style="font-size:11px;color:#a8b2c1;">{status_text}</span>
    </div>""", unsafe_allow_html=True)

    m = fetch_metrics()
    if m:
        st.markdown(f"""
        <div style="font-size:11px;color:#636e72;margin-bottom:4px;">
            Spot MAE
            <span style="color:#00d4aa;font-family:'JetBrains Mono',monospace;
                         margin-left:4px;">{m['mae']:.2f}</span>
            <span style="color:#636e72;"> EUR/MWh</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div style="border-top:1px solid #1e2530;margin:0.75rem 0;"></div>',
        unsafe_allow_html=True,
    )

    if st.button("↺  Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style="margin-top:1rem;">
        <a href="https://github.com/surajsingh108/SE3-electricity-price-forecast"
           target="_blank"
           style="font-size:10px;color:#3d4451;text-decoration:none;letter-spacing:0.3px;">
           GitHub ↗</a>
    </div>
    <div style="margin-top:0.5rem;font-size:10px;color:#3d4451;">
        Built by Suraj Singh
    </div>""", unsafe_allow_html=True)


# ── Page: Forecast ────────────────────────────────────────────────────────────

if page == "📈 Forecast":
    page_header("Day-ahead market", "Price Forecast",
                "Next 24 hours · SE3 Stockholm/Mälardalen")

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
            with col1:
                kpi_card("Avg forecast", f"{df['p50'].mean():.1f}", "EUR/MWh")
            with col2:
                kpi_card("Peak (p50)", f"{df['p50'].max():.1f}",
                         f"at {df['p50'].idxmax().strftime('%H:%M')}", accent=WARNING)
            with col3:
                night = df.loc[df.index.hour.isin([0, 1, 2, 3, 4, 5]), "p50"]
                kpi_card("Overnight low",
                         f"{night.min():.1f}" if not night.empty else "—",
                         "EUR/MWh  00–05h")
            with col4:
                kpi_card("Uncertainty",
                         f"±{((df['p95']-df['p05'])/2).mean():.1f}",
                         "avg half-width EUR/MWh", accent=NEUTRAL)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(df.index) + list(df.index[::-1]),
                y=list(df["p95"]) + list(df["p05"][::-1]),
                fill="toself", fillcolor=BAND,
                line=dict(color="rgba(0,0,0,0)"),
                name="90% confidence band",
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df["p50"],
                name="Forecast (median)",
                line=dict(color=FORECAST, width=2.5),
            ))
            chart_layout(fig, "SE3 Next 24h Price Forecast",
                         height=420, show_title=True)
            fig.update_xaxes(range=[now, now + pd.Timedelta(hours=24)])
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Hourly forecast table"):
                table = df.copy()
                table.columns = ["Lower (q5)", "Median (q50)", "Upper (q95)"]
                st.dataframe(table.round(2).style.apply(highlight_now, axis=1),
                             use_container_width=True)


# ── Page: Imbalance & Spike ───────────────────────────────────────────────────

elif page == "⚡ Imbalance & Spike":
    from plotly.subplots import make_subplots

    page_header("Balancing market", "Imbalance & Spike Detection",
                "15-min imbalance price forecast · spike probability")

    with st.spinner("Loading imbalance data..."):
        df_act = fetch_imbalance_actuals(hours_back=168)
        df_fc  = fetch_imbalance_forecast()

    tz  = pytz.timezone("Europe/Stockholm")
    now = pd.Timestamp.now(tz=tz).floor("15min")

    # ── Top metric cards ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not df_act.empty and "imbl_price" in df_act.columns and df_act["imbl_price"].notna().any():
            last_valid_idx = df_act["imbl_price"].last_valid_index()
            cur_price = df_act["imbl_price"][last_valid_idx]
            age_min = int((pd.Timestamp.now(tz=tz) - last_valid_idx).total_seconds() / 60)
            age_str = (f"{age_min}min ago" if age_min < 60
                       else f"{age_min//60}h {age_min%60}min ago")
            # eSett has a ~6-12h settlement lag; show context when data is stale
            if age_min > 180:
                sub_str = f"EUR/MWh · last settled {age_str} · eSett lag"
            else:
                sub_str = f"EUR/MWh · data from {age_str}"
            _d   = abs(cur_price) > 200
            _acc = WARNING if abs(cur_price) > 50 else PRIMARY
            kpi_card("Current imbalance price",
                     f"{cur_price:.1f}",
                     sub_str,
                     accent=_acc, danger=_d)
        else:
            kpi_card("Current imbalance price", "—", "no recent data", accent=NEUTRAL)

    with col2:
        if not df_fc.empty and "p50" in df_fc.columns:
            nxt = df_fc["p50"].iloc[0] if len(df_fc) else float("nan")
            kpi_card("Next 15min p50",
                     f"{nxt:.1f}", "EUR/MWh forecast",
                     danger=abs(nxt) > 200)
        else:
            kpi_card("Next 15min p50", "—", "model not trained", accent=NEUTRAL)

    with col3:
        if not df_fc.empty and "spike_proba" in df_fc.columns:
            proba = df_fc["spike_proba"].iloc[0] if len(df_fc) else float("nan")
            _acc = DANGER if proba >= 0.45 else WARNING if proba >= 0.15 else PRIMARY
            kpi_card("Spike probability (next 15min)",
                     f"{proba*100:.0f}%", "forecast",
                     accent=_acc, danger=proba >= 0.45)
        else:
            kpi_card("Spike probability", "—", "model not trained", accent=NEUTRAL)

    with col4:
        if not df_fc.empty and "regime" in df_fc.columns:
            regime = df_fc["regime"].iloc[0] if len(df_fc) else "—"
            _acc = (PRIMARY if regime in ("normal_long", "deep_long")
                    else WARNING if regime == "normal_short" else DANGER)
            kpi_card("Current regime",
                     regime.replace("_", " ").title(),
                     "market condition", accent=_acc)
        else:
            kpi_card("Current regime", "—", "model not trained", accent=NEUTRAL)

    _divider()

    # ── Main chart (2 panels) ────────────────────────────────────────────────
    has_actuals  = not df_act.empty and "imbl_price" in df_act.columns
    has_forecast = not df_fc.empty and "p50" in df_fc.columns

    if not has_actuals and not has_forecast:
        st.info(
            "Imbalance model not yet trained or data not yet synced. Run:\n\n"
            "```\npython pipeline.py\n"
            "python ml_imbalance.py --train-all\n"
            "python ml_imbalance.py --forecast\n```"
        )
    else:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[3, 1],
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[
                "SE3 Imbalance Price — Actual (48h) + Forecast (24h)",
                "Spike Probability",
            ],
        )

        # Panel 1: Actuals (last 48h)
        if has_actuals:
            cutoff_48h = now - pd.Timedelta(hours=48)
            df_48 = df_act[df_act.index >= cutoff_48h]
            if not df_48.empty:
                fig.add_trace(go.Scatter(
                    x=df_48.index, y=df_48["imbl_price"].clip(-200, 600),
                    name="Actual imbalance price",
                    line=dict(color=ACTUAL, width=1.2),
                    hovertemplate="%{x}<br>Actual: %{y:.1f} EUR/MWh<extra></extra>",
                ), row=1, col=1)

        # Vertical "Now" line
        fig.add_vline(
            x=now.timestamp() * 1000,
            line_width=1, line_dash="dot", line_color=NEUTRAL,
            row=1, col=1,
        )

        # Panel 1: Show full forecast from its anchor point (covers the eSett lag gap).
        # Filtering to >= now would create a visible gap between actuals end and forecast
        # start, because the forecast is anchored to the last valid eSett row (~6-12h ago).
        # The "now" vline already marks the present; showing past forecast lets users
        # see both the gap period and the future prediction in one view.
        if has_forecast:
            df_fc24 = df_fc.head(96)

            # Regime background shading
            if "regime" in df_fc24.columns and not df_fc24.empty:
                regime_colors = {
                    "normal_long":  "rgba(0,212,170,0.06)",
                    "deep_long":    "rgba(0,212,170,0.12)",
                    "normal_short": "rgba(255,165,2,0.06)",
                    "stress":       "rgba(255,71,87,0.08)",
                    "extreme":      "rgba(255,71,87,0.15)",
                }
                prev_regime, seg_start = None, None
                for ts_i, row_i in df_fc24.iterrows():
                    r = row_i.get("regime", "normal_short")
                    if r != prev_regime:
                        if prev_regime is not None and seg_start is not None:
                            fig.add_vrect(
                                x0=seg_start.timestamp() * 1000,
                                x1=ts_i.timestamp() * 1000,
                                fillcolor=regime_colors.get(prev_regime, "rgba(0,0,0,0.03)"),
                                line_width=0, layer="below",
                                row=1, col=1,
                            )
                        prev_regime = r
                        seg_start   = ts_i

            # p05–p95 confidence band
            fig.add_trace(go.Scatter(
                x=list(df_fc24.index) + list(df_fc24.index[::-1]),
                y=list(df_fc24["p95"].clip(-200, 600)) +
                  list(df_fc24["p05"].clip(-200, 600)[::-1]),
                fill="toself", fillcolor="rgba(0,212,170,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                name="p05–p95 band",
                hoverinfo="skip",
            ), row=1, col=1)

            # p50 median forecast
            fig.add_trace(go.Scatter(
                x=df_fc24.index, y=df_fc24["p50"].clip(-200, 600),
                name="Forecast p50",
                line=dict(color=PRIMARY, width=2, dash="dash"),
                hovertemplate="%{x}<br>p50: %{y:.1f} EUR/MWh<extra></extra>",
            ), row=1, col=1)

            # Panel 2: Spike probability bars
            if "spike_proba" in df_fc24.columns:
                bar_colors = [
                    DANGER  if p >= 0.45 else
                    WARNING if p >= 0.15 else
                    PRIMARY
                    for p in df_fc24["spike_proba"]
                ]
                fig.add_trace(go.Bar(
                    x=df_fc24.index,
                    y=df_fc24["spike_proba"],
                    name="Spike probability",
                    marker_color=bar_colors,
                    marker_opacity=0.85,
                    hovertemplate="%{x}<br>Spike prob: %{y:.2f}<extra></extra>",
                ), row=2, col=1)

                for thresh, color in [(0.15, WARNING), (0.45, DANGER)]:
                    fig.add_hline(
                        y=thresh, line_dash="dash",
                        line_color=color, line_width=1.2,
                        row=2, col=1,
                    )

        _tick_font = dict(family="JetBrains Mono, monospace", size=10, color="#747d8c")
        fig.update_yaxes(title_text="EUR/MWh", gridcolor="#edf0f4",
                         tickfont=_tick_font, row=1, col=1)

        _sp_ymax = 1.0
        if has_forecast and not df_fc24.empty and "spike_proba" in df_fc24.columns:
            _sp_ymax = max(0.5, df_fc24["spike_proba"].max() * 1.4)
        fig.update_yaxes(title_text="Probability", range=[0, _sp_ymax],
                         gridcolor="#edf0f4", tickfont=_tick_font, row=2, col=1)

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="#fafbfc",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#636e72"),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#0f1117", font_color="white",
                            font_size=11, bordercolor="#0f1117"),
            template=None,
            height=560,
            margin=dict(l=12, r=12, t=60, b=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(size=10, color="#636e72"),
                        bgcolor="rgba(0,0,0,0)"),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Table + direction distribution ────────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            section_label("Imbalance forecast table")

            rows_table: list[dict] = []
            if has_forecast:
                act_prices = (df_act["imbl_price"].dropna()
                              if has_actuals else pd.Series(dtype=float))
                for ts_i, row_i in df_fc.iterrows():
                    actual_val = act_prices.get(ts_i)
                    is_future  = ts_i > now
                    rows_table.append({
                        "Time":  ts_i.strftime("%d %b %H:%M"),
                        "Actual": (
                            f"{actual_val:.1f}"
                            if actual_val is not None and not pd.isna(actual_val)
                            else ("—" if is_future else "pending")
                        ),
                        "p05":        f"{float(row_i.get('p05', float('nan'))):.1f}",
                        "p50":        f"{float(row_i.get('p50', float('nan'))):.1f}",
                        "p95":        f"{float(row_i.get('p95', float('nan'))):.1f}",
                        "Spike prob": f"{float(row_i.get('spike_proba', float('nan')))*100:.0f}%",
                        "Regime":     (row_i.get("regime") or "—").replace("_", " ").title(),
                    })
            elif has_actuals:
                df_act24 = df_act[df_act.index >= now - pd.Timedelta(hours=24)]
                for ts_i, row_i in df_act24.iterrows():
                    v = row_i.get("imbl_price")
                    rows_table.append({
                        "Time":       ts_i.strftime("%d %b %H:%M"),
                        "Actual":     f"{float(v):.1f}" if v is not None and not pd.isna(v) else "pending",
                        "p05":        "—", "p50": "—", "p95": "—",
                        "Spike prob": "—", "Regime": "—",
                    })

            if rows_table:
                df_tbl = pd.DataFrame(rows_table)

                def _color_spike(val):
                    try:
                        v = float(str(val).strip("%")) / 100
                        if v >= 0.45:
                            return "background-color: rgba(255,71,87,0.15)"
                        if v >= 0.15:
                            return "background-color: rgba(255,165,2,0.15)"
                        return "background-color: rgba(0,212,170,0.10)"
                    except (TypeError, ValueError):
                        return ""

                st.dataframe(
                    df_tbl.style.applymap(_color_spike, subset=["Spike prob"]),
                    use_container_width=True,
                    height=380,
                )
            else:
                st.info("No data available for table.")

        with col_r:
            section_label("Last 7 days — direction distribution")
            if has_actuals and "direction" in df_act.columns:
                dirs      = df_act["direction"].dropna()
                n_long    = int((dirs < 0).sum())
                n_neutral = int((dirs == 0).sum())
                n_short   = int((dirs > 0).sum())
                total     = n_long + n_neutral + n_short
                if total > 0:
                    vals = [
                        round(100 * n_long    / total, 1),
                        round(100 * n_neutral / total, 1),
                        round(100 * n_short   / total, 1),
                    ]
                    fig_dir = go.Figure(go.Bar(
                        x=vals,
                        y=["Long (surplus)", "Neutral", "Short (deficit)"],
                        orientation="h",
                        marker_color=[FORECAST, "#e0e4ea", DANGER],
                        text=[f"{v}%" for v in vals],
                        textposition="auto",
                    ))
                    fig_dir.update_layout(
                        paper_bgcolor="white",
                        plot_bgcolor="#fafbfc",
                        font=dict(family="JetBrains Mono, monospace", size=11),
                        xaxis_title="% of 15-min periods",
                        template=None,
                        height=200,
                        margin=dict(l=0, r=60, t=10, b=0),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_dir, use_container_width=True)
                else:
                    st.info("Not enough direction data.")
            else:
                st.info("Direction data not available.")

        if not has_forecast:
            st.info(
                "Imbalance forecast not yet generated. Run:\n\n"
                "```\npython ml_imbalance.py --train-all\n"
                "python ml_imbalance.py --forecast\n```"
            )


# ── Page: Performance ─────────────────────────────────────────────────────────

elif page == "📊 Performance":
    m = fetch_metrics()
    if not m:
        st.info("No metrics found. Run the train job first.")
        st.stop()

    page_header("Evaluation", "Model Performance",
                f"Test set · {m.get('test_from', '')} → {m.get('test_to', '')}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("MAE",  f"{m['mae']:.2f}",  "EUR/MWh")
    with c2:
        kpi_card("RMSE", f"{m['rmse']:.2f}", "EUR/MWh")
    with c3:
        _acc = PRIMARY if m["mape"] < 20 else WARNING if m["mape"] < 40 else DANGER
        kpi_card("MAPE", f"{m['mape']:.1f}%", "excl. near-zero hours",
                 accent=_acc, danger=m["mape"] >= 40)
    with c4:
        _acc = PRIMARY if m["coverage_q5_q95"] >= 85 else WARNING
        kpi_card("PI Coverage", f"{m['coverage_q5_q95']:.1f}%",
                 "q5–q95 ideal ~90%", accent=_acc)
    with c5:
        kpi_card("Spike MAE",
                 f"{m['spike_mae']:.1f}" if m.get("spike_mae") else "N/A",
                 f"n={m.get('n_spikes', 0)} hours >100 EUR/MWh", accent=DANGER)

    _divider()
    c_left, c_right = st.columns(2)
    with c_left:
        section_label("MAE by hour of day")
        mae_hour = pd.Series(m["mae_by_hour"]).sort_index()
        mae_hour.index = mae_hour.index.astype(int)
        colors = [
            DANGER   if h in [8, 9, 17, 18, 19, 20] else
            FORECAST if h in [0, 1, 2, 3, 4, 5, 23] else
            "#e0e4ea"
            for h in mae_hour.index
        ]
        fig = go.Figure(go.Bar(x=mae_hour.index, y=mae_hour.values,
                               marker_color=colors))
        chart_layout(fig, y_label="MAE (EUR/MWh)", height=320)
        fig.update_layout(xaxis=dict(tickmode="linear"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Peak hours  🔵 Night hours  ⬛ Daytime")

    with c_right:
        section_label("Error breakdown")
        breakdown = pd.DataFrame({
            "Period": ["Overall", "Night (23–05h)", "Peak (07–09, 17–20h)"],
            "MAE (EUR/MWh)": [m["mae"], m["night_mae"], m["peak_mae"]],
        })
        fig2 = go.Figure(go.Bar(
            x=breakdown["MAE (EUR/MWh)"], y=breakdown["Period"],
            orientation="h", marker_color=[NEUTRAL, FORECAST, DANGER]))
        chart_layout(fig2, y_label="", height=200)
        fig2.update_layout(xaxis_title="MAE (EUR/MWh)", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


# ── Page: Backtesting ─────────────────────────────────────────────────────────

elif page == "📉 Backtesting":
    page_header("Analysis", "Backtesting",
                "Compare actual prices against historical forecasts")

    tab1, tab2 = st.tabs(["📈 Spot Price", "⚡ Imbalance Price"])

    # ── Tab 1: Spot Price (existing code, unchanged) ──────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        with col1: from_date = st.date_input("From", value=date.today() - timedelta(days=14))
        with col2: to_date   = st.date_input("To",   value=date.today())
        if from_date >= to_date:
            st.error("From date must be before To date.")
            st.stop()

        overlay = st.multiselect(
            "Overlay on chart",
            ["Wind speed (100m)", "Wind Forecast (Open-Meteo)",
             "Forecast error (actual – p50)", "Temperature"],
            default=[],
        )

        with st.spinner("Loading history..."):
            df         = fetch_history(str(from_date), str(to_date))
            df_weather = fetch_weather_range(str(from_date), str(to_date))

        if df.empty:
            st.info("No data for this range.")
            st.stop()

        has_fc = df["p50"].notna()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Hours", f"{len(df):,}", f"{from_date} → {to_date}")
        with c2:
            kpi_card("Hours with forecast", f"{has_fc.sum():,}",
                     f"{100*has_fc.mean():.0f}% coverage")
        with c3:
            mae = df.loc[has_fc, "abs_error"].mean() if has_fc.any() else None
            _acc = PRIMARY if (mae or 99) < 20 else WARNING
            kpi_card("MAE", f"{mae:.2f}" if mae else "—", "EUR/MWh", accent=_acc)
        with c4:
            if has_fc.any():
                cov = (
                    (df.loc[has_fc, "actual"] >= df.loc[has_fc, "p05"]) &
                    (df.loc[has_fc, "actual"] <= df.loc[has_fc, "p95"])
                ).mean() * 100
                _acc = PRIMARY if cov >= 85 else WARNING
                kpi_card("PI Coverage", f"{cov:.1f}%", "q5–q95", accent=_acc)
            else:
                kpi_card("PI Coverage", "—", "no forecast data")

        _divider()

        from plotly.subplots import make_subplots

        needs_secondary = any(o in overlay for o in [
            "Wind speed (100m)", "Wind Forecast (Open-Meteo)", "Temperature"
        ])
        fig = make_subplots(specs=[[{"secondary_y": needs_secondary}]])

        if has_fc.any():
            fc_df = df[has_fc]
            fig.add_trace(go.Scatter(
                x=list(fc_df.index) + list(fc_df.index[::-1]),
                y=list(fc_df["p95"]) + list(fc_df["p05"][::-1]),
                fill="toself", fillcolor="rgba(55,66,250,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="90% band"),
                secondary_y=False)
            fig.add_trace(go.Scatter(
                x=fc_df.index, y=fc_df["p50"],
                name="Forecast (median)",
                line=dict(color=FORECAST, width=1.8)),
                secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["actual"],
            name="Actual",
            line=dict(color=ACTUAL, width=1.5)),
            secondary_y=False)

        if "Forecast error (actual – p50)" in overlay and has_fc.any():
            errors = fc_df["actual"] - fc_df["p50"]
            fig.add_trace(go.Bar(
                x=fc_df.index, y=errors,
                name="Error (actual – p50)",
                marker_color=[
                    "rgba(255,71,87,0.5)" if v > 0 else "rgba(55,66,250,0.5)"
                    for v in errors
                ],
                opacity=0.6),
                secondary_y=False)

        if "Wind speed (100m)" in overlay and not df_weather.empty:
            if "windspeed_100m" in df_weather.columns:
                fig.add_trace(go.Scatter(
                    x=df_weather.index, y=df_weather["windspeed_100m"],
                    name="Wind speed 100m (m/s)",
                    line=dict(color="rgba(0,212,170,0.9)", width=1.5, dash="dot")),
                    secondary_y=True)

        if "Wind Forecast (Open-Meteo)" in overlay and not df_weather.empty:
            if "fcst_wind_100m" in df_weather.columns:
                fig.add_trace(go.Scatter(
                    x=df_weather.index, y=df_weather["fcst_wind_100m"],
                    name="Wind Forecast (Open-Meteo, m/s)",
                    line=dict(color="rgba(0,212,170,0.65)", width=1.5, dash="dash")),
                    secondary_y=True)

        if "Temperature" in overlay and not df_weather.empty:
            if "temperature" in df_weather.columns:
                fig.add_trace(go.Scatter(
                    x=df_weather.index, y=df_weather["temperature"],
                    name="Temperature (°C)",
                    line=dict(color="rgba(255,165,2,0.9)", width=1.5, dash="dot")),
                    secondary_y=True)

        chart_layout(fig, "Actual vs Forecast", y_label="Price (EUR/MWh)",
                     height=420, show_title=True)
        if needs_secondary:
            fig.update_yaxes(
                title_text="Wind (m/s) / Temp (°C)",
                secondary_y=True,
                showgrid=False,
            )
        st.plotly_chart(fig, use_container_width=True)

        if has_fc.any():
            col_l, col_r = st.columns(2)
            with col_l:
                section_label("Daily MAE trend")
                daily_mae = df.loc[has_fc, "abs_error"].resample("D").mean().dropna()
                fig2 = go.Figure(go.Scatter(
                    x=daily_mae.index, y=daily_mae.values,
                    fill="toself", fillcolor="rgba(55,66,250,0.08)",
                    line=dict(color=FORECAST, width=1.5)))
                chart_layout(fig2, y_label="MAE (EUR/MWh)", height=280)
                st.plotly_chart(fig2, use_container_width=True)

            with col_r:
                section_label("Error distribution")
                errors = df.loc[has_fc, "error"].dropna()
                fig3 = go.Figure(go.Histogram(
                    x=errors, nbinsx=40,
                    marker=dict(color=FORECAST, opacity=0.7)))
                fig3.add_vline(x=0, line_color=DANGER, line_dash="dash")
                chart_layout(fig3, y_label="Count", height=280)
                fig3.update_layout(xaxis_title="Error (EUR/MWh)", showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 2: Imbalance Price Backtesting ────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            imbl_from = st.date_input(
                "From",
                value=date.today() - timedelta(days=7),
                min_value=date(2021, 11, 1),
                max_value=date.today() - timedelta(days=1),
                key="imbl_from",
            )
        with col2:
            imbl_to = st.date_input(
                "To",
                value=date.today() - timedelta(days=1),
                min_value=date(2021, 11, 1),
                max_value=date.today(),
                key="imbl_to",
            )
        st.caption(
            "Each day's hindcast forecasts the following day. "
            "Selecting Jun 1–7 shows forecast evaluation for Jun 2–7 "
            "(Jun 1 is the first hindcast date — no prior forecast exists for it). "
            "Hindcast available from 2026-06-01 onwards."
        )

        with st.spinner("Loading imbalance data..."):
            df_act = fetch_imbalance_backtest_actuals(imbl_from, imbl_to)
            df_fc  = fetch_imbalance_backtest_forecasts(imbl_from, imbl_to)

        if df_act is None or df_act.empty:
            st.info("No imbalance data for this date range.")
            st.caption("Data available from November 2021 onwards.")
            st.stop()

        # Merge: actuals get forecast columns where timestamps match
        if df_fc is not None and not df_fc.empty:
            df_merged = df_act.join(
                df_fc[["p05", "p50", "p95", "spike_proba", "regime"]],
                how="left"
            )
        else:
            df_merged = df_act.copy()
            for c in ["p05", "p50", "p95", "spike_proba", "regime"]:
                df_merged[c] = None

        # Rows that have both actual and forecast
        df_with_fc = df_merged.dropna(subset=["imbl_price", "p50"])

        has_enough = len(df_with_fc) > 10

        n_periods  = len(df_act)
        n_days     = (imbl_to - imbl_from).days + 1

        # Compute metrics (drop NaN rows to avoid nan results)
        if has_enough:
            _valid = df_with_fc.dropna(subset=["imbl_price", "p50"])
            y_true = _valid["imbl_price"]
            y_pred = _valid["p50"]
            i_mae  = float((y_true - y_pred).abs().mean()) if len(_valid) > 0 else None
            i_rmse = float(((y_true - y_pred) ** 2).mean() ** 0.5) if len(_valid) > 0 else None
            dir_correct = (
                (y_pred.diff().apply(np.sign) == y_true.diff().apply(np.sign))
                .dropna().mean() * 100
            ) if len(_valid) > 1 else None
            in_band  = (y_true >= _valid["p05"]) & (y_true <= _valid["p95"])
            coverage = float(in_band.mean() * 100) if len(_valid) > 0 else None
            actual_spikes = y_true > 200
            if actual_spikes.sum() > 0 and "spike_proba" in _valid.columns:
                spike_recall = float(
                    (_valid["spike_proba"] > 0.45)[actual_spikes].mean() * 100
                )
            else:
                spike_recall = None
        else:
            i_mae = i_rmse = dir_correct = coverage = spike_recall = None

        # ── KPI cards ─────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi_card("Periods", f"{n_periods:,}", f"{n_days} days · 15-min")
        with c2:
            _acc = FORECAST if i_mae and i_mae < 35 else WARNING
            kpi_card("Forecast MAE",
                     f"{i_mae:.1f}" if i_mae else "—",
                     "EUR/MWh · p50 vs actual", accent=_acc)
        with c3:
            _acc = PRIMARY if dir_correct and dir_correct > 60 else WARNING
            kpi_card("Direction accuracy",
                     f"{dir_correct:.1f}%" if dir_correct else "—",
                     "p50 move direction", accent=_acc)
        with c4:
            _acc = PRIMARY if coverage and coverage >= 85 else WARNING
            kpi_card("PI coverage",
                     f"{coverage:.1f}%" if coverage else "—",
                     "p05–p95 · target 90%", accent=_acc)
        with c5:
            _d = spike_recall is not None and spike_recall < 25
            _acc = PRIMARY if spike_recall and spike_recall > 40 else DANGER
            kpi_card("Spike recall",
                     f"{spike_recall:.0f}%" if spike_recall is not None else "—",
                     "spikes caught @ 0.45 thr",
                     accent=_acc, danger=_d)

        # ── Main chart: Actual vs Forecast ────────────────────────────────────
        section_label(f"Actual vs Forecast — {imbl_from} to {imbl_to}")

        if has_enough:
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(
                x=df_with_fc.index.tolist() + df_with_fc.index.tolist()[::-1],
                y=df_with_fc["p95"].tolist() + df_with_fc["p05"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(0,212,170,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                name="p05–p95 band",
            ))
            fig_main.add_trace(go.Scatter(
                x=df_with_fc.index, y=df_with_fc["p50"],
                name="Forecast p50",
                line=dict(color=PRIMARY, width=1.5, dash="dash"),
            ))
            fig_main.add_trace(go.Scatter(
                x=df_act.index, y=df_act["imbl_price"].clip(-300, 800),
                name="Actual price",
                line=dict(color=ACTUAL, width=1.2),
            ))
            chart_layout(fig_main, y_label="Imbalance price (EUR/MWh)", height=380)
            st.plotly_chart(fig_main, use_container_width=True)
            st.caption(
                f"Prices clipped to [−300, 800] EUR/MWh for readability. "
                f"{int((df_act['imbl_price'] > 800).sum())} extreme spikes "
                f"above 800 EUR/MWh not shown."
            )
        else:
            # Actuals-only fallback: show when no forecast data overlaps the range
            try:
                _conn_tmp = duckdb.connect(DB_PATH, read_only=True)
                _earliest = _conn_tmp.execute(
                    "SELECT MIN(DATE(generated_at)) FROM imbalance_forecasts"
                ).fetchone()[0]
                _conn_tmp.close()
                _earliest_str = str(_earliest) if _earliest else "unknown"
            except Exception:
                _earliest_str = "2026-06-01"

            st.caption(
                f"No forecast data for this period — showing actuals only. "
                f"Hindcast available from {_earliest_str} onwards."
            )
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(
                x=df_act.index, y=df_act["imbl_price"].clip(-300, 800),
                name="Actual price", line=dict(color=ACTUAL, width=1.2),
            ))
            chart_layout(fig_main, y_label="Imbalance price (EUR/MWh)", height=300)
            st.plotly_chart(fig_main, use_container_width=True)

            # Show price + direction stats even without forecast
            section_label("Price distribution")
            _col_s1, _col_s2 = st.columns(2)
            with _col_s1:
                _pv = df_act["imbl_price"].dropna()
                _ps = pd.DataFrame({
                    "Metric": ["Mean", "Median", "Std dev", "% negative", "% stress (>200)"],
                    "Value": [
                        f"{_pv.mean():.1f} EUR/MWh",
                        f"{_pv.median():.1f} EUR/MWh",
                        f"{_pv.std():.1f} EUR/MWh",
                        f"{(_pv < 0).mean()*100:.1f}%",
                        f"{(_pv > 200).mean()*100:.1f}%",
                    ],
                })
                st.dataframe(_ps, use_container_width=True, hide_index=True)
            with _col_s2:
                _dc = df_act["direction"].value_counts()
                _td = len(df_act)
                _fd = go.Figure(go.Bar(
                    x=[int(_dc.get(1, 0)), int(_dc.get(0, 0)), int(_dc.get(-1, 0))],
                    y=["Short (+1)", "Neutral (0)", "Long (−1)"],
                    orientation="h",
                    marker_color=[DANGER, "#e0e4ea", FORECAST],
                    text=[f"{_dc.get(1,0)/_td*100:.1f}%",
                          f"{_dc.get(0,0)/_td*100:.1f}%",
                          f"{_dc.get(-1,0)/_td*100:.1f}%"],
                    textposition="auto",
                ))
                chart_layout(_fd, y_label="", height=180)
                _fd.update_layout(margin=dict(l=60, r=20, t=10, b=10))
                st.plotly_chart(_fd, use_container_width=True)

        # ── Three sub-charts ──────────────────────────────────────────────────
        if has_enough:
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                section_label("Residuals (actual − p50)")
                residuals = (df_with_fc["imbl_price"] - df_with_fc["p50"]).clip(-300, 300)
                fig_r = go.Figure()
                fig_r.add_trace(go.Bar(
                    x=df_with_fc.index,
                    y=residuals,
                    marker_color=[DANGER if v > 0 else FORECAST for v in residuals],
                    showlegend=False,
                ))
                fig_r.add_hline(y=0, line_color="#e0e4ea", line_width=1)
                chart_layout(fig_r, y_label="EUR/MWh", height=220)
                st.plotly_chart(fig_r, use_container_width=True)

            with col_b:
                section_label("Spike probability vs actuals")
                fig_s = go.Figure()
                if "spike_proba" in df_with_fc.columns:
                    fig_s.add_trace(go.Scatter(
                        x=df_with_fc.index,
                        y=df_with_fc["spike_proba"],
                        name="Spike probability",
                        line=dict(color=WARNING, width=1.2),
                        fill="tozeroy",
                        fillcolor="rgba(255,165,2,0.12)",
                    ))
                    actual_spikes_mask = df_with_fc["imbl_price"] > 200
                    if actual_spikes_mask.any():
                        fig_s.add_trace(go.Scatter(
                            x=df_with_fc.index[actual_spikes_mask],
                            y=[1.0] * int(actual_spikes_mask.sum()),
                            mode="markers",
                            marker=dict(symbol="triangle-down", size=8, color=DANGER),
                            name="Actual spike (>200)",
                        ))
                    fig_s.add_hline(
                        y=0.45, line_dash="dash",
                        line_color=DANGER, line_width=1,
                        annotation_text="dispatch thr",
                        annotation_font_size=9,
                    )
                chart_layout(fig_s, y_label="Probability", height=220)
                fig_s.update_yaxes(range=[0, 1.1])
                st.plotly_chart(fig_s, use_container_width=True)

            with col_c:
                section_label("MAE by hour of day")
                hourly_mae = (
                    (df_with_fc["imbl_price"] - df_with_fc["p50"])
                    .abs()
                    .groupby(df_with_fc.index.hour)
                    .mean()
                )
                fig_h = go.Figure(go.Bar(
                    x=hourly_mae.index,
                    y=hourly_mae.values,
                    marker_color=[
                        DANGER   if h in [7, 8, 9, 17, 18, 19, 20] else
                        FORECAST if h in [0, 1, 2, 3, 4, 5, 23] else
                        "#e0e4ea"
                        for h in hourly_mae.index
                    ],
                    showlegend=False,
                ))
                chart_layout(fig_h, y_label="MAE (EUR/MWh)", height=220)
                st.plotly_chart(fig_h, use_container_width=True)

        # ── Summary stats ─────────────────────────────────────────────────────
        if has_enough or not df_act.empty:
            section_label("Summary statistics")
            col_x, col_y = st.columns(2)

            with col_x:
                section_label("Direction distribution")
                dir_counts = df_act["direction"].value_counts()
                total_dir  = len(df_act)
                # Plotly horizontal bars render first item at bottom → put Long last to appear at top
                _d_vals  = [int(dir_counts.get(1,  0)), int(dir_counts.get(0,  0)), int(dir_counts.get(-1, 0))]
                _d_lbls  = ["Short (+1)", "Neutral (0)", "Long (−1)"]
                _d_clrs  = [DANGER, "#e0e4ea", FORECAST]
                _d_texts = [f"{v / total_dir * 100:.1f}%" for v in _d_vals]
                fig_d = go.Figure(go.Bar(
                    x=_d_vals,
                    y=_d_lbls,
                    orientation="h",
                    marker_color=_d_clrs,
                    text=_d_texts,
                    textposition="auto",
                ))
                chart_layout(fig_d, y_label="", height=180)
                fig_d.update_layout(margin=dict(l=80, r=20, t=10, b=10))
                st.plotly_chart(fig_d, use_container_width=True)

            with col_y:
                section_label("Price distribution")
                price_valid = df_act["imbl_price"].dropna()
                price_stats = pd.DataFrame({
                    "Metric": [
                        "Mean", "Median", "Std dev",
                        "Min", "Max",
                        "% negative", "% stress (>200)",
                        "% extreme (>1000)",
                    ],
                    "Value": [
                        f"{price_valid.mean():.1f} EUR/MWh",
                        f"{price_valid.median():.1f} EUR/MWh",
                        f"{price_valid.std():.1f} EUR/MWh",
                        f"{price_valid.min():.1f} EUR/MWh",
                        f"{price_valid.max():.1f} EUR/MWh",
                        f"{(price_valid < 0).mean() * 100:.1f}%",
                        f"{(price_valid > 200).mean() * 100:.1f}%",
                        f"{(price_valid > 1000).mean() * 100:.2f}%",
                    ],
                })
                st.dataframe(price_stats,
                             use_container_width=True,
                             hide_index=True,
                             height=280)


# ── Page: Data ────────────────────────────────────────────────────────────────

elif page == "🗃️ Data":
    page_header("Raw data", "Data Explorer",
                "Prices · weather · generation · imbalance")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: from_date = st.date_input("From", value=date.today() - timedelta(days=7))
    with col2: to_date   = st.date_input("To",   value=date.today())
    with col3: dataset   = st.selectbox("Dataset", ["Prices", "Weather", "Generation"])
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

    st.caption(f"**{len(df):,} rows** · {from_date} → {to_date}")

    if dataset == "Prices":
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Mean", f"{df['price_eur_mwh'].mean():.2f}", "EUR/MWh")
        with c2: kpi_card("Max",  f"{df['price_eur_mwh'].max():.2f}",  "EUR/MWh", danger=True)
        with c3: kpi_card("Min",  f"{df['price_eur_mwh'].min():.2f}",  "EUR/MWh", accent=PRIMARY)
        with c4: kpi_card("Std",  f"{df['price_eur_mwh'].std():.2f}",  "EUR/MWh", accent=NEUTRAL)
        fig = go.Figure(go.Scatter(
            x=df.index, y=df["price_eur_mwh"],
            name="Price", fill="tozeroy",
            line=dict(color=FORECAST, width=1.2),
            fillcolor="rgba(55,66,250,0.06)"))
        chart_layout(fig, "SE3 Day-Ahead Prices", height=420, show_title=True)
        st.plotly_chart(fig, use_container_width=True)

    elif dataset == "Weather":
        tab1, tab2, tab3 = st.tabs(["Wind", "Temperature", "Cloud & Solar"])
        with tab1:
            fig = go.Figure()
            if "windspeed_100m" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["windspeed_100m"],
                    name="100m", line=dict(color=FORECAST, width=1.5)))
            if "windspeed_10m" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["windspeed_10m"],
                    name="10m", line=dict(color=NEUTRAL, width=1, dash="dot")))
            chart_layout(fig, "Wind speed", "m/s", height=420, show_title=True)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = go.Figure(go.Scatter(
                x=df.index, y=df["temperature"],
                name="Temperature", fill="tozeroy",
                line=dict(color=WARNING, width=1.5),
                fillcolor="rgba(255,165,2,0.08)"))
            chart_layout(fig, "Temperature", "°C", height=420, show_title=True)
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            fig = go.Figure()
            if "cloudcover" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["cloudcover"],
                    name="Cloud cover (%)", line=dict(color=NEUTRAL, width=1.5)))
            if "solar_radiation" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["solar_radiation"],
                    name="Solar radiation (W/m²)",
                    line=dict(color=WARNING, width=1.5), yaxis="y2"))
            fig.update_layout(
                paper_bgcolor="white", plot_bgcolor="#fafbfc",
                yaxis2=dict(title="W/m²", overlaying="y", side="right"),
                template=None, height=420,
                margin=dict(l=12, r=12, t=40, b=12))
            st.plotly_chart(fig, use_container_width=True)

    elif dataset == "Generation":
        tab1, tab2 = st.tabs(["Generation mix", "Load"])
        with tab1:
            fig = go.Figure()
            if "wind_gen_mw" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["wind_gen_mw"],
                    name="Wind (MW)", line=dict(color=PRIMARY, width=1.5)))
            if "nuclear_gen_mw" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["nuclear_gen_mw"],
                    name="Nuclear (MW)", line=dict(color=WARNING, width=1.5)))
            chart_layout(fig, "Generation", "MW", height=420, show_title=True)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            if "load_mw" in df.columns:
                fig = go.Figure(go.Scatter(
                    x=df.index, y=df["load_mw"],
                    name="Load (MW)", fill="tozeroy",
                    line=dict(color=DANGER, width=1.5),
                    fillcolor="rgba(255,71,87,0.06)"))
                chart_layout(fig, "Total load SE3", "MW", height=420, show_title=True)
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Raw data table"):
        st.dataframe(df.round(2), use_container_width=True)
        csv = df.round(2).reset_index().to_csv(index=False)
        st.download_button("⬇️ Download CSV", data=csv,
            file_name=f"se3_{dataset.lower()}_{from_date}_{to_date}.csv",
            mime="text/csv")


# ── Page: Ask SE3 ─────────────────────────────────────────────────────────────

elif page == "💬 Ask SE3":
    page_header("AI assistant", "Ask SE3",
                "Natural language questions about the SE3 market")

    import httpx

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
        question  = st.text_input("Or type your own question:",
                                  value=st.session_state.se3_question)
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
                        st.metric("Current Price",
                                  f"{data['current_price_eur_mwh']:.1f} EUR/MWh")
                with col_b:
                    if data.get("forecast_p50_next_hour") is not None:
                        st.metric("Forecast (next hour)",
                                  f"{data['forecast_p50_next_hour']:.1f} EUR/MWh")
                with col_c:
                    if data.get("forecast_delta") is not None:
                        st.metric("Forecast vs Current",
                                  f"{data['forecast_delta']:+.1f} EUR/MWh")
                with st.expander("Data sources used"):
                    st.write("**Tools called:**", ", ".join(data.get("tools_called", [])))
                    if data.get("tools_failed"):
                        st.warning(f"Tools that failed: {', '.join(data['tools_failed'])}")
                    st.write("**Sources:**", ", ".join(data.get("sources", [])))
            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="border-top:1px solid #e8ecf0;margin-top:3rem;
            padding-top:1rem;display:flex;justify-content:space-between;
            align-items:center;">
    <span style="font-size:11px;color:#a0aab4;">
        Built by
        <a href="https://github.com/surajsingh108" target="_blank"
           style="color:#3742fa;text-decoration:none;">Suraj Singh</a>
    </span>
    <span style="font-size:11px;color:#c8cfd8;">
        Data:
        <a href="https://transparency.entsoe.eu" target="_blank"
           style="color:#a0aab4;text-decoration:none;">ENTSO-E</a> ·
        <a href="https://open-meteo.com" target="_blank"
           style="color:#a0aab4;text-decoration:none;">Open-Meteo</a> ·
        <a href="https://opendata.esett.com" target="_blank"
           style="color:#a0aab4;text-decoration:none;">eSett</a>
    </span>
</div>
""", unsafe_allow_html=True)
