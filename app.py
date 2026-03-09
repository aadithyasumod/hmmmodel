"""
app.py  —  NSE Regime-Based Trading Scanner
Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from universe_loader import UNIVERSE_OPTIONS
from scanner import scan_universe
from backtester import BacktestResult
from hmm_engine import REGIME_COLORS
from indicators import CONFIRMATION_LABELS, CONFIRMATION_COLS

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE Regime Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS  (dark terminal / quant-finance aesthetic)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0A0E1A;
    color: #C8D0E0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0D1120;
    border-right: 1px solid #1E2740;
  }
  section[data-testid="stSidebar"] * {
    color: #C8D0E0 !important;
  }

  /* Main background */
  .main .block-container {
    background: #0A0E1A;
    padding-top: 1.5rem;
    max-width: 1400px;
  }

  /* Header strip */
  .app-header {
    background: linear-gradient(135deg, #0D1120 0%, #111827 100%);
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 1.2rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #4FC3F7;
    letter-spacing: 0.04em;
  }
  .app-subtitle {
    font-size: 0.8rem;
    color: #5C7A9F;
    margin-top: 0.2rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Metric cards */
  .metric-card {
    background: #0D1120;
    border: 1px solid #1E2740;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #2C4A7C; }
  .metric-label {
    font-size: 0.7rem;
    color: #5C7A9F;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'IBM Plex Mono', monospace;
  }
  .metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    margin: 0.2rem 0;
  }
  .metric-sub {
    font-size: 0.7rem;
    color: #5C7A9F;
  }
  .green  { color: #00E676; }
  .red    { color: #FF5252; }
  .amber  { color: #FFD740; }
  .blue   { color: #4FC3F7; }
  .white  { color: #E8EDF5; }

  /* Signal badge */
  .signal-long {
    display: inline-block;
    background: #003300;
    border: 1.5px solid #00C853;
    color: #00E676;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.3rem 1rem;
    border-radius: 4px;
    letter-spacing: 0.08em;
  }
  .signal-cash {
    display: inline-block;
    background: #1A1A00;
    border: 1.5px solid #FFD740;
    color: #FFD740;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.3rem 1rem;
    border-radius: 4px;
    letter-spacing: 0.08em;
  }
  .regime-bull  { color: #00E676; font-weight: 600; }
  .regime-bear  { color: #FF5252; font-weight: 600; }
  .regime-other { color: #90A4AE; font-weight: 600; }

  /* Confirmation pills */
  .conf-pass {
    display: inline-block;
    background: #003300;
    border: 1px solid #00C853;
    color: #00E676;
    font-size: 0.72rem;
    padding: 0.15rem 0.6rem;
    border-radius: 12px;
    margin: 2px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .conf-fail {
    display: inline-block;
    background: #1A0000;
    border: 1px solid #B71C1C;
    color: #EF5350;
    font-size: 0.72rem;
    padding: 0.15rem 0.6rem;
    border-radius: 12px;
    margin: 2px;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Section divider */
  .section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #5C7A9F;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #1E2740;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* Dataframe overrides */
  .stDataFrame { background: #0D1120 !important; }

  /* Buttons */
  .stButton > button {
    background: #0D2137;
    border: 1.5px solid #1E6FA5;
    color: #4FC3F7;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: #1E3A5F;
    border-color: #4FC3F7;
    color: #E8F4FD;
  }

  /* Progress bar */
  .stProgress > div > div { background-color: #1E6FA5; }

  /* Selectbox */
  .stSelectbox > div > div {
    background: #0D1120;
    border-color: #1E2740;
    color: #C8D0E0;
  }

  /* Trade table row colours */
  .win-row  { background: rgba(0,200,83,0.06) !important; }
  .loss-row { background: rgba(255,82,82,0.06) !important; }

  /* Ticker header */
  .ticker-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #4FC3F7;
    letter-spacing: 0.06em;
  }
  .ticker-sub {
    font-size: 0.75rem;
    color: #5C7A9F;
    font-family: 'IBM Plex Mono', monospace;
  }

  hr { border-color: #1E2740; }

  /* Hide Streamlit chrome */
  #MainMenu { visibility: hidden; }
  footer     { visibility: hidden; }
  header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Chart builder
# ──────────────────────────────────────────────────────────────────────────────

def _regime_bg_shapes(df: pd.DataFrame) -> list[dict]:
    """
    Generates plotly shape dicts for coloured regime background bands.
    Consecutive same-regime days are merged into single rectangles (fast).
    """
    shapes = []
    regime_col = df["Regime"].values
    dates = df.index.tolist()

    FILL = {
        "Bull Run":   "rgba(0,200,83,0.10)",
        "Bear/Crash": "rgba(255,23,68,0.12)",
        "Neutral":    "rgba(144,164,174,0.05)",
    }

    start_i = 0
    cur_reg = regime_col[0]

    for i in range(1, len(regime_col)):
        if regime_col[i] != cur_reg or i == len(regime_col) - 1:
            end_i = i if regime_col[i] != cur_reg else i + 1
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=dates[start_i], x1=dates[min(end_i, len(dates) - 1)],
                y0=0, y1=1,
                fillcolor=FILL.get(cur_reg, FILL["Neutral"]),
                line=dict(width=0),
                layer="below",
            ))
            start_i = i
            cur_reg = regime_col[i]

    return shapes


def build_chart(result: BacktestResult) -> go.Figure:
    df = result.df

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=["", "", ""],
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing=dict(line=dict(color="#00C853", width=1), fillcolor="#00C853"),
        decreasing=dict(line=dict(color="#FF1744", width=1), fillcolor="#FF1744"),
    ), row=1, col=1)

    # EMA 50 & 200
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA50"],
        name="EMA 50", line=dict(color="#FFD740", width=1.2, dash="dot"),
        opacity=0.8,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA200"],
        name="EMA 200", line=dict(color="#FF9100", width=1.5),
        opacity=0.8,
    ), row=1, col=1)

    # Trade entry/exit markers
    if not result.trades.empty and "Entry Date" in result.trades.columns:
        for _, trade in result.trades.iterrows():
            try:
                entry_y = df.loc[trade["Entry Date"], "Close"] if trade["Entry Date"] in df.index else None
                exit_y  = df.loc[trade["Exit Date"],  "Close"] if trade["Exit Date"]  in df.index else None

                if entry_y is not None:
                    fig.add_trace(go.Scatter(
                        x=[trade["Entry Date"]], y=[entry_y],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color="#00E676"),
                        name="Entry", showlegend=False,
                        hovertemplate=f"ENTRY ₹{trade['Entry Price']:.2f}<extra></extra>",
                    ), row=1, col=1)

                if exit_y is not None:
                    color = "#FF5252" if trade.get("PnL (₹)", 0) < 0 else "#FFD740"
                    fig.add_trace(go.Scatter(
                        x=[trade["Exit Date"]], y=[exit_y],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=10, color=color),
                        name="Exit", showlegend=False,
                        hovertemplate=f"EXIT ₹{trade['Exit Price']:.2f}<extra></extra>",
                    ), row=1, col=1)
            except Exception:
                pass

    # ── Volume bar chart ──────────────────────────────────────────────────────
    vol_colors = ["#00C853" if c >= o else "#FF1744"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=vol_colors, opacity=0.7,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Volume_SMA20"],
        name="Vol SMA20", line=dict(color="#4FC3F7", width=1.2),
    ), row=2, col=1)

    # ── Portfolio value vs Buy & Hold ─────────────────────────────────────────
    capital = result.portfolio.iloc[0]
    bh_series = (df["Close"] / df["Close"].iloc[0]) * capital

    fig.add_trace(go.Scatter(
        x=result.portfolio.index, y=result.portfolio,
        name="Strategy", line=dict(color="#4FC3F7", width=2),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.05)",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=bh_series.index, y=bh_series,
        name="Buy & Hold", line=dict(color="#90A4AE", width=1.5, dash="dash"),
    ), row=3, col=1)

    # ── Regime background shapes ──────────────────────────────────────────────
    shapes = _regime_bg_shapes(df)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        shapes=shapes,
        plot_bgcolor="#0A0E1A",
        paper_bgcolor="#0A0E1A",
        font=dict(family="IBM Plex Mono", color="#C8D0E0", size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", x=0, y=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=700,
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor="#0F1829", showgrid=True,
        zeroline=False,
        showspikes=True, spikecolor="#2C4A7C",
        spikethickness=1,
    )
    fig.update_yaxes(
        gridcolor="#0F1829", showgrid=True,
        zeroline=False, tickformat=",.0f",
    )
    fig.update_yaxes(tickformat=".2s", row=2, col=1)
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f", row=3, col=1)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _signed(val: float, fmt: str = ".1f") -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:{fmt}}"


def _color_class(val: float, invert: bool = False) -> str:
    # For max_drawdown (invert=True): colour by severity, not just sign
    if invert:
        if val >= -10: return "green"   # mild drawdown
        if val >= -25: return "amber"   # moderate drawdown
        return "red"                    # severe drawdown
    if val > 0:  return "green"
    if val < 0:  return "red"
    return "white"


def _metric_card(label: str, value: str, sub: str = "", color: str = "white") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {color}">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>
    """


def _signal_badge(signal: str) -> str:
    cls = "signal-long" if signal == "LONG" else "signal-cash"
    icon = "▲ LONG" if signal == "LONG" else "◆ CASH"
    return f'<span class="{cls}">{icon}</span>'


def _regime_badge(regime: str) -> str:
    if regime == "Bull Run":
        return f'<span class="regime-bull">● Bull Run</span>'
    if regime == "Bear/Crash":
        return f'<span class="regime-bear">● Bear/Crash</span>'
    return f'<span class="regime-other">● {regime}</span>'


def _confirmation_pills(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    pills = []
    for col, label in CONFIRMATION_LABELS.items():
        passed = bool(last.get(col, False))
        cls    = "conf-pass" if passed else "conf-fail"
        icon   = "✓" if passed else "✗"
        pills.append(f'<span class="{cls}">{icon} {label}</span>')
    return " ".join(pills)


# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────

if "scan_done"    not in st.session_state: st.session_state.scan_done    = False
if "summary_df"   not in st.session_state: st.session_state.summary_df   = pd.DataFrame()
if "all_results"  not in st.session_state: st.session_state.all_results  = []
if "selected_idx" not in st.session_state: st.session_state.selected_idx = 0


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
         color:#4FC3F7; font-weight:600; margin-bottom:0.2rem;">
    📡 NSE REGIME SCANNER
    </div>
    <div style="font-size:0.7rem; color:#5C7A9F; margin-bottom:1.5rem;
         font-family:'IBM Plex Mono',monospace;">
    HMM · 7 States · 8 Confirmations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Universe**")
    universe_choice = st.selectbox(
        "Universe", list(UNIVERSE_OPTIONS.keys()), index=1,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Strategy Parameters**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="font-size:0.72rem;color:#5C7A9F;">Min Confirms</div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="font-size:1rem;color:#C8D0E0;">7 / 8</div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="font-size:0.72rem;color:#5C7A9F;">Cooldown</div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="font-size:1rem;color:#C8D0E0;">2 trading days</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Capital per Stock**")
    st.markdown('<div style="font-size:1.05rem;color:#4FC3F7;font-family:\'IBM Plex Mono\';">₹1,00,000</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Exit Rule**")
    st.markdown("""
    <div style="font-size:0.75rem;color:#5C7A9F;line-height:1.6;">
    ◆ Exit immediately when HMM regime flips to <span style="color:#FF5252;">Bear/Crash</span><br>
    ◆ 20% circuit breaker cap on exit price<br>
    ◆ Brokerage: ₹20/order + 0.1% STT (sell)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Scan button
    run_scan = st.button("⚡  RUN SCAN", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem;color:#3A4A60;line-height:1.7;font-family:'IBM Plex Mono',monospace;">
    Data: NSE daily via yfinance<br>
    Model: GaussianHMM (7 states)<br>
    Features: Returns · Range · Vol-Z<br>
    Period: 2 years · Daily candles
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <div>
    <div class="app-title">📡 NSE REGIME SCANNER</div>
    <div class="app-subtitle">Hidden Markov Model · 7 States · 8-Confirmation Voting · NSE Daily Data</div>
  </div>
  <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#5C7A9F; text-align:right;">
    Strategy: Regime-Based Entry<br>
    Exit: Bear/Crash Flip + 2-Day Cooldown
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Scanner execution
# ──────────────────────────────────────────────────────────────────────────────

if run_scan:
    st.session_state.scan_done    = False
    st.session_state.summary_df   = pd.DataFrame()
    st.session_state.all_results  = []
    st.session_state.selected_idx = 0

    # Load universe
    with st.spinner("Loading NSE universe list…"):
        loader = UNIVERSE_OPTIONS[universe_choice]
        tickers = loader()

    total = len(tickers)
    st.info(f"🔍 Scanning **{total}** tickers — this may take a few minutes…", icon="⏳")

    dl_text = st.empty()
    dl_bar  = st.progress(0)
    pr_text = st.empty()
    pr_bar  = st.progress(0)

    def on_dl(done, total_):
        pct = done / max(total_, 1)
        dl_bar.progress(pct)
        dl_text.markdown(
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;color:#5C7A9F;">'
            f'📥 Downloading data… {done}/{total_} tickers</span>',
            unsafe_allow_html=True,
        )

    def on_pr(done, total_):
        pct = done / max(total_, 1)
        pr_bar.progress(pct)
        pr_text.markdown(
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;color:#5C7A9F;">'
            f'⚙️  Processing HMM + Backtest… {done}/{total_}</span>',
            unsafe_allow_html=True,
        )

    t0 = time.time()
    summary_df, all_results = scan_universe(
        tickers,
        on_download_progress=on_dl,
        on_process_progress=on_pr,
    )
    elapsed = time.time() - t0

    dl_text.empty(); dl_bar.empty()
    pr_text.empty(); pr_bar.empty()

    st.session_state.scan_done   = True
    st.session_state.summary_df  = summary_df
    st.session_state.all_results = all_results

    n_valid  = sum(1 for _ in all_results)
    n_long   = sum(1 for r in all_results if r.current_signal == "LONG")
    st.success(
        f"✅ Scan complete in **{elapsed:.0f}s** · "
        f"{n_valid} stocks analysed · "
        f"**{n_long} LONG signals** found",
        icon="✅",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Results — Top 20 table
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.scan_done and not st.session_state.summary_df.empty:
    summary = st.session_state.summary_df
    results: list[BacktestResult] = st.session_state.all_results

    n_total = len(summary)
    n_long  = (summary["Signal"] == "LONG").sum()
    st.markdown(
        f'<div class="section-title">ALL RANKED STOCKS ({n_total} analysed · {n_long} LONG signals)</div>',
        unsafe_allow_html=True,
    )

    # Style the DataFrame
    def _style_table(df: pd.DataFrame) -> pd.DataFrame.style:
        def row_color(row):
            styles = [""] * len(row)
            if row["Signal"] == "LONG":
                styles = ["background-color: rgba(0,200,83,0.05)"] * len(row)
            return styles

        def col_color(val, col_name):
            if col_name in ("Total Return %", "Alpha %", "Win Rate %"):
                if isinstance(val, (int, float)):
                    return "color: #00E676" if val > 0 else "color: #FF5252"
            if col_name == "Max Drawdown %":
                if isinstance(val, (int, float)):
                    return "color: #FF5252" if val < -10 else "color: #FFD740"
            return ""

        styled = df.style.apply(row_color, axis=1)
        for c in ("Total Return %", "Alpha %", "Win Rate %", "Max Drawdown %"):
            if c in df.columns:
                styled = styled.map(lambda v, cn=c: col_color(v, cn), subset=[c])

        styled = styled.format({
            "Total Return %": "{:+.2f}%",
            "Alpha %":        "{:+.2f}%",
            "B&H Return %":   "{:+.2f}%",
            "Win Rate %":     "{:.1f}%",
            "Max Drawdown %": "{:.2f}%",
            "Sharpe":         "{:.3f}",
            "Score":          "{:.3f}",
        })

        styled = styled.set_properties(**{
            "font-family": "IBM Plex Mono, monospace",
            "font-size":   "0.78rem",
            "color":       "#C8D0E0",
            "background":  "#0D1120",
        })
        return styled

    st.dataframe(
        _style_table(summary),
        use_container_width=True,
        height=int(min(len(summary) * 38 + 55, 1800)),
        hide_index=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Detail view — select a stock from the top 20
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.scan_done and st.session_state.all_results:
    results = st.session_state.all_results
    summary = st.session_state.summary_df

    st.markdown('<div class="section-title">STOCK DEEP-DIVE</div>', unsafe_allow_html=True)

    ticker_labels = [r.ticker.replace(".NS", "") for r in results]
    selected_label = st.selectbox(
        "Select stock to analyse:",
        ticker_labels,
        index=st.session_state.selected_idx,
        key="stock_select",
    )
    selected_idx   = ticker_labels.index(selected_label)
    st.session_state.selected_idx = selected_idx
    sel: BacktestResult = results[selected_idx]

    # ── Header row ────────────────────────────────────────────────────────────
    hcol1, hcol2, hcol3 = st.columns([3, 2, 5])

    with hcol1:
        last_close = sel.df["Close"].iloc[-1]
        prev_close = sel.df["Close"].iloc[-2]
        chg_pct    = (last_close - prev_close) / prev_close * 100
        chg_class  = "green" if chg_pct >= 0 else "red"
        st.markdown(f"""
        <div class="ticker-header">{sel.ticker.replace('.NS','')}.NS</div>
        <div class="ticker-sub">
          ₹{last_close:,.2f}
          <span class="{chg_class}">&nbsp;{_signed(chg_pct, '.2f')}% today</span>
        </div>
        """, unsafe_allow_html=True)

    with hcol2:
        sig_html = _signal_badge(sel.current_signal)
        reg_html = _regime_badge(sel.current_regime)
        st.markdown(f"""
        {sig_html}&nbsp;&nbsp;{reg_html}
        <div style="margin-top:0.5rem;font-size:0.72rem;color:#5C7A9F;font-family:'IBM Plex Mono',monospace;">
          {sel.confirmations_now}/8 confirmations passing
        </div>
        """, unsafe_allow_html=True)

    with hcol3:
        pills = _confirmation_pills(sel.df)
        st.markdown(f'<div style="margin-top:0.3rem;">{pills}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    metrics = [
        (m1, "TOTAL RETURN", f"{_signed(sel.total_return, '.1f')}%",
         f"on ₹1,00,000", _color_class(sel.total_return)),
        (m2, "ALPHA",        f"{_signed(sel.alpha, '.1f')}%",
         "vs Buy & Hold", _color_class(sel.alpha)),
        (m3, "BUY & HOLD",   f"{_signed(sel.bh_return, '.1f')}%",
         "2-year return", _color_class(sel.bh_return)),
        (m4, "WIN RATE",     f"{sel.win_rate:.1f}%",
         f"{sel.n_trades} trades", "green" if sel.win_rate >= 50 else "amber"),
        (m5, "MAX DRAWDOWN", f"{sel.max_drawdown:.1f}%",
         "peak-to-trough", _color_class(sel.max_drawdown, invert=True)),
        (m6, "SHARPE",       f"{sel.sharpe:.3f}",
         "annualised", "green" if sel.sharpe >= 1 else "amber" if sel.sharpe >= 0 else "red"),
    ]
    for col, label, val, sub, color in metrics:
        with col:
            st.markdown(_metric_card(label, val, sub, color), unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">PRICE CHART · VOLUME · PORTFOLIO EQUITY</div>',
                unsafe_allow_html=True)

    chart_note = (
        '<span style="font-size:0.72rem;color:#5C7A9F;font-family:\'IBM Plex Mono\',monospace;">'
        '🟢 Green background = Bull Run regime &nbsp;|&nbsp; 🔴 Red = Bear/Crash &nbsp;|&nbsp; '
        '▲ Entry marker &nbsp;▼ Exit marker</span>'
    )
    st.markdown(chart_note, unsafe_allow_html=True)

    with st.spinner("Rendering chart…"):
        fig = build_chart(sel)
    st.plotly_chart(fig, use_container_width=True)

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">TRADE LOG</div>', unsafe_allow_html=True)

    if sel.trades.empty:
        st.markdown(
            '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.82rem;color:#5C7A9F;">'
            'No completed trades in this period.</span>',
            unsafe_allow_html=True,
        )
    else:
        trades = sel.trades.copy()
        trades["Entry Date"] = pd.to_datetime(trades["Entry Date"]).dt.strftime("%d %b %Y")
        trades["Exit Date"]  = pd.to_datetime(trades["Exit Date"]).dt.strftime("%d %b %Y")

        def style_trades(df):
            def row_style(row):
                pnl = row.get("PnL (₹)", 0)
                bg = "rgba(0,200,83,0.06)" if pnl > 0 else "rgba(255,82,82,0.06)"
                return [f"background-color:{bg}"] * len(row)

            styled = df.style.apply(row_style, axis=1)
            styled = styled.format({
                "Entry Price": "₹{:,.2f}",
                "Exit Price":  "₹{:,.2f}",
                "PnL (₹)":    "₹{:+,.2f}",
                "Return %":   "{:+.2f}%",
                "Duration (d)": "{:.0f}d",
            })
            styled = styled.map(
                lambda v: "color:#00E676" if isinstance(v, (int, float)) and v > 0
                     else "color:#FF5252" if isinstance(v, (int, float)) and v < 0
                     else "",
                subset=["PnL (₹)", "Return %"],
            )
            styled = styled.set_properties(**{
                "font-family": "IBM Plex Mono, monospace",
                "font-size":   "0.78rem",
                "color":       "#C8D0E0",
            })
            return styled

        st.dataframe(style_trades(trades), use_container_width=True, hide_index=True)

        # Summary footer
        total_pnl  = sel.trades["PnL (₹)"].sum()
        n_wins     = (sel.trades["PnL (₹)"] > 0).sum()
        n_losses   = (sel.trades["PnL (₹)"] < 0).sum()
        avg_win    = sel.trades.loc[sel.trades["PnL (₹)"] > 0, "PnL (₹)"].mean() if n_wins else 0
        avg_loss   = sel.trades.loc[sel.trades["PnL (₹)"] < 0, "PnL (₹)"].mean() if n_losses else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        pnl_color = "#00E676" if total_pnl >= 0 else "#FF5252"
        st.markdown(f"""
        <div style="display:flex;gap:2rem;margin-top:0.8rem;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;">
          <span style="color:#5C7A9F;">Total P&L:</span>
          <span style="color:{pnl_color};font-weight:600;">₹{total_pnl:+,.2f}</span>
          <span style="color:#5C7A9F;">Wins:</span>
          <span style="color:#00E676;">{n_wins}</span>
          <span style="color:#5C7A9F;">Losses:</span>
          <span style="color:#FF5252;">{n_losses}</span>
          <span style="color:#5C7A9F;">Avg Win:</span>
          <span style="color:#00E676;">₹{avg_win:+,.2f}</span>
          <span style="color:#5C7A9F;">Avg Loss:</span>
          <span style="color:#FF5252;">₹{avg_loss:+,.2f}</span>
          <span style="color:#5C7A9F;">Reward:Risk:</span>
          <span style="color:#FFD740;">{rr:.2f}x</span>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Empty-state prompt
# ──────────────────────────────────────────────────────────────────────────────

elif not st.session_state.scan_done:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#3A4A60;">
      <div style="font-size:3rem; margin-bottom:1rem;">📡</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem; color:#4FC3F7; margin-bottom:0.5rem;">
        SELECT A UNIVERSE AND CLICK <em>RUN SCAN</em>
      </div>
      <div style="font-size:0.8rem; color:#3A4A60; max-width:480px; margin:0 auto; line-height:1.8;">
        The scanner will:<br>
        1. Download daily OHLCV for all NSE stocks<br>
        2. Fit a 7-state HMM to identify market regimes<br>
        3. Score 8 confirmation indicators per stock<br>
        4. Backtest the strategy and rank the top 20
      </div>
    </div>
    """, unsafe_allow_html=True)
