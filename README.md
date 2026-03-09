# 📡 NSE Regime-Based Trading Scanner

A multi-stock research tool for Indian NSE equities using Hidden Markov Models
to detect market regimes and an 8-confirmation voting system to surface trade candidates.

---

## Architecture

```
regime_trader/
├── app.py              ← Streamlit dashboard (run this)
├── universe_loader.py  ← NSE stock universe (NIFTY 50 / 500 / All NSE)
├── data_loader.py      ← Batch yfinance OHLCV downloader
├── hmm_engine.py       ← 7-state GaussianHMM, auto-labels Bull/Bear states
├── indicators.py       ← RSI, ADX, EMA, MACD, Momentum, Volatility
├── backtester.py       ← Full simulation with ₹1L capital per stock
├── scanner.py          ← Parallel scan, produces top-20 ranked results
└── requirements.txt
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## Strategy Logic

### Regime Detection (HMM Core)
- **Model**: `hmmlearn.GaussianHMM` with **7 components**, `diag` covariance
- **Features** (3):
  1. Daily returns
  2. Range ratio  `(High − Low) / Close` — volatility proxy
  3. Volume z-score  (20-period rolling)
- **Auto-labelling**: Bull Run = highest mean-return state, Bear/Crash = lowest
- All features are winsorised at 1st/99th percentile to handle NSE small-cap outliers

### Entry: 8-Confirmation Voting System
Entry only when **Regime = Bull Run AND ≥ 7 of 8 conditions pass**:

| # | Condition | Threshold |
|---|-----------|-----------|
| 1 | RSI | < 90 (not overbought) |
| 2 | Momentum | > 1% (positive 20-day trend) |
| 3 | Volatility | < 6% (annualised, not choppy) |
| 4 | Volume | > 20-period SMA |
| 5 | ADX | > 25 (trending, not ranging) |
| 6 | Price vs EMA 50 | Price > EMA 50 |
| 7 | Price vs EMA 200 | Price > EMA 200 |
| 8 | MACD | MACD line > Signal line |

### Exit Rules
- **Regime flip**: Exit immediately if HMM detects Bear/Crash state
- **Cooldown**: 2 trading days (≈ 48hrs) after any exit before re-entry
- **Circuit cap**: Exit price capped at ±20% of entry (NSE circuit breaker)

### Risk & Costs
- **Capital**: ₹1,00,000 per stock (₹1 lakh)
- **Brokerage**: ₹20 flat per order (Zerodha model)
- **STT**: 0.1% on sell-side turnover

---

## Universe Options

| Option | Stocks | Time |
|--------|--------|------|
| NIFTY 50 | ~50 | ~1 min |
| NIFTY 500 ⭐ | ~500 | ~5 min |
| All NSE | ~1,800 | ~15 min |

> **Recommended**: Start with NIFTY 500. It covers the most liquid and data-reliable stocks.
> The full NSE list includes many SME/illiquid stocks where yfinance data may be sparse.

---

## Output

The **Top 20** stocks are ranked by a composite score:
```
Score = (Alpha × 0.40) + (Win Rate × 0.30) + (Max Drawdown × −0.30)
```
LONG signals always surface above CASH signals in the ranking.

### Metrics displayed
- **Total Return %** — strategy equity curve return
- **Alpha %** — outperformance vs Buy & Hold
- **Win Rate %** — % of closed trades in profit
- **Max Drawdown %** — worst peak-to-trough drop
- **Sharpe Ratio** — annualised (daily return / daily σ × √252)

---

## Notes

- This is a **backtesting / research tool**, not a live trading system.
- Past performance does not guarantee future results.
- yfinance data for smaller NSE stocks can have gaps; the scanner silently skips
  tickers with < 252 bars of usable data.
- For live signal generation, integrate the backtester output with the Kite MCP.
