"""
backtester.py
Simulates the regime-based strategy on a single NSE stock.

Rules:
  • Entry  : HMM = Bull Run  AND  Confirmations >= 7 out of 8
  • Exit   : HMM flips to Bear/Crash  (regime-based hard stop)
  • Cooldown: 2 trading days after any exit before re-entry allowed
  • Capital : ₹1,00,000 per stock (configurable)
  • Costs   : ₹20 flat per order (Zerodha) + 0.1% STT on sell side
  • Circuit  : If stock hits a ±20% circuit on exit day, exit at 20% limit price
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from indicators import add_indicators, add_confirmations
from hmm_engine import fit_hmm, build_regime_series, state_to_label

STARTING_CAPITAL  = 100_000   # ₹1 lakh
BROKERAGE_PER_LEG = 20.0      # Zerodha flat ₹20
STT_SELL_RATE     = 0.001     # 0.1% on sell turnover
CIRCUIT_LIMIT     = 0.20      # NSE 20% circuit breaker
COOLDOWN_DAYS     = 2         # trading-day cooldown after any exit
MIN_CONFIRMATIONS = 7


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    ticker:          str
    df:              pd.DataFrame      # enriched OHLCV + indicators + regime
    trades:          pd.DataFrame      # trade log
    portfolio:       pd.Series         # daily portfolio value
    total_return:    float             # %
    bh_return:       float             # buy-and-hold %
    alpha:           float             # total_return − bh_return
    win_rate:        float             # % of profitable trades
    max_drawdown:    float             # % (negative)
    sharpe:          float
    current_signal:  str               # "LONG" | "CASH"
    current_regime:  str
    confirmations_now: int
    n_trades:        int
    score:           float             # composite ranking score


# ──────────────────────────────────────────────────────────────────────────────
# Transaction cost helper
# ──────────────────────────────────────────────────────────────────────────────

def _transaction_cost(price: float, shares: int, side: str) -> float:
    brokerage = BROKERAGE_PER_LEG
    stt = (price * shares * STT_SELL_RATE) if side == "sell" else 0.0
    return brokerage + stt


# ──────────────────────────────────────────────────────────────────────────────
# Core simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(
    df_raw: pd.DataFrame,
    ticker: str,
    capital: float = STARTING_CAPITAL,
) -> Optional[BacktestResult]:
    """
    Full backtest on daily OHLCV data.
    Returns None if data is insufficient or HMM fails.
    """

    # ── 1. Indicators ─────────────────────────────────────────────────────────
    df = add_indicators(df_raw)
    df = add_confirmations(df)
    df.dropna(inplace=True)

    if len(df) < 200:
        return None

    # ── 2. HMM regime ─────────────────────────────────────────────────────────
    try:
        hmm_result = fit_hmm(df)
    except Exception:
        return None

    regime_series = build_regime_series(hmm_result)
    df = df.loc[regime_series.index]          # align to HMM-valid rows
    df["Regime"]    = regime_series.values
    df["HMM_State"] = hmm_result.states

    df["is_bull"] = df["HMM_State"] == hmm_result.bull_state
    df["is_bear"] = df["HMM_State"] == hmm_result.bear_state

    # ── 3. Entry signal ───────────────────────────────────────────────────────
    df["Entry_Signal"] = df["is_bull"] & (df["Confirmations"] >= MIN_CONFIRMATIONS)

    # ── 4. Simulation loop ────────────────────────────────────────────────────
    trades_list: list[dict] = []
    portfolio_vals: list[float] = []
    cash        = capital
    in_position = False
    entry_price = 0.0
    entry_date  = None
    shares      = 0
    last_exit_idx: Optional[int] = None   # integer location of last exit

    dates  = df.index.tolist()
    closes = df["Close"].values
    bears  = df["is_bear"].values
    sigs   = df["Entry_Signal"].values

    for i, date in enumerate(dates):
        close = closes[i]

        if in_position:
            # ── Exit condition: regime flips to Bear/Crash ──────────────────
            if bears[i]:
                # Circuit-breaker: cap price move at ±20%
                exit_price = min(close, entry_price * (1 + CIRCUIT_LIMIT))
                exit_price = max(exit_price, entry_price * (1 - CIRCUIT_LIMIT))

                costs = _transaction_cost(exit_price, shares, "sell")
                pnl   = (exit_price - entry_price) * shares - costs - _transaction_cost(entry_price, shares, "buy")
                ret_pct = (exit_price - entry_price) / entry_price * 100

                trades_list.append({
                    "Entry Date":  entry_date,
                    "Exit Date":   date,
                    "Entry Price": round(entry_price, 2),
                    "Exit Price":  round(exit_price,  2),
                    "Shares":      shares,
                    "PnL (₹)":     round(pnl, 2),
                    "Return %":    round(ret_pct, 2),
                    "Exit Reason": "Bear/Crash Regime",
                    "Duration (d)": (date - entry_date).days,
                })

                cash += exit_price * shares - _transaction_cost(exit_price, shares, "sell")
                in_position = False
                last_exit_idx = i
                shares = 0

        else:
            # ── Cooldown check ───────────────────────────────────────────────
            if last_exit_idx is not None and (i - last_exit_idx) < COOLDOWN_DAYS:
                portfolio_vals.append(cash)
                continue

            # ── Entry condition ──────────────────────────────────────────────
            if sigs[i]:
                max_shares = int(capital / close)   # deploy exactly ₹1L of capital
                if max_shares > 0:
                    cost = close * max_shares + _transaction_cost(close, max_shares, "buy")
                    if cost <= cash:
                        cash        -= cost
                        shares       = max_shares
                        entry_price  = close
                        entry_date   = date
                        in_position  = True

        pv = cash + (shares * close if in_position else 0)
        portfolio_vals.append(pv)

    # ── Close any open position at end of period ──────────────────────────────
    if in_position:
        exit_price = closes[-1]
        costs      = _transaction_cost(exit_price, shares, "sell")
        pnl        = (exit_price - entry_price) * shares - costs - _transaction_cost(entry_price, shares, "buy")
        ret_pct    = (exit_price - entry_price) / entry_price * 100
        trades_list.append({
            "Entry Date":  entry_date,
            "Exit Date":   dates[-1],
            "Entry Price": round(entry_price, 2),
            "Exit Price":  round(exit_price,  2),
            "Shares":      shares,
            "PnL (₹)":     round(pnl, 2),
            "Return %":    round(ret_pct, 2),
            "Exit Reason": "End of Period",
            "Duration (d)": (dates[-1] - entry_date).days,
        })

    # Pad portfolio_vals to match df length if any `continue` shortened it
    while len(portfolio_vals) < len(df):
        portfolio_vals.append(portfolio_vals[-1] if portfolio_vals else capital)

    portfolio = pd.Series(portfolio_vals[: len(df)], index=df.index, name="Portfolio")
    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(
        columns=["Entry Date", "Exit Date", "Entry Price", "Exit Price",
                 "Shares", "PnL (₹)", "Return %", "Exit Reason", "Duration (d)"]
    )

    # ── 5. Performance metrics ────────────────────────────────────────────────
    total_return = (portfolio.iloc[-1] - capital) / capital * 100
    bh_return    = (closes[-1] - closes[0]) / closes[0] * 100
    alpha        = total_return - bh_return

    win_rate = 0.0
    if not trades_df.empty and "PnL (₹)" in trades_df.columns:
        win_rate = (trades_df["PnL (₹)"] > 0).mean() * 100

    rolling_max  = portfolio.cummax()
    drawdown     = (portfolio - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    daily_ret = portfolio.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    # ── 6. Current state ──────────────────────────────────────────────────────
    last_row    = df.iloc[-1]
    last_state  = int(last_row["HMM_State"])
    last_regime = state_to_label(last_state, hmm_result.bull_state, hmm_result.bear_state)
    last_conf   = int(last_row["Confirmations"])

    current_signal = "LONG" if (last_regime == "Bull Run" and last_conf >= MIN_CONFIRMATIONS) else "CASH"

    # ── 7. Composite ranking score (higher = better) ──────────────────────────
    score = alpha * 0.40 + win_rate * 0.30 + max_drawdown * (-0.30)

    return BacktestResult(
        ticker=ticker,
        df=df,
        trades=trades_df,
        portfolio=portfolio,
        total_return=round(total_return, 2),
        bh_return=round(bh_return, 2),
        alpha=round(alpha, 2),
        win_rate=round(win_rate, 2),
        max_drawdown=round(max_drawdown, 2),
        sharpe=round(sharpe, 3),
        current_signal=current_signal,
        current_regime=last_regime,
        confirmations_now=last_conf,
        n_trades=len(trades_df),
        score=round(score, 3),
    )
