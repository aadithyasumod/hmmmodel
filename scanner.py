"""
scanner.py
Orchestrates the full universe scan:
  1. Batch-download OHLCV for all tickers
  2. Run HMM + backtest in parallel (ThreadPoolExecutor)
  3. Return top-20 ranked by composite score (LONG signals first)
"""

from __future__ import annotations

import time
import concurrent.futures
from typing import Callable, Optional

import pandas as pd

from data_loader import fetch_batch, BATCH_SIZE
from backtester import run_backtest, BacktestResult, DEFAULT_RISK_PER_TRADE, DEFAULT_ATR_MULTIPLIER

MAX_WORKERS      = 8     # parallel HMM fits
DOWNLOAD_CHUNK   = 50    # tickers per yfinance batch call


def _process_one(
    ticker: str,
    df: pd.DataFrame,
    risk_per_trade: float,
    atr_multiplier: float,
) -> Optional[BacktestResult]:
    """Runs backtest for a single ticker; returns None on any failure."""
    try:
        return run_backtest(df, ticker, risk_per_trade=risk_per_trade, atr_multiplier=atr_multiplier)
    except Exception:
        return None


def scan_universe(
    tickers: list[str],
    on_download_progress: Optional[Callable[[int, int], None]] = None,
    on_process_progress:  Optional[Callable[[int, int], None]] = None,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
) -> tuple[pd.DataFrame, list[BacktestResult]]:
    """
    Full pipeline scan.

    Parameters
    ----------
    tickers               : list of .NS ticker strings
    on_download_progress  : callback(done, total)  — called after each download batch
    on_process_progress   : callback(done, total)  — called after each processed ticker

    Returns
    -------
    summary_df  : top-20 ranked DataFrame for the results table
    all_results : list of BacktestResult objects (for detail views)
    """

    # ── Phase 1: Download ─────────────────────────────────────────────────────
    data_map: dict[str, pd.DataFrame] = {}
    total_tickers = len(tickers)

    for i in range(0, total_tickers, DOWNLOAD_CHUNK):
        chunk = tickers[i : i + DOWNLOAD_CHUNK]
        batch = fetch_batch(chunk)
        data_map.update(batch)
        if on_download_progress:
            on_download_progress(min(i + DOWNLOAD_CHUNK, total_tickers), total_tickers)

    valid_tickers = list(data_map.keys())
    n_valid = len(valid_tickers)

    if n_valid == 0:
        return pd.DataFrame(), []

    # ── Phase 2: Parallel HMM + backtest ─────────────────────────────────────
    results: list[BacktestResult] = []
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_ticker = {
            ex.submit(_process_one, t, data_map[t], risk_per_trade, atr_multiplier): t
            for t in valid_tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            if res is not None:
                results.append(res)
            done += 1
            if on_process_progress:
                on_process_progress(done, n_valid)

    if not results:
        return pd.DataFrame(), []

    # ── Phase 3: Detect cross-listed duplicates ───────────────────────────────
    # A company is "duplicate" if its base symbol appears on both .NS and .BO.
    # In that case label each row with (NSE) / (BSE) so the user can tell them apart.
    base_symbols: dict[str, int] = {}   # base → count of exchanges
    for r in results:
        base = r.ticker.rsplit(".", 1)[0]
        base_symbols[base] = base_symbols.get(base, 0) + 1

    def _ticker_label(ticker: str) -> str:
        base, _, suffix = ticker.rpartition(".")
        exchange = "NSE" if suffix == "NS" else "BSE"
        if base_symbols.get(base, 0) > 1:
            return f"{base} ({exchange})"
        return base

    # ── Phase 4: Rank all results ─────────────────────────────────────────────
    long_results = [r for r in results if r.current_signal == "LONG"]
    cash_results = [r for r in results if r.current_signal == "CASH"]

    long_results.sort(key=lambda r: r.score, reverse=True)
    cash_results.sort(key=lambda r: r.score, reverse=True)

    all_ranked = long_results + cash_results

    rows = []
    for r in all_ranked:
        rows.append({
            "Ticker":         _ticker_label(r.ticker),
            "Signal":         r.current_signal,
            "Regime":         r.current_regime,
            "Confirms (now)": r.confirmations_now,
            "Total Return %": r.total_return,
            "Alpha %":        r.alpha,
            "B&H Return %":   r.bh_return,
            "Win Rate %":     r.win_rate,
            "Max Drawdown %": r.max_drawdown,
            "Sharpe":         r.sharpe,
            "# Trades":       r.n_trades,
            "Score":          r.score,
        })

    summary_df = pd.DataFrame(rows)
    return summary_df, all_ranked
