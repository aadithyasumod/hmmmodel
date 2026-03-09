"""
data_loader.py
Efficient batch OHLCV downloader for NSE (.NS suffix) tickers via yfinance.
Uses chunked batch downloads (50 tickers per call) for speed.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
from typing import Optional

MIN_BARS = 252          # ~1 year of daily bars — minimum for HMM + indicators
BATCH_SIZE = 50         # yfinance handles ~50 tickers per batch well
DOWNLOAD_PERIOD = "2y"  # 2 years of daily data
SLEEP_BETWEEN_BATCHES = 0.5   # seconds — be polite to Yahoo servers


def _clean_df(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Validate and clean a single-ticker OHLCV DataFrame."""
    if raw is None or raw.empty:
        return None

    # Flatten MultiIndex columns if present (happens in batch downloads)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(raw.columns)):
        return None

    df = raw[list(required)].copy()
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df["Volume"] = df["Volume"].fillna(0)
    df = df[df["Close"] > 0]

    if len(df) < MIN_BARS:
        return None

    return df


def fetch_single(ticker: str) -> Optional[pd.DataFrame]:
    """Download data for a single ticker. Returns None on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                ticker,
                period=DOWNLOAD_PERIOD,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
        return _clean_df(raw)
    except Exception:
        return None


def fetch_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Batch-download a list of tickers. Returns a dict {ticker: df}.
    Tickers with insufficient data are silently dropped.
    """
    if not tickers:
        return {}

    result: dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), BATCH_SIZE):
        chunk = tickers[i : i + BATCH_SIZE]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    chunk,
                    period=DOWNLOAD_PERIOD,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                )

            if raw.empty:
                continue

            if len(chunk) == 1:
                # Single ticker — yfinance returns flat columns
                df = _clean_df(raw)
                if df is not None:
                    result[chunk[0]] = df
            else:
                # Multiple tickers — MultiIndex (metric, ticker)
                for ticker in chunk:
                    try:
                        df = _clean_df(raw[ticker])
                        if df is not None:
                            result[ticker] = df
                    except (KeyError, TypeError):
                        pass

        except Exception:
            pass

        if i + BATCH_SIZE < len(tickers):
            time.sleep(SLEEP_BETWEEN_BATCHES)

    return result
