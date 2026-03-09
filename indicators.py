"""
indicators.py
Computes all 8 confirmation indicators used by the voting system:
  1. RSI (14)
  2. Momentum (20-period price change %)
  3. Volatility (20-period rolling σ of returns, annualised %)
  4. Volume vs 20-period SMA
  5. ADX (14)
  6. Price > EMA 50
  7. Price > EMA 200
  8. MACD > Signal line
"""

import numpy as np
import pandas as pd


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA) — used for ATR and ADX."""
    result = series.copy().astype(float)
    result[:] = np.nan
    first_valid = series.first_valid_index()
    if first_valid is None:
        return result

    # Seed with simple mean of first `period` values
    idx_start = series.index.get_loc(first_valid)
    seed_end = idx_start + period
    if seed_end > len(series):
        return result

    result.iloc[seed_end - 1] = series.iloc[idx_start:seed_end].mean()
    for i in range(seed_end, len(series)):
        result.iloc[i] = (result.iloc[i - 1] * (period - 1) + series.iloc[i]) / period

    return result


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    dm_plus_raw  = high.diff().clip(lower=0)
    dm_minus_raw = (-low.diff()).clip(lower=0)

    # A directional move only counts when it exceeds the opposite
    dm_plus  = dm_plus_raw.where(dm_plus_raw  > dm_minus_raw, 0.0)
    dm_minus = dm_minus_raw.where(dm_minus_raw > dm_plus_raw,  0.0)

    atr      = _wilder_smooth(tr,       period)
    sdi_plus  = 100 * _wilder_smooth(dm_plus,  period) / atr.replace(0, np.nan)
    sdi_minus = 100 * _wilder_smooth(dm_minus, period) / atr.replace(0, np.nan)

    dx = 100 * (sdi_plus - sdi_minus).abs() / (sdi_plus + sdi_minus).replace(0, np.nan)
    return _wilder_smooth(dx, period)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all indicator columns to a OHLCV DataFrame in-place.
    Returns the same DataFrame (with NaN rows at the top from rolling windows).
    """
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── 1. RSI ────────────────────────────────────────────────────────────────
    df["RSI"] = compute_rsi(close, 14)

    # ── 2. Momentum (20-period % change) ─────────────────────────────────────
    df["Momentum"] = close.pct_change(20) * 100

    # ── 3. Annualised Volatility (%) ──────────────────────────────────────────
    df["Volatility"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # ── 4. Volume vs 20-SMA ───────────────────────────────────────────────────
    df["Volume_SMA20"] = volume.rolling(20).mean()

    # ── 5. ADX ────────────────────────────────────────────────────────────────
    df["ADX"] = compute_adx(high, low, close, 14)

    # ── 6 & 7. EMAs ──────────────────────────────────────────────────────────
    df["EMA50"]  = close.ewm(span=50,  adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # ── 8. MACD & Signal ─────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


CONFIRMATION_COLS = [
    "conf_RSI",        # RSI < 90
    "conf_Momentum",   # Momentum > 1 %
    "conf_Volatility", # Volatility < 6 %
    "conf_Volume",     # Volume > 20-SMA
    "conf_ADX",        # ADX > 25
    "conf_EMA50",      # Close > EMA50
    "conf_EMA200",     # Close > EMA200
    "conf_MACD",       # MACD > Signal
]

CONFIRMATION_LABELS = {
    "conf_RSI":        "40 < RSI < 75",
    "conf_Momentum":   "Momentum > 1%",
    "conf_Volatility": "Volatility < 50%",
    "conf_Volume":     "Vol > SMA20",
    "conf_ADX":        "ADX > 25",
    "conf_EMA50":      "Price > EMA50",
    "conf_EMA200":     "Price > EMA200",
    "conf_MACD":       "MACD > Signal",
}


def add_confirmations(df: pd.DataFrame) -> pd.DataFrame:
    """Adds boolean confirmation columns and a total score column."""
    df = df.copy()
    df["conf_RSI"]        = (df["RSI"] > 40) & (df["RSI"] < 75)
    df["conf_Momentum"]   = df["Momentum"] > 1.0
    df["conf_Volatility"] = df["Volatility"] < 50.0
    df["conf_Volume"]     = df["Volume"]   > df["Volume_SMA20"]
    df["conf_ADX"]        = df["ADX"]      > 25
    df["conf_EMA50"]      = df["Close"]    > df["EMA50"]
    df["conf_EMA200"]     = df["Close"]    > df["EMA200"]
    df["conf_MACD"]       = df["MACD"]     > df["MACD_Signal"]
    df["Confirmations"]   = df[CONFIRMATION_COLS].sum(axis=1)
    return df
