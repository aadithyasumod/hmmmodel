"""
hmm_engine.py
Trains a 7-component GaussianHMM on three market features:
  • Returns          — directional signal
  • Range ratio      — (High-Low)/Close  → realised volatility proxy
  • Volume z-score   — 20-period rolling z-score of volume
Automatically labels:
  • Bull Run state   — state with highest mean return
  • Bear/Crash state — state with lowest mean return
"""

import io
import sys
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from hmmlearn import hmm


N_COMPONENTS = 7
N_ITER       = 150
RANDOM_STATE = 42
MIN_STATE_OBS = 10     # a state must have at least this many observations


@dataclass
class HMMResult:
    model:       hmm.GaussianHMM
    states:      np.ndarray        # aligned to `index`
    index:       pd.DatetimeIndex
    bull_state:  int
    bear_state:  int
    state_means: dict[int, float]  # state → mean daily return


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Returns (X, index) where X has shape (T, 3).
    All three features are winsorised at 1st / 99th percentile to handle
    extreme outliers common in small-cap NSE stocks.
    """
    returns     = df["Close"].pct_change()
    range_ratio = (df["High"] - df["Low"]) / df["Close"]

    vol_mean = df["Volume"].rolling(20).mean()
    vol_std  = df["Volume"].rolling(20).std()
    vol_z    = (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)

    feat_df = pd.DataFrame({
        "returns":     returns,
        "range_ratio": range_ratio,
        "vol_z":       vol_z,
    }).dropna()

    # Winsorise each feature
    for col in feat_df.columns:
        lo, hi = feat_df[col].quantile([0.01, 0.99])
        feat_df[col] = feat_df[col].clip(lo, hi)

    X = feat_df.values.astype(np.float64)
    return X, feat_df.index


# ──────────────────────────────────────────────────────────────────────────────
# Model fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_hmm(df: pd.DataFrame) -> HMMResult:
    """
    Fits the HMM and returns an HMMResult.
    Raises ValueError if training fails or data is insufficient.
    """
    X, index = _build_features(df)

    if len(X) < 200:
        raise ValueError(f"Insufficient data for HMM: {len(X)} bars (need ≥ 200)")

    model = hmm.GaussianHMM(
        n_components=N_COMPONENTS,
        covariance_type="diag",
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        tol=1e-4,
    )

    # Suppress both Python warnings and hmmlearn's direct stderr prints
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _devnull = io.StringIO()
        _old_stderr = sys.stderr
        sys.stderr = _devnull
        try:
            model.fit(X)

            # Fix degenerate startprob_ / transmat_ rows (NaN or zero-sum)
            # that hmmlearn can produce on low-data or ill-conditioned stocks
            sp = model.startprob_
            if np.isnan(sp).any() or sp.sum() == 0:
                model.startprob_ = np.ones(N_COMPONENTS) / N_COMPONENTS
            else:
                model.startprob_ = sp / sp.sum()

            for i in range(N_COMPONENTS):
                row = model.transmat_[i]
                if np.isnan(row).any() or row.sum() == 0:
                    model.transmat_[i] = np.ones(N_COMPONENTS) / N_COMPONENTS
            model.transmat_ /= model.transmat_.sum(axis=1, keepdims=True)

            states = model.predict(X)
        finally:
            sys.stderr = _old_stderr

    # Per-state mean return (feature 0 = returns)
    returns_arr = X[:, 0]
    state_means: dict[int, float] = {}
    for s in range(N_COMPONENTS):
        mask = states == s
        if mask.sum() >= MIN_STATE_OBS:
            state_means[s] = float(returns_arr[mask].mean())

    if not state_means:
        raise ValueError("HMM produced no valid states")

    bull_state = max(state_means, key=state_means.get)
    bear_state = min(state_means, key=state_means.get)

    return HMMResult(
        model=model,
        states=states,
        index=index,
        bull_state=bull_state,
        bear_state=bear_state,
        state_means=state_means,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Regime label helpers
# ──────────────────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    "Bull Run":   "#00C853",   # vivid green
    "Bear/Crash": "#FF1744",   # vivid red
    "Neutral":    "#90A4AE",   # blue-grey
}

def state_to_label(state: int, bull: int, bear: int) -> str:
    if state == bull:
        return "Bull Run"
    if state == bear:
        return "Bear/Crash"
    return "Neutral"


def build_regime_series(result: HMMResult) -> pd.Series:
    """Returns a DatetimeIndex-aligned Series of regime label strings."""
    labels = [
        state_to_label(s, result.bull_state, result.bear_state)
        for s in result.states
    ]
    return pd.Series(labels, index=result.index, name="Regime")
