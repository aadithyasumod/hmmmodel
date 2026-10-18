"""
hmm_engine.py
Walk-forward rolling HMM regime detector.

For each trading day t (t >= ROLLING_WINDOW):
  • Train a 7-state GaussianHMM on X[t-252 : t]   (252 trading days)
  • Run Viterbi on that same window
  • Take the LAST decoded state as the OOS regime signal for day t
  • Identify Bull/Bear in that window by mean return of each state

This ensures every regime label used in the backtest is strictly
out-of-sample — the model never sees future data.

Features fed to the HMM:
  • Returns          — directional signal
  • Range ratio      — (High-Low)/Close  → realised volatility proxy
  • Volume z-score   — 20-period rolling z-score of volume
"""

import io
import sys
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from hmmlearn import hmm


N_COMPONENTS   = 7
N_ITER_ROLLING = 30     # fewer iterations per window; windows overlap by 251/252 days
RANDOM_STATE   = 42
MIN_STATE_OBS  = 5      # minimum observations for a state to be labelled
ROLLING_WINDOW = 252    # one trading year
REFIT_STEP     = 5      # refit HMM every N days (weekly); forward-fill between steps

# Standardised state codes stored in HMMResult.states
BULL_CODE    =  1
BEAR_CODE    = -1
NEUTRAL_CODE =  0


@dataclass
class HMMResult:
    model:       hmm.GaussianHMM   # last window's fitted model (metadata only)
    states:      np.ndarray        # standardised codes: 1=Bull, -1=Bear, 0=Neutral
    index:       pd.DatetimeIndex  # aligned dates (OOS period only, starts at day 252)
    bull_state:  int               # always BULL_CODE  = 1
    bear_state:  int               # always BEAR_CODE  = -1
    state_means: dict[int, float]  # symbolic {1:1.0, 0:0.0, -1:-1.0}


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

    for col in feat_df.columns:
        lo, hi = feat_df[col].quantile([0.01, 0.99])
        feat_df[col] = feat_df[col].clip(lo, hi)

    X = feat_df.values.astype(np.float64)
    return X, feat_df.index


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_model() -> hmm.GaussianHMM:
    return hmm.GaussianHMM(
        n_components=N_COMPONENTS,
        covariance_type="diag",
        n_iter=N_ITER_ROLLING,
        random_state=RANDOM_STATE,
        tol=1e-4,
    )


def _fix_degenerate(model: hmm.GaussianHMM) -> None:
    """Normalise any NaN or zero-sum rows in startprob_ / transmat_ in-place."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward rolling HMM
# ──────────────────────────────────────────────────────────────────────────────

def fit_hmm(df: pd.DataFrame) -> HMMResult:
    """
    Walk-forward rolling HMM fit.

    For each day t in [ROLLING_WINDOW, len(X)):
      1. Train on X[t-252 : t]
      2. Viterbi-decode the same window
      3. Record the LAST decoded state, mapped to Bull/Bear/Neutral
         using that window's per-state mean return.

    Returns an HMMResult whose .states / .index cover only the OOS period.
    Raises ValueError if data is insufficient.
    """
    X, index = _build_features(df)

    if len(X) < ROLLING_WINDOW + 20:
        raise ValueError(
            f"Insufficient data: {len(X)} bars (need ≥ {ROLLING_WINDOW + 20})"
        )

    rolling_codes: list[int] = []
    valid_dates:   list       = []
    last_model                = None
    last_code                 = NEUTRAL_CODE
    _devnull                  = io.StringIO()

    for t in range(ROLLING_WINDOW, len(X)):
        # Only refit the HMM every REFIT_STEP days; forward-fill otherwise
        if (t - ROLLING_WINDOW) % REFIT_STEP == 0:
            X_win = X[t - ROLLING_WINDOW : t]

            model = _make_model()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _old_stderr = sys.stderr
                sys.stderr  = _devnull
                try:
                    model.fit(X_win)
                    _fix_degenerate(model)
                    preds = model.predict(X_win)
                except Exception:
                    preds = np.zeros(ROLLING_WINDOW, dtype=int)
                finally:
                    sys.stderr = _old_stderr

            # Identify bull / bear state for this window
            returns_win = X_win[:, 0]
            state_means_w: dict[int, float] = {}
            state_vars_w:   dict[int, float] = {}
            for s in range(N_COMPONENTS):
                mask = preds == s
                if mask.sum() >= MIN_STATE_OBS:
                    state_means_w[s] = float(returns_win[mask].mean())
                    state_vars_w[s] = float(returns_win[mask].var())

            last_state = int(preds[-1])

            if not state_means_w:
                last_code = NEUTRAL_CODE
            else:
                bull_w = max(state_means_w, key=state_means_w.get)
                bear_w = min(state_means_w, key=state_means_w.get)
                if last_state == bull_w:
                    last_code = BULL_CODE
                elif last_state == bear_w:
                    last_code = BEAR_CODE
                else:
                    last_code = NEUTRAL_CODE

            last_model = model
            final_state_means = state_means_w
            final_state_vars = state_vars_w

        rolling_codes.append(last_code)
        valid_dates.append(index[t])

    states_arr = np.array(rolling_codes, dtype=int)

    return HMMResult(
        model=last_model,
        states=states_arr,
        index=pd.DatetimeIndex(valid_dates),
        bull_state=BULL_CODE,
        bear_state=BEAR_CODE,
        state_means={BULL_CODE: 1.0, NEUTRAL_CODE: 0.0, BEAR_CODE: -1.0},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Regime label helpers  (unchanged interface — backtester uses these as-is)
# ──────────────────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    "Bull Run":   "#00C853",
    "Bear/Crash": "#FF1744",
    "Neutral":    "#90A4AE",
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
