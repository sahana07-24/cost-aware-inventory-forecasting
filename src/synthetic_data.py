"""
Synthetic demand-series generator -- Products A, B, C and the controlled
trend-sign conditions used to diagnose the gate.

HISTORY THIS CODE ENCODES (see Technical_Validation_Summary.docx): the
original generator placed its single volatile-demand block only in the test
period. Tree-based models cannot extrapolate beyond the range of feature
values seen in training, so the model had never seen a volatile-regime
example -- causing a large loss to a trivial naive baseline that was
mistaken, at first, for a modelling failure rather than an experimental-design
bug. Fixed by cycling the volatile regime through the series three times
(see `_apply_volatility_blocks` below), so training data always contains
genuine volatile-regime examples. This was checked for leakage of its own
(an injected demand spike, confirming rolling-window features immediately
before it were unaffected) and confirmed clean -- see the diagnostic
scripts this module's logic was extracted from.
"""
import numpy as np
import pandas as pd

N_WEEKS = 260
ROLLING_WINDOW = 12

# Three synthetic demand "personalities" -- not real products, used to check
# the method under varied conditions rather than tuned to one specific case.
# (Project_Explainer.docx, Section 3.)
PRODUCT_CONFIG = {
    "product_A": dict(mean_demand=200, base_cv=0.15, volatile_cv=0.90, trend_per_week=0.0005),
    "product_B": dict(mean_demand=150, base_cv=0.20, volatile_cv=1.10, trend_per_week=-0.0003),
    "product_C": dict(mean_demand=300, base_cv=0.10, volatile_cv=0.70, trend_per_week=0.0008),
}

# Controlled trend-sign sweep: everything IDENTICAL to product_A's base
# parameters except trend_per_week, isolating trend direction/magnitude as
# the sole variable under test (see trend_sign_test.py's original design
# rationale in Technical_Validation_Summary.docx Section 3.2).
TREND_SWEEP_FIXED_PARAMS = dict(mean_demand=200, base_cv=0.15, volatile_cv=0.90)
TREND_CONDITIONS = {
    "strong_negative": -0.0010,
    "mild_negative":   -0.0005,
    "no_trend":          0.0000,
    "mild_positive":     0.0005,
    "strong_positive":   0.0010,
}

BASE_FEATURES = ["weekofyear_sin", "weekofyear_cos", "promo_flag", "lag_1", "lag_2",
                  "week_norm", "rolling_mean", "rolling_std", "rolling_cv"]
GATE_FEATURES = ["rolling_trend_slope", "trend_vol_interaction"]
FULL_FEATURES = BASE_FEATURES + GATE_FEATURES

VOLATILE_BLOCK_BOUNDS = [(0.20, 0.30), (0.45, 0.55), (0.80, 0.90)]  # 3 cycles through the series


def rolling_ols_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """OLS slope of arr[t-window+1 : t+1] against a 0..window-1 time index,
    for every t -- a legitimate 'recent trend as observed so far' feature.
    Backward-looking only; the caller still needs to .shift(1) this to keep
    week t's feature strictly pre-t, matching every other rolling feature
    in this project.
    """
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1: i + 1]
        out[i] = ((x - x_mean) * (y - y.mean())).sum() / denom
    return out


def make_product_series(seed: int, mean_demand: float, base_cv: float,
                         volatile_cv: float, trend_per_week: float,
                         n_weeks: int = N_WEEKS, rolling_window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Generate one synthetic weekly demand series: trend x seasonality x
    promotions x noise, with volatility cycling calm/volatile 3 times
    through the series (see module docstring). Returns a DataFrame with the
    demand series, the ground-truth CV regime (for test-time evaluation
    masks -- never used as a model feature), and every rolling feature the
    gated-hybrid model consumes.
    """
    rng = np.random.default_rng(seed)
    week = np.arange(n_weeks)

    cv_by_week = np.full(n_weeks, base_cv)
    for lo, hi in VOLATILE_BLOCK_BOUNDS:
        cv_by_week[int(n_weeks * lo):int(n_weeks * hi)] = volatile_cv

    trend = 1 + trend_per_week * week
    seasonality = 1 + 0.15 * np.sin(2 * np.pi * week / 52)
    promo_flag = rng.binomial(1, 0.08, n_weeks)
    promo_effect = 1 + promo_flag * rng.uniform(0.4, 0.8, n_weeks)
    true_mean = mean_demand * trend * seasonality * promo_effect
    true_sd = cv_by_week * mean_demand
    demand = np.clip(true_mean + rng.normal(0, true_sd), 0, None)

    df = pd.DataFrame({"week": week, "demand": demand, "true_cv_regime": cv_by_week})
    df["weekofyear_sin"] = np.sin(2 * np.pi * week / 52)
    df["weekofyear_cos"] = np.cos(2 * np.pi * week / 52)
    df["promo_flag"] = promo_flag
    df["lag_1"] = df["demand"].shift(1).fillna(mean_demand)
    df["lag_2"] = df["demand"].shift(2).fillna(mean_demand)
    df["week_norm"] = week / n_weeks
    df["rolling_cv"] = (df["demand"].rolling(rolling_window, min_periods=4).std()
                         / df["demand"].rolling(rolling_window, min_periods=4).mean()).shift(1).fillna(base_cv)
    df["rolling_mean"] = df["demand"].rolling(rolling_window, min_periods=4).mean().shift(1).fillna(mean_demand)
    df["rolling_std"] = df["demand"].rolling(rolling_window, min_periods=4).std().shift(1).fillna(base_cv * mean_demand)

    raw_slope = rolling_ols_slope(df["demand"].values, rolling_window)
    df["rolling_trend_slope"] = pd.Series(raw_slope).shift(1).fillna(0.0).values
    df["trend_vol_interaction"] = df["rolling_trend_slope"] * df["rolling_cv"]
    return df


def train_test_split(df: pd.DataFrame, train_frac: float = 0.7):
    split = int(len(df) * train_frac)
    return df.iloc[:split].copy(), df.iloc[split:].copy()
