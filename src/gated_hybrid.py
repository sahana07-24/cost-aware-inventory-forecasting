"""
The gated-hybrid model: this project's core method, and the three earlier
architectures it superseded (kept here, not deleted, because the dissertation
narrative depends on showing the progression, not just the final version).

PROGRESSION (see Technical_Validation_Summary.docx):
  1. Pure LightGBM quantile regression (predict demand directly) -- lost
     68-242% to a trivial naive baseline (validated, 800 seeds per product,
     forced-min stopping rule -- see results/01_lgbm_vs_naive_validated.csv
     and scripts/01_baseline_comparison.py) once the extrapolation bug was
     fixed, because a frozen, batch-trained model can't match a method that
     recomputes its estimate from live test-period observations every week.
  2. Ungated hybrid -- LightGBM predicts the RESIDUAL (actual - naive
     prediction) instead of the raw quantile. Inherits the naive method's
     live-updating advantage. Closed most of the gap; still lost on Products
     A and C.
  3. Diagnosis found the residual correction is actively HARMFUL under
     strong positive trend (not just unhelpful) -- a trend-sign test with 5
     controlled conditions showed a clean monotonic relationship.
  4. Gated hybrid (this project's method) -- a smooth sigmoid gate, driven
     by a rolling trend-slope feature, shrinks the residual correction
     toward zero as that slope rises. This is the architecture every
     headline result in this project (synthetic and real-data) is built on.

GATE THRESHOLD: calibrated from the 75th percentile of rolling_trend_slope
observed in TRAINING data under the synthetic no-trend condition, chosen
BEFORE any post-gate outcome was examined (see Technical_Validation_Summary
Section 3.4). This numeric value (4.6) is specific to the synthetic data's
demand scale and does NOT transfer to real data -- the real-data pipeline
recalibrates it per-article, per-fold, from that fold's own training data
(same principled method, different numbers). See real_data.py.
"""
import numpy as np
import pandas as pd
from scipy import stats

from cost_model import tau_bins_for, business_cost

GATE_THRESHOLD_SYNTHETIC = 4.6
GATE_STEEPNESS = 0.3


def gate_weight(trend_slope, threshold: float = GATE_THRESHOLD_SYNTHETIC,
                 steepness: float = GATE_STEEPNESS):
    return 1.0 / (1.0 + np.exp(steepness * (np.asarray(trend_slope, dtype=float) - threshold)))


def naive_prediction(df: pd.DataFrame, tau: float) -> pd.Series:
    z = stats.norm.ppf(tau)
    return np.clip(df["rolling_mean"] + z * df["rolling_std"], 0, None)


def train_quantile_model(X_train, y_train, tau: float, seed: int = 0,
                          num_leaves: int = 7, learning_rate: float = 0.05,
                          min_data_in_leaf: int = 10, num_boost_round: int = 200):
    """Shared LightGBM quantile-regression trainer -- used both for the
    'pure' architecture (target = raw demand) and the hybrid architectures
    (target = residual). Same hyperparameters everywhere in this project,
    deliberately not re-tuned per architecture so comparisons stay fair.
    """
    import lightgbm as lgb
    params = dict(num_leaves=num_leaves, learning_rate=learning_rate,
                  min_data_in_leaf=min_data_in_leaf, verbose=-1, seed=seed,
                  objective="quantile", alpha=tau)
    return lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=num_boost_round)


def train_mse_model(X_train, y_train, seed: int = 0, num_leaves: int = 7,
                     learning_rate: float = 0.05, min_data_in_leaf: int = 10,
                     num_boost_round: int = 200):
    """Same architecture and hyperparameters as train_quantile_model, but
    trained with a symmetric MSE (L2) objective instead of an asymmetric
    quantile objective -- this is the 'standard MSE' arm RQ2 asks for.
    Deliberately not re-tuned relative to the quantile trainer so the only
    thing that differs between this baseline and 'fixed'/'adaptive' is the
    loss function and the target level (central point forecast, not an
    upper quantile), not the model capacity."""
    import lightgbm as lgb
    params = dict(num_leaves=num_leaves, learning_rate=learning_rate,
                  min_data_in_leaf=min_data_in_leaf, verbose=-1, seed=seed,
                  objective="regression")
    return lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=num_boost_round)


def predict_pure_lgbm(train: pd.DataFrame, test: pd.DataFrame, features: list,
                       q_star: float, seed: int = 0) -> np.ndarray:
    """Architecture 1: LightGBM predicts the raw demand quantile directly,
    at the CV-adaptive tau for each test row. No naive baseline involved."""
    tau_bins = tau_bins_for(q_star)
    X_train, y_train = train[features], train["demand"]
    X_test = test[features]
    bank = {b: train_quantile_model(X_train, y_train, tau, seed).predict(X_test)
            for b, tau in tau_bins.items()}
    est_bins = test["rolling_cv"].apply(lambda cv: _bin_for(cv, q_star)).values
    return np.select([est_bins == b for b in tau_bins], [bank[b] for b in tau_bins])


def _bin_for(cv, q_star):
    from cost_model import cv_to_bin
    return cv_to_bin(cv)


def fit_residual_bank(train: pd.DataFrame, features: list, tau_bins: dict, seed: int = 0) -> dict:
    """Trains one residual-quantile model per regime bin (calm/moderate/
    volatile). bank['calm'] is trained at tau=q_star -- this is also what
    'fixed' uses, so fixed and adaptive share this exact model whenever a
    test row falls in the calm bin."""
    X_train = train[features]
    naive_train_pred = {tau: naive_prediction(train, tau) for tau in tau_bins.values()}
    bank = {}
    for b, tau in tau_bins.items():
        residual_train = train["demand"].values - naive_train_pred[tau].values
        bank[b] = train_quantile_model(X_train, residual_train, tau, seed)
    return bank


def evaluate_gated_hybrid(train: pd.DataFrame, test: pd.DataFrame, features: list,
                           q_star: float, c_o: float, c_u: float,
                           gate_threshold: float = GATE_THRESHOLD_SYNTHETIC,
                           gate_steepness: float = GATE_STEEPNESS,
                           seed: int = 0, mask: np.ndarray = None) -> dict:
    """The central evaluation used by every headline result in this
    project: trains the gated-hybrid residual bank once, then produces FOUR
    prediction sets from it --
      naive     : rolling-mean + safety-stock at the fixed q_star, no ML at all
      fixed     : gated hybrid, but always using the q_star-tau model (the
                  'calm' bin's model) regardless of the test row's regime
      adaptive  : gated hybrid, using the CV-adaptive tau/model per test row
      mse       : gated hybrid, but the residual model is trained with a
                  symmetric MSE (L2) objective around the central point
                  forecast (tau=0.5), not an asymmetric quantile objective
                  -- the 'standard MSE' baseline RQ2 compares against. Same
                  features, same gate, same walk-forward split as fixed/
                  adaptive; only the loss function and target level differ.

    Comparison A (gated hybrid vs. naive) = naive vs. adaptive.
    Comparison B (the thesis: adaptive vs. fixed) = fixed vs. adaptive.
    Comparison C (closing RQ2: adaptive vs. standard MSE) = mse vs. adaptive.
    All comparisons come out of this single trained model bank, exactly as
    they do in phase5_hm_validation.py's evaluate_one_fold -- this function
    is that logic, generalized to also serve the synthetic-data scripts.

    `mask` (optional) restricts the returned costs to a subset of `test`
    rows (e.g. the synthetic volatile-regime-only evaluation window used in
    the diagnostic scripts). If omitted, all of `test` is used (the
    real-data walk-forward convention).
    """
    from cost_model import cv_to_bin, tau_bins_for
    tau_bins = tau_bins_for(q_star)
    bin_names = list(tau_bins.keys())

    bank = fit_residual_bank(train, features, tau_bins, seed)

    naive_test_pred = {tau: naive_prediction(test, tau) for tau in tau_bins.values()}
    X_train, X_test = train[features], test[features]
    bank_pred = {b: m.predict(X_test) for b, m in bank.items()}

    # -- 'standard MSE' baseline: same architecture, symmetric L2 loss,
    # targets the central point forecast (tau=0.5) instead of an upper
    # quantile -- the arm needed to close RQ2's MSE comparison.
    naive_train_mean = naive_prediction(train, 0.5)
    naive_test_mean = naive_prediction(test, 0.5)
    residual_train_mean = train["demand"].values - naive_train_mean.values
    mse_model = train_mse_model(X_train, residual_train_mean, seed=seed)
    mse_pred = mse_model.predict(X_test)

    weights = gate_weight(test["rolling_trend_slope"].values, gate_threshold, gate_steepness)
    est_bins = test["rolling_cv"].apply(cv_to_bin).values

    preds_naive = naive_prediction(test, q_star).values
    preds_fixed = np.clip(naive_test_pred[q_star].values + weights * bank_pred["calm"], 0, None)
    preds_mse = np.clip(naive_test_mean.values + weights * mse_pred, 0, None)

    preds_adaptive = np.zeros(len(test))
    for b in bin_names:
        row_mask = est_bins == b
        tau = tau_bins[b]
        preds_adaptive[row_mask] = naive_test_pred[tau].values[row_mask] + weights[row_mask] * bank_pred[b][row_mask]
    preds_adaptive = np.clip(preds_adaptive, 0, None)

    y_test = test["demand"].values
    if mask is not None:
        y_test, preds_naive, preds_fixed, preds_adaptive, preds_mse = (
            y_test[mask], preds_naive[mask], preds_fixed[mask], preds_adaptive[mask], preds_mse[mask])

    return dict(
        n_train=len(train), n_test=len(y_test), q_star=q_star, gate_threshold=gate_threshold,
        cost_naive=business_cost(y_test, preds_naive, c_o, c_u),
        cost_fixed=business_cost(y_test, preds_fixed, c_o, c_u),
        cost_adaptive=business_cost(y_test, preds_adaptive, c_o, c_u),
        cost_mse=business_cost(y_test, preds_mse, c_o, c_u),
    )
