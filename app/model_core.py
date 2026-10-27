"""
Thin layer between the dissertation's validated `src/` code and the Streamlit app.

`src/gated_hybrid.evaluate_gated_hybrid()` returns only the summary COSTS of each method. The app
also needs the week-by-week ORDER QUANTITIES so it can plot them, so `run_scenario()` below repeats
exactly the same steps, calling the same `src/` functions, and keeps the predictions.

Nothing in `src/` is modified. `tests/test_model_core.py` checks that the costs computed here are
identical to `evaluate_gated_hybrid()` for every product and several seeds, so the app can never
silently drift from the validated method.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cost_model import (  # noqa: E402
    ADAPTIVE_Z_SPAN,
    SBC_CV_THRESHOLD,
    assign_tau_adaptive,
    business_cost,
    cv_to_bin,
    tau_bins_for,
)
from gated_hybrid import (  # noqa: E402
    GATE_STEEPNESS,
    GATE_THRESHOLD_SYNTHETIC,
    fit_residual_bank,
    gate_weight,
    naive_prediction,
    train_mse_model,
)
from synthetic_data import FULL_FEATURES, PRODUCT_CONFIG, make_product_series, train_test_split  # noqa: E402

METHODS = ["naive", "fixed", "adaptive", "mse"]
METHOD_LABELS = {
    "naive": "Naive (rolling mean + safety stock)",
    "fixed": "Fixed target (gated hybrid, q*)",
    "adaptive": "Adaptive target (gated hybrid, CV-driven)",
    "mse": "Standard MSE forecast",
}


@dataclass
class CostSettings:
    unit_cost: float = 50.0
    retail_price: float = 150.0
    carrying_rate_per_year: float = 0.25
    restock_surcharge_rate: float = 0.12

    @property
    def c_o(self) -> float:
        """Weekly cost of holding one unsold unit (annual rate converted to weekly)."""
        return (self.carrying_rate_per_year / 52) * self.unit_cost

    @property
    def c_u(self) -> float:
        """Cost of one unit of unmet demand: lost margin plus expedited-restock surcharge."""
        return (self.retail_price - self.unit_cost) + self.restock_surcharge_rate * self.unit_cost

    @property
    def q_star(self) -> float:
        """Newsvendor critical fractile."""
        return self.c_u / (self.c_u + self.c_o)


def make_series(product: str, seed: int, trend_override: float | None = None) -> pd.DataFrame:
    cfg = dict(PRODUCT_CONFIG[product])
    if trend_override is not None:
        cfg["trend_per_week"] = trend_override
    return make_product_series(seed=seed, **cfg)


def predict_all(train: pd.DataFrame, test: pd.DataFrame, features: list, q_star: float,
                gate_threshold: float = GATE_THRESHOLD_SYNTHETIC,
                gate_steepness: float = GATE_STEEPNESS, seed: int = 0) -> pd.DataFrame:
    """Week-by-week order quantities for all four methods.
    Mirrors src/gated_hybrid.evaluate_gated_hybrid() step for step."""
    tau_bins = tau_bins_for(q_star)
    bank = fit_residual_bank(train, features, tau_bins, seed)

    naive_test_pred = {tau: naive_prediction(test, tau) for tau in tau_bins.values()}
    X_train, X_test = train[features], test[features]
    bank_pred = {b: m.predict(X_test) for b, m in bank.items()}

    naive_train_mean = naive_prediction(train, 0.5)
    naive_test_mean = naive_prediction(test, 0.5)
    mse_model = train_mse_model(X_train, train["demand"].values - naive_train_mean.values, seed=seed)
    mse_pred = mse_model.predict(X_test)

    weights = gate_weight(test["rolling_trend_slope"].values, gate_threshold, gate_steepness)
    est_bins = test["rolling_cv"].apply(cv_to_bin).values

    out = pd.DataFrame(index=test.index)
    out["naive"] = naive_prediction(test, q_star).values
    out["fixed"] = np.clip(naive_test_pred[q_star].values + weights * bank_pred["calm"], 0, None)
    out["mse"] = np.clip(naive_test_mean.values + weights * mse_pred, 0, None)

    adaptive = np.zeros(len(test))
    tau_used = np.zeros(len(test))
    for b, tau in tau_bins.items():
        m = est_bins == b
        adaptive[m] = naive_test_pred[tau].values[m] + weights[m] * bank_pred[b][m]
        tau_used[m] = tau
    out["adaptive"] = np.clip(adaptive, 0, None)

    out["gate_weight"] = weights
    out["regime_estimated"] = est_bins
    out["tau_adaptive"] = tau_used
    return out


def run_scenario(product: str, seed: int, costs: CostSettings,
                 trend_override: float | None = None,
                 evaluate_on: str = "volatile") -> dict:
    """Generate a series, train the gated hybrid on the first 70% of weeks, predict the last 30%,
    and cost every method. `evaluate_on` = 'volatile' (the dissertation's synthetic evaluation
    window) or 'all' (every test week)."""
    df = make_series(product, seed, trend_override)
    train, test = train_test_split(df)
    preds = predict_all(train, test, FULL_FEATURES, costs.q_star, seed=seed)

    true_bins = test["true_cv_regime"].apply(cv_to_bin).values
    mask = true_bins == "volatile" if evaluate_on == "volatile" else np.ones(len(test), dtype=bool)

    y = test["demand"].values
    summary = {}
    for m in METHODS:
        p = preds[m].values
        summary[m] = dict(
            cost_per_week=business_cost(y[mask], p[mask], costs.c_o, costs.c_u),
            stockout_rate=float((y[mask] > p[mask]).mean()) if mask.any() else float("nan"),
            avg_order=float(p[mask].mean()) if mask.any() else float("nan"),
            leftover_units=float(np.clip(p[mask] - y[mask], 0, None).sum()),
            shortfall_units=float(np.clip(y[mask] - p[mask], 0, None).sum()),
        )
    return dict(df=df, train=train, test=test, preds=preds, mask=mask, summary=summary,
                n_eval_weeks=int(mask.sum()))


def tau_curve(q_star: float, cvs: np.ndarray) -> np.ndarray:
    return np.array([assign_tau_adaptive(cv, q_star) for cv in cvs])


__all__ = ["CostSettings", "METHODS", "METHOD_LABELS", "PRODUCT_CONFIG", "run_scenario", "predict_all",
           "tau_curve", "tau_bins_for", "cv_to_bin", "SBC_CV_THRESHOLD", "ADAPTIVE_Z_SPAN"]
