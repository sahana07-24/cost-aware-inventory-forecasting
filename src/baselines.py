"""
Baselines: what retail actually uses in practice, not just variants of this
project's own method. Three baselines, increasing sophistication:
  1. Naive mean + safety stock -- forecast = rolling mean + z(q*) * rolling
     std. Non-adaptive, no ML.
  2. Exponential smoothing (Holt-Winters) + safety margin -- standard
     textbook time-series method, still non-adaptive, still no ML.
  3. Naive step-function adaptive heuristic -- ONE simple rule on top of #2:
     jump the safety margin's quantile target when rolling CV crosses a
     threshold. Tests whether the adaptive IDEA alone, without any ML,
     already captures most of the benefit -- i.e. does the LightGBM
     machinery in this project earn its complexity?
"""
import numpy as np
from scipy import stats


def naive_mean_safety_stock(test, q_star: float) -> np.ndarray:
    z = stats.norm.ppf(q_star)
    return np.clip((test["rolling_mean"] + z * test["rolling_std"]).values, 0, None)


def holt_winters_safety_margin(train, test, q_star: float, seasonal_periods: int = 52):
    """Returns (predictions, point_forecast, resid_std) so callers can also
    build the step-heuristic (baseline 3) from the same fitted model without
    refitting."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    z_star = stats.norm.ppf(q_star)
    try:
        model = ExponentialSmoothing(
            train["demand"].values, trend="add", seasonal="add",
            seasonal_periods=seasonal_periods, initialization_method="estimated",
        ).fit(optimized=True)
        point_forecast = model.forecast(len(test))
        resid_std = np.std(model.resid)
    except Exception:
        # Holt-Winters can fail to converge for some seeds/products -- fall
        # back to the rolling mean/std already available, rather than crash
        # a whole validation run over one bad fit.
        point_forecast = test["rolling_mean"].values
        resid_std = test["rolling_std"].mean()
    preds = np.clip(point_forecast + z_star * resid_std, 0, None)
    return preds, point_forecast, resid_std


def step_adaptive_heuristic(test, point_forecast, resid_std: float,
                             q_star: float, cv_threshold: float = 0.7,
                             alpha_ceiling: float = 0.95) -> np.ndarray:
    z_star = stats.norm.ppf(q_star)
    z_high = stats.norm.ppf(alpha_ceiling)
    test_cv = test["rolling_cv"].values
    z_row = np.where(test_cv >= cv_threshold, z_high, z_star)
    return np.clip(point_forecast + z_row * resid_std, 0, None)
