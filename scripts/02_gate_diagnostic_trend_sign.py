"""
Gate diagnostic: reproduces the trend-sign test (Technical_Validation_Summary
Section 3.2) and the gate fix it motivated (Section 3.4), in one script.

DESIGN, fixed before running: 5 synthetic conditions, IDENTICAL mean_demand,
base_cv, volatile_cv -- the only thing that varies is trend_per_week. This
isolates trend sign/magnitude as the sole difference between conditions.

PART 1 -- ungated hybrid vs naive, across the trend sweep: confirms the
hybrid's residual correction is harmful specifically under positive trend
(a monotonic pattern, not just "sometimes worse").
PART 2 -- gated hybrid vs naive, same sweep: confirms the gate neutralizes
that harm (loss should shrink toward/through zero) without needing the
trend sign to be known in advance -- the gate infers it from the same
rolling trend-slope feature at prediction time.

Ungated is reproduced here as gate_threshold=+inf: the sigmoid gate weight
is then 1.0 for every possible trend-slope value, i.e. no gating at all --
the same evaluate_gated_hybrid() function used everywhere else in this
pipeline, not a separate code path.

STOPPING RULE: batches of 50 seeds, bootstrapped 95% CI, stop at half-width
< 2pp (a looser, diagnostic bar -- not the 1pp bar used for headline
results), capped at 300 seeds per condition. Same rule Technical_Validation
_Summary reports using originally.

Run: python 02_gate_diagnostic_trend_sign.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy import stats

from cost_model import Q_STAR, C_O, C_U, cv_to_bin
from synthetic_data import (TREND_CONDITIONS, TREND_SWEEP_FIXED_PARAMS, FULL_FEATURES,
                             make_product_series, train_test_split)
from gated_hybrid import evaluate_gated_hybrid, GATE_THRESHOLD_SYNTHETIC
from bootstrap import validate_with_stopping_rule

BATCH, CAP, TARGET_HW = 50, 300, 2.0


def make_run_one(trend_per_week: float, gated: bool):
    threshold = GATE_THRESHOLD_SYNTHETIC if gated else np.inf

    def run_one(seed: int):
        df = make_product_series(seed=seed, trend_per_week=trend_per_week, **TREND_SWEEP_FIXED_PARAMS)
        train, test = train_test_split(df)
        true_bins = test["true_cv_regime"].apply(cv_to_bin).values
        vmask = true_bins == "volatile"
        if vmask.sum() < 4:
            return None
        r = evaluate_gated_hybrid(train, test, FULL_FEATURES, Q_STAR, C_O, C_U,
                                   gate_threshold=threshold, seed=seed, mask=vmask)
        # positive = hybrid/gated beats naive
        return 100 * (r["cost_naive"] - r["cost_adaptive"]) / r["cost_naive"]
    return run_one


if __name__ == "__main__":
    print("=" * 82)
    print("PART 1: UNGATED hybrid vs naive, across the trend-sign sweep")
    print("=" * 82)
    ungated_results = {}
    for cond_name, trend in TREND_CONDITIONS.items():
        ungated_results[cond_name] = validate_with_stopping_rule(
            make_run_one(trend, gated=False), BATCH, CAP, TARGET_HW, label=cond_name)

    print("\n" + "=" * 82)
    print("REGRESSION: ungated hybrid-vs-naive gap against trend_per_week")
    print("=" * 82)
    trends = np.array([TREND_CONDITIONS[k] for k in ungated_results])
    means = np.array([v["mean"] for v in ungated_results.values()])
    slope, intercept, r, p, se = stats.linregress(trends, means)
    print(f"Slope: {slope:.1f}  (% improvement per unit of trend_per_week)")
    print(f"R-squared: {r**2:.3f}   p-value: {p:.4f}")
    print(f"-> {'SIGNIFICANT relationship between trend sign and hybrid performance' if p < 0.05 else 'NOT significant'}")

    print("\n" + "=" * 82)
    print(f"PART 2: GATED hybrid vs naive, same sweep (threshold={GATE_THRESHOLD_SYNTHETIC}, "
          f"pre-committed from no-trend training data)")
    print("=" * 82)
    gated_results = {}
    for cond_name, trend in TREND_CONDITIONS.items():
        gated_results[cond_name] = validate_with_stopping_rule(
            make_run_one(trend, gated=True), BATCH, CAP, TARGET_HW, label=cond_name)

    print("\n" + "=" * 82)
    print("BEFORE / AFTER GATE, side by side")
    print("=" * 82)
    print(f"{'condition':18s} {'trend/wk':>10s}   {'ungated mean':>14s}   {'gated mean':>12s}")
    for cond_name, trend in TREND_CONDITIONS.items():
        u, g = ungated_results[cond_name], gated_results[cond_name]
        print(f"{cond_name:18s} {trend:+10.4f}   {u['mean']:+13.2f}%   {g['mean']:+11.2f}%")
