"""
The two headline synthetic-data comparisons, on Products A/B/C under the
final gated-hybrid pipeline:

COMPARISON A -- gated hybrid vs. naive baseline. Establishes the method is
at least competitive with strong simple practice. Necessary for credibility,
NOT the dissertation's research question.

COMPARISON B -- adaptive vs. fixed forecasting target (the actual thesis).
Within the SAME gated-hybrid architecture, does letting the quantile target
adapt to the current volatility regime beat using one fixed quantile
throughout? This is the number the dissertation depends on.

COMPARISON C -- adaptive vs. standard MSE (closes RQ2's other half). Same
gated-hybrid architecture and walk-forward split; only the residual model's
loss function and target level differ (symmetric MSE at the central point
forecast, vs. the adaptive scheme's CV-driven asymmetric quantile target).
This is the "compared to standard MSE" arm of RQ2 that earlier updates left
untested.

All three comparisons are produced from the SAME trained model bank per seed
(evaluate_gated_hybrid returns cost_naive/cost_fixed/cost_adaptive/cost_mse
together) -- exactly as in Technical_Validation_Summary's methodology.

STOPPING RULES (per Technical_Validation_Summary Section 4 -- this project's
own history of an early, under-powered apparent null in Comparison B that
reversed to significant once forced further):
  Comparison A : batches of 50, target half-width 1pp, cap 400.
  Comparison B : batches of 50, target half-width 1pp, FORCED minimum 400
                 seeds before the stopping rule is allowed to fire, cap 600.
  Comparison C : same forced-minimum protocol as Comparison B (same
                 rationale -- an early apparent null is not trusted here
                 either), batches of 50, target half-width 1pp, forced
                 minimum 400, cap 600.

Run: python 03_comparison_A_and_B_gated_hybrid.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from cost_model import C_O, C_U, Q_STAR, cv_to_bin
from synthetic_data import PRODUCT_CONFIG, FULL_FEATURES, make_product_series, train_test_split
from gated_hybrid import evaluate_gated_hybrid, GATE_THRESHOLD_SYNTHETIC
from bootstrap import validate_with_stopping_rule

BATCH = 50
CAP_A, TARGET_A = 400, 1.0
CAP_B, TARGET_B, FORCED_MIN_B = 600, 1.0, 400


def make_run_one(cfg: dict, comparison: str):
    def run_one(seed: int):
        df = make_product_series(seed=seed, **cfg)
        train, test = train_test_split(df)
        true_bins = test["true_cv_regime"].apply(cv_to_bin).values
        vmask = true_bins == "volatile"
        if vmask.sum() < 4:
            return None
        r = evaluate_gated_hybrid(train, test, FULL_FEATURES, Q_STAR, C_O, C_U,
                                   gate_threshold=GATE_THRESHOLD_SYNTHETIC, seed=seed, mask=vmask)
        if comparison == "A":  # gated hybrid (adaptive) vs naive
            return 100 * (r["cost_naive"] - r["cost_adaptive"]) / r["cost_naive"]
        elif comparison == "B":  # adaptive vs fixed -- the thesis
            return 100 * (r["cost_fixed"] - r["cost_adaptive"]) / r["cost_fixed"]
        else:  # C: adaptive vs standard MSE -- closes RQ2
            return 100 * (r["cost_mse"] - r["cost_adaptive"]) / r["cost_mse"]
    return run_one


if __name__ == "__main__":
    print("=" * 82)
    print("COMPARISON A: gated hybrid (adaptive) vs. naive baseline")
    print("=" * 82)
    results_a = {}
    for name, cfg in PRODUCT_CONFIG.items():
        results_a[name] = validate_with_stopping_rule(
            make_run_one(cfg, "A"), BATCH, CAP_A, TARGET_A, label=name)

    print("\n" + "=" * 82)
    print("COMPARISON B: adaptive vs. fixed target (THE THESIS)")
    print("=" * 82)
    results_b = {}
    for name, cfg in PRODUCT_CONFIG.items():
        results_b[name] = validate_with_stopping_rule(
            make_run_one(cfg, "B"), BATCH, CAP_B, TARGET_B, forced_min=FORCED_MIN_B, label=name)

    print("\n" + "=" * 82)
    print("COMPARISON C: adaptive vs. standard MSE (closes RQ2)")
    print("=" * 82)
    results_c = {}
    for name, cfg in PRODUCT_CONFIG.items():
        results_c[name] = validate_with_stopping_rule(
            make_run_one(cfg, "C"), BATCH, CAP_B, TARGET_B, forced_min=FORCED_MIN_B, label=name)

    print("\n" + "=" * 82)
    print("SUMMARY")
    print("=" * 82)
    rows = []
    for name in PRODUCT_CONFIG:
        a, b, c = results_a[name], results_b[name], results_c[name]
        a_sig = a["lo"] > 0 or a["hi"] < 0
        b_sig = b["lo"] > 0 or b["hi"] < 0
        c_sig = c["lo"] > 0 or c["hi"] < 0
        rows.append(dict(product=name,
                          comparison_A_mean=a["mean"], comparison_A_ci=f"[{a['lo']:+.2f}%, {a['hi']:+.2f}%]",
                          comparison_A_n=a["n"], comparison_A_significant=a_sig,
                          comparison_B_mean=b["mean"], comparison_B_ci=f"[{b['lo']:+.2f}%, {b['hi']:+.2f}%]",
                          comparison_B_n=b["n"], comparison_B_significant=b_sig,
                          comparison_C_mean=c["mean"], comparison_C_ci=f"[{c['lo']:+.2f}%, {c['hi']:+.2f}%]",
                          comparison_C_n=c["n"], comparison_C_significant=c_sig))
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "03_comparison_A_and_B_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
