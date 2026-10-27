"""
Baseline comparison: the adaptive LightGBM method (pure, no hybrid, no gate)
against three baselines of increasing sophistication, on the SAME synthetic
products, SAME cost model, SAME volatile-period test window.

This reproduces the ORIGINAL baselines.py result -- the one that first
surfaced the extrapolation bug (pure LightGBM losing 15-24% to a naive
baseline). Kept as the first script in this consolidated pipeline
deliberately: it's the discovery that everything else in this project
responds to, not a footnote.

Run: python 01_baseline_comparison.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy import stats

from cost_model import Q_STAR, C_O, C_U, business_cost, cv_to_bin
from synthetic_data import PRODUCT_CONFIG, BASE_FEATURES, make_product_series, train_test_split
from gated_hybrid import predict_pure_lgbm
from bootstrap import validate_with_stopping_rule

N_SEEDS = 50  # quick-look table below -- context only, not the validated claim (see bottom of script)

# The comparison that actually matters here -- pure LightGBM vs. naive -- gets
# the same validation discipline as every other headline number in this
# project. Technical_Validation_Summary reports this needed 500-800 seeds to
# converge (looser statistics than Comparison A/B's 300-400, because percentage
# gaps here are larger and noisier). Forced minimum + cap taken directly from
# that documented range, not re-chosen.
BATCH, FORCED_MIN, CAP, TARGET_HW = 50, 500, 800, 1.0


def run_one_seed_lgbm_vs_naive(seed: int, cfg: dict):
    df = make_product_series(seed=seed, **cfg)
    train, test = train_test_split(df)
    true_bins = test["true_cv_regime"].apply(cv_to_bin).values
    volatile_mask = true_bins == "volatile"
    if volatile_mask.sum() < 4:
        return None
    yv = test["demand"].values[volatile_mask]

    from baselines import naive_mean_safety_stock
    preds_naive = naive_mean_safety_stock(test, Q_STAR)
    preds_lgbm = predict_pure_lgbm(train, test, BASE_FEATURES, Q_STAR, seed)

    cost_naive = business_cost(yv, preds_naive[volatile_mask], C_O, C_U)
    cost_lgbm = business_cost(yv, np.asarray(preds_lgbm)[volatile_mask], C_O, C_U)
    return 100 * (cost_naive - cost_lgbm) / cost_naive  # positive = LightGBM beats naive


def run_one_seed(seed: int) -> dict:
    out = {}
    for name, cfg in PRODUCT_CONFIG.items():
        df = make_product_series(seed=seed, **cfg)
        train, test = train_test_split(df)

        true_bins = test["true_cv_regime"].apply(cv_to_bin).values
        volatile_mask = true_bins == "volatile"
        if volatile_mask.sum() < 4:
            continue
        yv = test["demand"].values[volatile_mask]

        from baselines import naive_mean_safety_stock, holt_winters_safety_margin, step_adaptive_heuristic
        preds_naive = naive_mean_safety_stock(test, Q_STAR)
        preds_hw, point_forecast, resid_std = holt_winters_safety_margin(train, test, Q_STAR)
        preds_step = step_adaptive_heuristic(test, point_forecast, resid_std, Q_STAR)
        preds_lgbm = predict_pure_lgbm(train, test, BASE_FEATURES, Q_STAR, seed)

        for label, preds in [
            ("1. Naive mean + safety stock", preds_naive),
            ("2. Exp. smoothing + safety margin", preds_hw),
            ("3. Naive step-adaptive heuristic", preds_step),
            ("4. LightGBM adaptive (pure, no hybrid)", preds_lgbm),
        ]:
            out[(name, label)] = business_cost(yv, np.asarray(preds)[volatile_mask], C_O, C_U)
    return out


if __name__ == "__main__":
    print(f"Running {N_SEEDS} seeds across {len(PRODUCT_CONFIG)} products x 4 methods...")
    rows = []
    for s in range(N_SEEDS):
        for (product, label), cost in run_one_seed(s).items():
            rows.append({"seed": s, "product": product, "method": label, "cost": cost})
    results = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "01_baseline_comparison.csv")
    results.to_csv(out_path, index=False)
    print(f"Done. Raw results saved to {out_path}\n")

    print("=" * 90)
    print("BASELINE COMPARISON: average weekly £ cost during volatile periods, by method")
    print("=" * 90)
    for product in PRODUCT_CONFIG:
        print(f"\n{product}:")
        sub = results[results["product"] == product].groupby("method")["cost"].agg(["mean", "std"]).sort_values("mean")
        print(sub.to_string())

    print("\n" + "=" * 90)
    print("OVERALL (averaged across products)")
    print("=" * 90)
    print(results.groupby("method")["cost"].mean().sort_values().to_string())

    print("\n" + "=" * 90)
    print("Does LightGBM beat the naive step-adaptive heuristic significantly?")
    print("(does the ML machinery earn its complexity over one simple if/else rule?)")
    print("=" * 90)
    for product in PRODUCT_CONFIG:
        sub = results[results["product"] == product]
        step = sub[sub["method"] == "3. Naive step-adaptive heuristic"].sort_values("seed")["cost"].values
        lgbm = sub[sub["method"] == "4. LightGBM adaptive (pure, no hybrid)"].sort_values("seed")["cost"].values
        if len(step) == len(lgbm) and len(step) > 1:
            t, p = stats.ttest_rel(step, lgbm)
            diff = 100 * (step.mean() - lgbm.mean()) / step.mean()
            print(f"  {product:12s}: step-heuristic=£{step.mean():,.0f}  LightGBM=£{lgbm.mean():,.0f}  "
                  f"({diff:+.1f}%)  p={p:.4f}  -> {'SIGNIFICANT' if p < 0.05 else 'not significant'}")

    print("\n" + "=" * 90)
    print(f"VALIDATED: pure LightGBM vs. naive baseline (forced min {FORCED_MIN}, cap {CAP}, "
          f"target half-width {TARGET_HW}pp -- the finding that started this whole investigation)")
    print("=" * 90)
    validated_rows = []
    for name, cfg in PRODUCT_CONFIG.items():
        r = validate_with_stopping_rule(
            lambda seed, cfg=cfg: run_one_seed_lgbm_vs_naive(seed, cfg),
            BATCH, CAP, TARGET_HW, forced_min=FORCED_MIN, label=name)
        sig = r["lo"] > 0 or r["hi"] < 0
        validated_rows.append(dict(product=name, n=r["n"], mean_pct=r["mean"], lo=r["lo"], hi=r["hi"],
                                    half_width=r["half_width"], significant=sig))
    validated = pd.DataFrame(validated_rows)
    print()
    print(validated.to_string(index=False))
    out_path2 = os.path.join(os.path.dirname(__file__), "..", "results", "01_lgbm_vs_naive_validated.csv")
    validated.to_csv(out_path2, index=False)
    print(f"\nSaved to {out_path2}")
