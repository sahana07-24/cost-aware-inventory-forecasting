"""
RQ3 validation: is demand uncertainty (rolling CV) a practical, usable
indicator of stockout risk?

RQ1 already shows the target quantile CAN be adapted with volatility (see
05's sibling diagram, ppt/charts/utility_diagram.py, kept outside this
pipeline as a presentation asset). RQ3 asks a different, prior question:
is the volatility signal itself actually informative about risk, i.e. is
adapting on it a sensible thing to do in the first place?

DESIGN: for each of the three synthetic products, simulate many independent
demand series (same generator as every other script in this pipeline) and
evaluate two non-training policies week-by-week:
  - FIXED  : order quantity = rolling_mean + z(Q_STAR) * rolling_std
             (same target quantile regardless of current volatility --
             this is the project's own "fixed" baseline, see cost_model.py)
  - ADAPTIVE: order quantity uses assign_tau_adaptive(rolling_cv, Q_STAR)
             instead of the fixed Q_STAR -- the mechanism RQ1 validates.

A "stockout" is any week where realized demand exceeds the order quantity.
Weeks are grouped by the GENERATOR'S TRUE demand-CV regime (calm/volatile),
not by the noisy rolling_cv estimate itself -- an earlier attempt using the
raw rolling estimate produced a confusing, non-monotonic pattern, traced to
12-week rolling-window lag interacting with promotional demand spikes. The
true-regime grouping is the clean, honest way to answer RQ3; see
Project_Explainer_Full.md's RQ3 section for the full writeup of why.

This is a descriptive/illustrative simulation (large n -> precise point
estimates), not a hypothesis test like the Comparison A/B/C stopping-rule
validations elsewhere in this pipeline -- it's making a mechanism-validation
argument (does the signal track risk, and does acting on it help), not a
headline effect-size claim.

Run: python 05_rq3_stockout_risk.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import norm

from cost_model import Q_STAR, assign_tau_adaptive
from synthetic_data import PRODUCT_CONFIG, ROLLING_WINDOW, make_product_series

N_SEEDS = 400  # per product -- large pool so per-regime rates are precise, not just directionally right
BURN_IN = ROLLING_WINDOW + 3  # drop weeks before rolling stats have stabilised

Z_STAR_FIXED = norm.ppf(Q_STAR)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion -- better-behaved
    than a normal approximation at the low stockout rates this project sees."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return center - half, center + half


def simulate_product(product: str, cfg: dict, n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """One product's pooled (seed x week) panel, with both policies'
    stockout indicators and the week's true demand-CV regime."""
    rows = []
    for seed in range(n_seeds):
        df = make_product_series(seed=seed, **cfg).iloc[BURN_IN:]

        q_fixed = df["rolling_mean"] + Z_STAR_FIXED * df["rolling_std"]

        tau_adapt = df["rolling_cv"].apply(lambda cv: assign_tau_adaptive(cv, Q_STAR))
        z_adapt = norm.ppf(tau_adapt.clip(1e-6, 1 - 1e-6))
        q_adapt = df["rolling_mean"] + z_adapt * df["rolling_std"]

        rows.append(pd.DataFrame({
            "true_cv_regime": df["true_cv_regime"].values,
            "stockout_fixed": (df["demand"] > q_fixed).astype(int).values,
            "stockout_adapt": (df["demand"] > q_adapt).astype(int).values,
        }))
    pooled = pd.concat(rows, ignore_index=True)
    pooled.insert(0, "product", product)
    return pooled


if __name__ == "__main__":
    print(f"Simulating {N_SEEDS} seeds x {len(PRODUCT_CONFIG)} products "
          f"(calm + volatile regime weeks, both policies)...")

    all_pooled = pd.concat(
        [simulate_product(name, cfg) for name, cfg in PRODUCT_CONFIG.items()],
        ignore_index=True,
    )

    summary_rows = []
    for (product, regime), sub in all_pooled.groupby(["product", "true_cv_regime"], sort=False):
        n = len(sub)
        kf, ka = int(sub["stockout_fixed"].sum()), int(sub["stockout_adapt"].sum())
        f_lo, f_hi = wilson_ci(kf, n)
        a_lo, a_hi = wilson_ci(ka, n)
        summary_rows.append(dict(
            product=product, true_cv_regime=regime, n=n,
            fixed_rate=kf / n, fixed_lo=f_lo, fixed_hi=f_hi,
            adapt_rate=ka / n, adapt_lo=a_lo, adapt_hi=a_hi,
        ))
    summary = pd.DataFrame(summary_rows).sort_values(["product", "true_cv_regime"])

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "05_rq3_stockout_risk_summary.csv")
    summary.to_csv(out_path, index=False)

    print("\n" + "=" * 100)
    print("RQ3: realized stockout rate by TRUE demand-CV regime -- fixed vs. adaptive policy")
    print("=" * 100)
    for row in summary.itertuples():
        print(f"  {row.product:12s} cv={row.true_cv_regime:.2f}  n={row.n:6d}  "
              f"fixed={row.fixed_rate:6.2%} [{row.fixed_lo:.2%}, {row.fixed_hi:.2%}]  "
              f"adaptive={row.adapt_rate:6.2%} [{row.adapt_lo:.2%}, {row.adapt_hi:.2%}]")

    print("\n" + "=" * 100)
    print("Calm -> volatile relative increase in stockout risk under the FIXED policy")
    print("(the direct answer to RQ3: does the true volatility regime predict risk?)")
    print("=" * 100)
    for product in PRODUCT_CONFIG:
        sub = summary[summary["product"] == product].sort_values("true_cv_regime")
        calm, vol = sub.iloc[0], sub.iloc[1]
        delta = (vol.fixed_rate / calm.fixed_rate - 1) * 100
        reduction = (1 - vol.adapt_rate / vol.fixed_rate) * 100
        print(f"  {product:12s}: calm={calm.fixed_rate:.2%} -> volatile={vol.fixed_rate:.2%}  "
              f"({delta:+.0f}% relative)   adaptive policy reduces volatile-regime risk by {reduction:.0f}%")

    print(f"\nSaved to {out_path}")
