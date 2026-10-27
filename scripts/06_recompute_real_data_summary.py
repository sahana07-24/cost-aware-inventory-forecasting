"""
Re-summarize an already-saved real-data fold CSV -- the cluster-bootstrapped
mean, median, win-rate, and £-weighted pooled statistics for Comparisons A/B
(and C, when the column is present) -- WITHOUT re-running the full
walk-forward pipeline over the raw H&M transaction files.

WHY THIS EXISTS SEPARATELY FROM 04_real_data_validation.py: that script's
own summary block (which this one deliberately mirrors, statistic for
statistic) only runs at the end of a full walk-forward pass, which needs
transactions_train.csv and articles.csv in data/hm/ and can take a while at
the larger article counts. Once a fold-level CSV has already been produced
and saved to results/ (04_real_data_folds_N300.csv, _N750.csv, _N1500.csv,
...), this script recomputes the same summary from that CSV alone --
useful for regenerating a write-up table, or checking a stored result,
without needing the raw data files or the runtime of a fresh pass.

Run: python 06_recompute_real_data_summary.py [folds_csv_filename]
     (filename is resolved inside results/; defaults to
     04_real_data_folds_N300.csv, i.e. what 04_real_data_validation.py
     produces with its own default N_TOP_ARTICLES=300. Pass
     04_real_data_folds_N1500.csv for the largest, headline-result sample --
     see PROJECT_JOURNEY.md Section 4.)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bootstrap import bootstrap_ci_clustered, bootstrap_ratio_ci_clustered

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEFAULT_FILENAME = "04_real_data_folds_N300.csv"

COMPARISONS = [
    ("Adaptive vs Fixed (the thesis)", "adaptive_vs_fixed_pct", "cost_adaptive", "cost_fixed"),
    ("Adaptive vs Naive (Comparison A)", "adaptive_vs_naive_pct", "cost_adaptive", "cost_naive"),
    ("Adaptive vs Standard MSE (closes RQ2)", "adaptive_vs_mse_pct", "cost_adaptive", "cost_mse"),
]

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILENAME
    in_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(in_path):
        print(f"No such results file: {in_path}\n"
              f"Run 04_real_data_validation.py first, or pass the filename of an "
              f"existing results/*.csv fold file as an argument.")
        sys.exit(0)

    fdf = pd.read_csv(in_path)
    n_articles = fdf["article_id"].nunique()
    print(f"Source: {filename}")
    print(f"n folds={len(fdf)}, n articles={n_articles}\n")

    for label, col, num_col, den_col in COMPARISONS:
        if col not in fdf.columns:
            # Older fold files (saved before the standard-MSE arm existed)
            # only have the fixed/naive columns -- skip silently rather than
            # crash on a column that was never computed for this run.
            continue

        print(f"{label}:")
        finite_mask = np.isfinite(fdf[col])
        n_bad = (~finite_mask).sum()
        if n_bad:
            bad = fdf.loc[~finite_mask, ["article_id", "fold", "cost_naive", "cost_fixed", "cost_adaptive"]]
            print(f"  WARNING: {n_bad} fold(s) have a zero denominator ({den_col}=0 exactly). "
                  f"Excluded from mean/CI below, NOT from underlying data:")
            print(bad.to_string(index=False))

        clean = fdf.loc[finite_mask]
        mean, lo, hi, hw = bootstrap_ci_clustered(clean, col)
        median = fdf[col].median()
        win_rate = (fdf["cost_adaptive"] < fdf[den_col]).mean()
        print(f"  mean={mean:+.2f}%  cluster-bootstrap 95% CI=[{lo:+.2f}%, {hi:+.2f}%]  "
              f"(n={len(clean)}/{len(fdf)} folds, {clean['article_id'].nunique()} articles, half-width={hw:.2f}pp)")
        print(f"  median={median:+.2f}%   win-rate(adaptive cheaper)={win_rate:.1%}  [all {len(fdf)} folds]")

        pooled, plo, phi, phw = bootstrap_ratio_ci_clustered(fdf, num_col, den_col)
        print(f"  pooled £-weighted: {pooled:+.2f}%  cluster-bootstrap 95% CI=[{plo:+.2f}%, {phi:+.2f}%]  "
              f"half-width={phw:.2f}pp")
        print()
