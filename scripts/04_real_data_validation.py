"""
Real-data validation on the H&M Personalized Fashion Recommendations dataset
-- the same walk-forward, cluster-bootstrapped validation this project ran
via phase5_hm_validation.py, now built on the consolidated src/ modules
(shared cost model, shared gated-hybrid evaluation, shared bootstrap code).

BEFORE RUNNING: place transactions_train.csv and articles.csv in data/hm/,
alongside this pipeline's own scripts/ and src/ folders (DATA_DIR below is
resolved relative to this script's own location, not hardcoded to any one
machine). Only t_dat/article_id/price and article_id/product_type_name/
index_name are required (see real_data.py).

Run: python 04_real_data_validation.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

import numpy as np

from real_data import load_top_articles, build_weekly_panel, evaluate_article_walk_forward
from bootstrap import bootstrap_ci_clustered, bootstrap_ratio_ci_clustered

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "hm")
N_TOP_ARTICLES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
RESULTS_SUFFIX = f"_N{N_TOP_ARTICLES}"

if __name__ == "__main__":
    if not os.path.exists(os.path.join(DATA_DIR, "transactions_train.csv")):
        print(f"H&M data not found in {DATA_DIR}. Download from Kaggle and place the CSVs there, then re-run.")
        sys.exit(0)

    tx = load_top_articles(DATA_DIR, N_TOP_ARTICLES)
    panels = build_weekly_panel(tx)
    print(f"Built weekly panels for {len(panels)} articles (target was {N_TOP_ARTICLES}).\n")

    all_folds = []
    for article_id, df in panels.items():
        print(f"Article {article_id}: {len(df)} weeks of history, "
              f"empirical CV range [{df['rolling_cv'].min():.2f}, {df['rolling_cv'].max():.2f}], "
              f"median CV={df['rolling_cv'].median():.2f}")
        folds = evaluate_article_walk_forward(article_id, df)
        if not folds:
            print("  Skipped (insufficient history for walk-forward folds)\n")
            continue
        for r in folds:
            print(f"  Fold {r['fold']}: train={r['n_train']}wk test={r['n_test']}wk  "
                  f"q*={r['q_star']:.3f}  gate_threshold={r['gate_threshold']:.2f}")
            print(f"    naive=£{r['cost_naive']:.2f}  fixed=£{r['cost_fixed']:.2f}  "
                  f"adaptive=£{r['cost_adaptive']:.2f}  mse=£{r['cost_mse']:.2f}")
        all_folds.extend(folds)
        print()

    if not all_folds:
        print("No articles produced valid folds.")
        sys.exit(0)

    fdf = pd.DataFrame(all_folds)
    fdf["adaptive_vs_fixed_pct"] = 100 * (fdf["cost_fixed"] - fdf["cost_adaptive"]) / fdf["cost_fixed"]
    fdf["adaptive_vs_naive_pct"] = 100 * (fdf["cost_naive"] - fdf["cost_adaptive"]) / fdf["cost_naive"]
    fdf["adaptive_vs_mse_pct"] = 100 * (fdf["cost_mse"] - fdf["cost_adaptive"]) / fdf["cost_mse"]
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", f"04_real_data_folds{RESULTS_SUFFIX}.csv")
    fdf.to_csv(out_path, index=False)

    print("=" * 70)
    print(f"SUMMARY: {len(fdf)} (article x fold) observations across {fdf['article_id'].nunique()} article(s)")
    print("=" * 70)
    n_articles = fdf["article_id"].nunique()
    if n_articles < 8:
        print(f"NOTE: only {n_articles} articles contributed folds -- cluster-bootstrap CIs need more "
              f"clusters than this to be stable. Treat as illustrative only.\n")

    for label, col, num_col, den_col in [
        ("Adaptive vs Fixed (the thesis)", "adaptive_vs_fixed_pct", "cost_adaptive", "cost_fixed"),
        ("Adaptive vs Naive (Comparison A)", "adaptive_vs_naive_pct", "cost_adaptive", "cost_naive"),
        ("Adaptive vs Standard MSE (closes RQ2)", "adaptive_vs_mse_pct", "cost_adaptive", "cost_mse"),
    ]:
        print(f"{label}:")
        finite_mask = np.isfinite(fdf[col])
        n_bad = (~finite_mask).sum()
        if n_bad:
            bad = fdf.loc[~finite_mask, ["article_id", "fold", "cost_naive", "cost_fixed", "cost_adaptive"]]
            print(f"  WARNING: {n_bad} fold(s) have a zero denominator ({den_col}=0 exactly -- a near-zero-"
                  f"volume article-fold where the clipped prediction matched actual demand in every test "
                  f"week), making the per-fold percentage undefined (+/-inf). Excluded from the mean/CI "
                  f"below, NOT from the underlying data:")
            print(bad.to_string(index=False))

        clean = fdf.loc[finite_mask]
        mean, lo, hi, hw = bootstrap_ci_clustered(clean, col)
        median = fdf[col].median()  # median is robust to the zero-denominator rows -- keep all data
        win_rate = (fdf["cost_adaptive"] < fdf[den_col]).mean()
        print(f"  mean={mean:+.2f}%  cluster-bootstrap 95% CI=[{lo:+.2f}%, {hi:+.2f}%]  "
              f"(n={len(clean)}/{len(fdf)} article-fold pairs, {clean['article_id'].nunique()} articles, "
              f"half-width={hw:.2f}pp)  [mean-of-per-fold-%, equal-weighted per fold]")
        print(f"  median={median:+.2f}%   win-rate(adaptive cheaper)={win_rate:.1%}  [all {len(fdf)} folds]")

        pooled, plo, phi, phw = bootstrap_ratio_ci_clustered(fdf, num_col, den_col)
        print(f"  pooled £-weighted: {pooled:+.2f}%  cluster-bootstrap 95% CI=[{plo:+.2f}%, {phi:+.2f}%]  "
              f"half-width={phw:.2f}pp  [total £{den_col.split('_')[1]} cost vs total £adaptive cost, "
              f"summed across all {len(fdf)} folds -- weights each fold by its actual £ size, not equally]")
        print()

    print(f"\nSaved fold-level detail to {out_path}")
    print("\nPer-article-fold detail:")
    print(fdf[["article_id", "fold", "n_train", "n_test",
               "cost_naive", "cost_fixed", "cost_adaptive"]].to_string(index=False))
