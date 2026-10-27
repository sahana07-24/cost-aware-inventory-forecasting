"""
Statistical validation: bootstrap confidence intervals and the pre-committed
stopping rule used throughout this project.

TWO DIFFERENT REPLICATION UNITS, deliberately not treated the same way:
  - Synthetic data: each random seed is an independent draw of an entire
    demand history. A flat bootstrap over seeds is correct here -- seeds
    really are i.i.d.
  - Real data: each article has only ONE real transaction history, evaluated
    across several correlated walk-forward folds. Folds within an article
    share the same underlying demand process and overlapping/expanding
    training windows -- NOT independent draws. Resampling rows would
    understate uncertainty (CIs too tight). bootstrap_ci_clustered resamples
    whole ARTICLES instead, taking all of a sampled article's folds together.

STOPPING RULE (used identically across every diagnostic and headline result
in this project): run in batches, recompute the bootstrap CI after each
batch, stop once the CI half-width drops below a pre-committed target,
capped at a maximum sample size. The target and cap are chosen BEFORE
looking at any result, per batch -- see individual scripts for the specific
values used (looser bars for quick diagnostics, tighter for headline claims).
"""
import numpy as np
import pandas as pd


def bootstrap_ci(values, n_boot: int = 5000, seed: int = 0):
    """Flat bootstrap over independent values (e.g. one value per synthetic
    seed). Returns (mean, lo, hi, half_width)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                            for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi), float((hi - lo) / 2)


def bootstrap_ci_clustered(df: pd.DataFrame, col: str, group_col: str = "article_id",
                            n_boot: int = 5000, seed: int = 0):
    """Cluster bootstrap over GROUPS (e.g. real-data articles), not over
    pooled (group, fold) rows -- see module docstring. Each draw resamples
    whole groups with replacement; a group drawn twice is counted twice
    (concatenate explicitly, don't isin()/dedupe)."""
    rng = np.random.default_rng(seed)
    group_ids = df[group_col].unique()
    groups = {gid: g[col].to_numpy() for gid, g in df.groupby(group_col)}
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        boot_means[b] = np.concatenate([groups[gid] for gid in sampled]).mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(df[col].mean()), float(lo), float(hi), float((hi - lo) / 2)


def bootstrap_ratio_ci_clustered(df: pd.DataFrame, numerator_col: str, denominator_col: str,
                                  group_col: str = "article_id", n_boot: int = 5000, seed: int = 0):
    """Cluster-bootstrapped CI for a POOLED ratio -- e.g. total £ cost under
    method A vs total £ cost under method B, summed across all folds of all
    resampled articles, not averaged per-fold-percentage.

    This is deliberately a different statistic from bootstrap_ci_clustered
    applied to a pre-computed per-row percentage column. A mean-of-percentages
    treats a near-zero-cost article-fold (£0.0002/week) as equally important
    as a high-cost one (£2/week), and is undefined outright when a fold's
    denominator is exactly zero. Pooling totals first weights every fold by
    its actual economic size and sidesteps the zero-denominator problem
    structurally (a single zero-cost fold barely moves a sum). Report both;
    they answer related but different questions.
    """
    # Pre-aggregate to one (num_sum, den_sum) pair per group -- resampling
    # then only needs to add up ~n_groups scalars per draw, not concatenate
    # whole DataFrames 5000 times (which is what made an earlier version of
    # this function time out on 282 groups).
    per_group = df.groupby(group_col)[[numerator_col, denominator_col]].sum()
    group_ids = per_group.index.to_numpy()
    num_sums = per_group[numerator_col].to_numpy()
    den_sums = per_group[denominator_col].to_numpy()
    idx_by_id = {gid: i for i, gid in enumerate(group_ids)}

    def pooled_pct(num_total: float, den_total: float) -> float:
        return 100 * (den_total - num_total) / den_total

    point = pooled_pct(num_sums.sum(), den_sums.sum())
    rng = np.random.default_rng(seed)
    n_groups = len(group_ids)
    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n_groups, size=n_groups)
        boot_vals[b] = pooled_pct(num_sums[sample_idx].sum(), den_sums[sample_idx].sum())
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return float(point), float(lo), float(hi), float((hi - lo) / 2)


def validate_with_stopping_rule(run_one_fn, batch: int = 50, cap: int = 300,
                                 target_halfwidth: float = 1.0, forced_min: int = None,
                                 label: str = "", verbose: bool = True):
    """Runs `run_one_fn(seed)` in batches, stopping once the bootstrap CI
    half-width drops below `target_halfwidth`, capped at `cap` seeds. If
    `forced_min` is set, keeps running past an early "converged" read until
    at least that many valid seeds are collected -- this project's own
    history (Technical_Validation_Summary Section 4) found an early stop at
    n=50-250 produced an apparent null that reversed to significant once
    forced to 400+; forced_min exists specifically to not repeat that.

    run_one_fn(seed) should return a float, or None to skip an invalid seed
    (e.g. too few volatile-regime weeks in that draw).
    """
    vals = []
    seed_counter = 0
    mean = lo = hi = hw = float("nan")
    while seed_counter < cap:
        for s in range(seed_counter, seed_counter + batch):
            v = run_one_fn(s)
            if v is not None:
                vals.append(v)
        seed_counter += batch
        arr = np.array(vals)
        if len(arr) > 1:
            mean, lo, hi, hw = bootstrap_ci(arr)
            past_forced_min = forced_min is None or len(arr) >= forced_min
            if hw < target_halfwidth and past_forced_min:
                break
    if verbose and len(vals) > 1:
        print(f"  {label:28s}: n={len(vals):4d}  mean={mean:+.2f}%  "
              f"CI=[{lo:+.2f}%, {hi:+.2f}%]  half-width={hw:.2f}pp")
    return dict(label=label, n=len(vals), mean=mean, lo=lo, hi=hi, half_width=hw, values=vals)
