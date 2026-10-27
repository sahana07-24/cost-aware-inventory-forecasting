"""
Real-data (H&M) pipeline: weekly aggregation from raw transactions, a
leak-free feature pipeline, walk-forward validation, and the gate threshold
recalibrated per-article-per-fold from that fold's own training data.

This is phase5_hm_validation.py's logic, refactored to share cost_model.py
and gated_hybrid.py with the synthetic pipeline instead of duplicating them,
plus the cost-model harmonization noted in cost_model.py's docstring (the
restock surcharge is now included for real articles too, matching the
documented business scenario).

THREE THINGS THAT DON'T TRANSFER FROM SYNTHETIC TO REAL DATA (unchanged from
phase5_hm_validation.py's original notes -- still true here):
  1. The calm/moderate/volatile CV thresholds were inspired by synthetic-like
     assumptions; real H&M CV distributions are printed per-article so this
     can be checked empirically, not assumed.
  2. The gate threshold (4.6, calibrated on synthetic no-trend data, in
     synthetic-demand-scale units) does NOT transfer numerically. This
     pipeline recalibrates it from EACH fold's own training data: the 75th
     percentile of that fold's rolling trend-slope distribution, chosen
     before looking at test-period outcomes -- same principle as the
     synthetic gate, different numbers.
  3. Whether real weekly demand partitions into clean calm/volatile regimes
     the way the synthetic generator was designed to is an open question,
     not an assumption going in.
"""
import os
import numpy as np
import pandas as pd

from cost_model import cost_params_from_price
from gated_hybrid import evaluate_gated_hybrid
from synthetic_data import rolling_ols_slope

MIN_WEEKS_HISTORY = 40
ROLLING_WINDOW = 12

REQUIRED_TX_COLUMNS = ["t_dat", "article_id", "price"]
REQUIRED_ARTICLE_COLUMNS = ["article_id", "product_type_name", "index_name"]

FEATURES = ["weekofyear_sin", "weekofyear_cos", "promo_flag", "lag_1", "lag_2",
            "week_norm", "rolling_mean", "rolling_std", "rolling_cv",
            "rolling_trend_slope", "trend_vol_interaction"]


def validate_schema(tx: pd.DataFrame, articles: pd.DataFrame) -> None:
    missing_tx = [c for c in REQUIRED_TX_COLUMNS if c not in tx.columns]
    missing_art = [c for c in REQUIRED_ARTICLE_COLUMNS if c not in articles.columns]
    if missing_tx or missing_art:
        msg = ["Schema mismatch -- the real CSVs don't have the columns this pipeline needs:"]
        if missing_tx:
            msg.append(f"  transactions_train.csv missing: {missing_tx}")
            msg.append(f"  transactions_train.csv actually has: {list(tx.columns)}")
        if missing_art:
            msg.append(f"  articles.csv missing: {missing_art}")
            msg.append(f"  articles.csv actually has: {list(articles.columns)}")
        raise ValueError("\n".join(msg))
    print(f"Schema check passed: transactions_train.csv has {list(tx.columns)}")
    print(f"                     articles.csv has {len(articles.columns)} columns "
          f"(checked: {REQUIRED_ARTICLE_COLUMNS})\n")


def load_top_articles(data_dir: str, n_top: int) -> pd.DataFrame:
    tx = pd.read_csv(os.path.join(data_dir, "transactions_train.csv"), parse_dates=["t_dat"])
    articles = pd.read_csv(os.path.join(data_dir, "articles.csv"))
    validate_schema(tx, articles)
    top_ids = tx["article_id"].value_counts().head(n_top).index
    tx = tx[tx["article_id"].isin(top_ids)]
    tx = tx.merge(articles[["article_id", "product_type_name", "index_name"]], on="article_id", how="left")
    return tx


def build_weekly_panel(tx: pd.DataFrame) -> dict:
    tx = tx.copy()
    tx["week"] = tx["t_dat"].dt.to_period("W").dt.start_time
    panels = {}
    for article_id, group in tx.groupby("article_id"):
        weekly = group.groupby("week").agg(demand=("article_id", "count"), price=("price", "mean")).reset_index()
        full_range = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq="W-MON")
        weekly = weekly.set_index("week").reindex(full_range).rename_axis("week").reset_index()
        weekly["demand"] = weekly["demand"].fillna(0)
        # ffill only -- a leading NaN can't occur (a panel's first row is
        # always its article's actual first transaction week), but ffill
        # rather than bfill keeps this leak-free even if that assumption
        # changes later.
        weekly["price"] = weekly["price"].ffill()
        if len(weekly) < MIN_WEEKS_HISTORY:
            continue

        wk_num = weekly["week"].dt.isocalendar().week.astype(int)
        weekly["weekofyear_sin"] = np.sin(2 * np.pi * wk_num / 52)
        weekly["weekofyear_cos"] = np.cos(2 * np.pi * wk_num / 52)
        weekly["week_norm"] = np.arange(len(weekly)) / len(weekly)
        # No fillna(mean) anywhere below -- that mean would be computed over
        # the full train+test panel, leaking future information into early
        # rows. Leading NaNs are dropped naturally by dropna() below instead.
        weekly["lag_1"] = weekly["demand"].shift(1)
        weekly["lag_2"] = weekly["demand"].shift(2)
        weekly["rolling_mean"] = weekly["demand"].rolling(ROLLING_WINDOW, min_periods=4).mean().shift(1)
        weekly["rolling_std"] = weekly["demand"].rolling(ROLLING_WINDOW, min_periods=4).std().shift(1)
        weekly["rolling_cv"] = (weekly["rolling_std"] / weekly["rolling_mean"].replace(0, np.nan)).fillna(0)
        weekly["rolling_trend_slope"] = pd.Series(
            rolling_ols_slope(weekly["demand"].values, ROLLING_WINDOW)
        ).shift(1).fillna(0.0).values
        weekly["trend_vol_interaction"] = weekly["rolling_trend_slope"] * weekly["rolling_cv"]
        # H&M's base transactions file has no promotions field -- disclosed
        # as a feature gap versus the synthetic pipeline, not silently filled.
        weekly["promo_flag"] = 0

        panels[article_id] = weekly.dropna().reset_index(drop=True)
    return panels


def evaluate_one_fold(article_id, train: pd.DataFrame, test: pd.DataFrame,
                       markup_ratio: float = 3.0) -> dict:
    if len(train) < 20 or len(test) < 5:
        return None
    mean_price = train["price"].mean()
    c_o, c_u, q_star = cost_params_from_price(mean_price, markup_ratio=markup_ratio)
    # Gate threshold recalibrated from THIS fold's own real training data --
    # not the synthetic-derived 4.6 (see module docstring).
    gate_threshold = train["rolling_trend_slope"].quantile(0.75)
    result = evaluate_gated_hybrid(train, test, FEATURES, q_star, c_o, c_u,
                                    gate_threshold=gate_threshold)
    result["article_id"] = article_id
    return result


def evaluate_article_walk_forward(article_id, df: pd.DataFrame, n_folds: int = 3,
                                   test_weeks: int = 10) -> list:
    """Rolling, expanding-window folds (non-overlapping test windows moving
    forward) -- real data has no seed dimension, so folds are the
    replication unit here (see bootstrap.py's clustered bootstrap)."""
    results = []
    n = len(df)
    if n < 20 + n_folds * test_weeks:
        return results
    for i in range(n_folds):
        test_end = n - (n_folds - 1 - i) * test_weeks
        test_start = test_end - test_weeks
        train, test = df.iloc[:test_start].copy(), df.iloc[test_start:test_end].copy()
        r = evaluate_one_fold(article_id, train, test)
        if r:
            r["fold"] = i
            results.append(r)
    return results
