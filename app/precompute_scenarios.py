"""Precompute 100 scenarios per product at the dissertation's default costs, so the app can show the
typical result instantly. Re-run after changing src/:  python app/precompute_scenarios.py"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_core import METHODS, PRODUCT_CONFIG, CostSettings, run_scenario  # noqa: E402

N_SEEDS = 100

if __name__ == "__main__":
    rows = []
    for product in PRODUCT_CONFIG:
        for seed in range(N_SEEDS):
            r = run_scenario(product, seed, CostSettings(), evaluate_on="volatile")
            if r["n_eval_weeks"] < 4:
                continue
            rows.append(dict(product=product, seed=seed,
                             **{f"cost_{m}": r["summary"][m]["cost_per_week"] for m in METHODS}))
        print(product, "done")
    out = os.path.join(HERE, "data", "default_scenarios.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)
