"""The app's model layer must reproduce the validated evaluate_gated_hybrid() costs exactly.

Run: python -m pytest tests/
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model_core import CostSettings, run_scenario  # noqa: E402
from cost_model import C_O, C_U, Q_STAR, cv_to_bin  # noqa: E402
from gated_hybrid import evaluate_gated_hybrid  # noqa: E402
from synthetic_data import FULL_FEATURES, PRODUCT_CONFIG, make_product_series, train_test_split  # noqa: E402


def test_default_costs_match_src():
    c = CostSettings()
    assert c.c_o == pytest.approx(C_O)
    assert c.c_u == pytest.approx(C_U)
    assert c.q_star == pytest.approx(Q_STAR)


@pytest.mark.parametrize("product", list(PRODUCT_CONFIG))
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_costs_identical_to_evaluate_gated_hybrid(product, seed):
    df = make_product_series(seed=seed, **PRODUCT_CONFIG[product])
    train, test = train_test_split(df)
    vmask = test["true_cv_regime"].apply(cv_to_bin).values == "volatile"
    ref = evaluate_gated_hybrid(train, test, FULL_FEATURES, Q_STAR, C_O, C_U, seed=seed, mask=vmask)

    app = run_scenario(product, seed, CostSettings(), evaluate_on="volatile")["summary"]
    for m in ["naive", "fixed", "adaptive", "mse"]:
        assert app[m]["cost_per_week"] == pytest.approx(ref[f"cost_{m}"], rel=1e-12), m
