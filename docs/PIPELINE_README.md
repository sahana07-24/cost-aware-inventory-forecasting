# Cost-Sensitive Adaptive-Quantile Demand Forecasting — Consolidated Pipeline

This is the dissertation's codebase, rebuilt from scratch as one coherent
package. It is **not** a reinvention of the modelling approach — every piece
of logic here (the synthetic data generator, the newsvendor cost model, the
baselines, the gated-hybrid architecture, the bootstrap validation) is taken
from scripts that were already built and validated over the course of this
project (`baselines.py`, `hybrid_correction.py`, `per_week_diagnosis.py`,
`trend_sign_test.py`, `trend_slope_fix.py`, `phase5_hm_validation.py`).
What's new is that it's now one package instead of five scripts that each
redefined the same ~90 lines of cost model, data generator, and feature
engineering with small, easy-to-miss inconsistencies between them.

For the full narrative of *why* the architecture looks like this (the
extrapolation bug, the hybrid fix, the trend-sign diagnosis, the gate),
see `../Technical_Validation_Summary.docx` (in the `files1.zip`/`files.zip`
archives) and `../../Downloads/Project_Explainer.docx`. This README covers
the code, not the story.

## One deliberate change: the real-data cost model

`phase5_hm_validation.py` (the real-data script) computed the underage cost
`c_u` as margin-only (`price - unit_cost`), while every synthetic script
included a restock surcharge (`c_u = (price - unit_cost) + 0.12 * unit_cost`)
matching the documented business scenario (`£106` stockout cost = `£100`
margin + `£6` surcharge). `src/cost_model.py` applies the surcharge
convention to real articles too, via `cost_params_from_price()`. This moves
real-data `q*` from `0.8889` to `~0.8945` — a small but real difference from
the numbers in `phase5_hm_validation_output_N300.txt`. Flagged here, not
buried in a diff.

## Structure

```
pipeline/
├── src/
│   ├── cost_model.py     # newsvendor economics: C_O, C_U, Q_STAR, tau_bins_for, cv_to_bin
│   ├── synthetic_data.py # Products A/B/C, the trend-sign sweep, make_product_series
│   ├── gated_hybrid.py   # the core method: residual bank, gate, evaluate_gated_hybrid
│   ├── baselines.py      # naive / Holt-Winters / step-heuristic baselines
│   ├── bootstrap.py      # bootstrap_ci, bootstrap_ci_clustered, pre-committed stopping rule
│   └── real_data.py      # H&M weekly aggregation, schema check, walk-forward folds
├── scripts/
│   ├── 01_baseline_comparison.py          # pure LightGBM vs 3 simple baselines
│   ├── 02_gate_diagnostic_trend_sign.py   # trend-sign test + gate before/after
│   ├── 03_comparison_A_and_B_gated_hybrid.py  # THE HEADLINE RESULT (synthetic)
│   ├── 04_real_data_validation.py         # THE HEADLINE RESULT (real H&M data)
│   ├── 05_rq3_stockout_risk.py            # RQ3: is demand-CV a practical stockout-risk signal?
│   └── 06_recompute_real_data_summary.py  # re-summarize a saved results/04_*.csv, no raw data needed
├── results/               # CSV output from every script lands here
├── requirements.txt
└── README.md
```

## The two comparisons that matter

Both come out of the **same trained model bank** per seed/fold — see
`gated_hybrid.evaluate_gated_hybrid()`, which returns `cost_naive`,
`cost_fixed`, and `cost_adaptive` together:

- **Comparison A** (credibility check): gated hybrid (`adaptive`) vs. naive
  baseline (`naive`). Not the thesis — establishes the method isn't losing
  to a rolling-mean-plus-safety-stock formula.
- **Comparison B** (the thesis): adaptive vs. fixed target (`adaptive` vs.
  `fixed`), *within* the same gated-hybrid architecture. Does letting the
  quantile target adapt to the live volatility regime beat using one fixed
  quantile throughout?

## Running

```bash
pip install -r requirements.txt
cd scripts
python 01_baseline_comparison.py
python 02_gate_diagnostic_trend_sign.py
python 03_comparison_A_and_B_gated_hybrid.py
python 04_real_data_validation.py   # needs transactions_train.csv + articles.csv in DATA_DIR
python 05_rq3_stockout_risk.py
python 06_recompute_real_data_summary.py [folds_csv_filename]  # re-derives the summary
    # stats from an already-saved results/04_real_data_folds_N*.csv, without
    # needing the raw H&M CSVs or a fresh walk-forward pass
```

Each script's docstring states its stopping rule (target CI half-width,
batch size, cap) and where that rule came from in the project's history —
none of them were chosen post-hoc to match a desired answer.

## What's intentionally NOT reproduced here

`per_week_diagnosis.py`'s specific plotting/hypothesis-testing code (the
"is the shortfall concentrated at regime transition" check) isn't ported —
it was a rejected hypothesis (p=0.52/0.14/0.78), kept in the zip archives as
a record of a dead end properly tested and reported, not as reusable
pipeline logic. The trend-sign *mechanism* it helped rule in (Section 3.2 of
Technical_Validation_Summary) is what `02_gate_diagnostic_trend_sign.py`
reproduces instead.
