# Project Journey: Pipeline, Bugs, and How We Reached Each Conclusion

This document records the technical development of *Cost-Sensitive Adaptive Demand
Forecasting for Fashion Retail Inventory* as it actually happened — the pipeline as
built, every bug that was found and fixed along the way, and how each research
question's conclusion was reached. It is written to sit alongside the code in this
repository as a permanent record, and to be a direct reference for viva preparation.

---

## 1. The Pipeline, in the Order It Runs

| # | Script | What it does | Answers |
|---|--------|---------------|---------|
| 1 | `scripts/01_baseline_comparison.py` | Tests pure LightGBM (predicting demand directly) against a naive rolling mean + safety-stock heuristic. | Motivates the architecture — this is where the project discovered ML alone doesn't work. |
| 2 | `scripts/02_gate_diagnostic_trend_sign.py` | A controlled sweep, varying only trend direction, to test whether the ML residual correction is safe under all trend conditions. | Motivates the gate. |
| 3 | `scripts/03_comparison_A_and_B_gated_hybrid.py` | The headline synthetic-data result: Comparisons A (vs. naive), B (adaptive vs. fixed — the thesis), and C (vs. standard MSE), across all three synthetic products. | RQ1, RQ2 (synthetic). |
| 4 | `scripts/04_real_data_validation.py` | The same three comparisons, run on real H&M transaction data via walk-forward validation. | RQ2 (real-data). |
| 5 | `scripts/05_rq3_stockout_risk.py` | Tests whether true demand-volatility regime predicts realized stockout risk, and whether the adaptive policy reduces that risk. | RQ3. |

Every script is deterministic (seeded) or reads the same real H&M data, self-contained,
and runnable with `pip install -r requirements.txt` followed by `python <script>.py`
from the `scripts/` folder.

---

## 2. Bugs Found and Fixed, in the Order They Were Discovered

### 2.1 The extrapolation bug (synthetic data design)

**Symptom:** an early version of the synthetic generator placed its volatile-demand
block only in the test period, never in training.

**Root cause:** tree-based models like LightGBM cannot extrapolate beyond the range of
feature values they saw during training. Since `rolling_cv` never reached volatile
levels in training, the model had no learned behaviour for volatile-regime inputs at
test time.

**Fix:** the volatile-demand block is now cycled through the 260-week series **three
times** (`VOLATILE_BLOCK_BOUNDS` in `src/synthetic_data.py`), so training data always
contains genuine volatile-regime examples. Checked for leakage of its own (an injected
demand spike, confirming rolling-window features immediately before it were unaffected)
and confirmed clean.

### 2.2 Pure LightGBM loses to a naive heuristic (Script 01)

**Finding:** even after fixing the extrapolation bug, a pure LightGBM model — trained
once, then frozen — lost to a simple naive rolling-mean-plus-safety-stock heuristic by
68–242%, depending on the product (800 seeds each, all significant).

**Why:** the naive heuristic recomputes its estimate fresh every week directly from
real recent observations; a frozen, batch-trained model cannot match that live-updating
advantage, and its own learned corrections can introduce error on top of an
already-accurate signal rather than improving it.

**Consequence:** this is the origin of the hybrid architecture — instead of predicting
demand directly, LightGBM was redesigned to predict only the *residual* on top of the
naive baseline, inheriting the baseline's live-updating advantage while still allowing
ML to contribute whatever additional pattern it can find.

### 2.3 The ungated residual correction is harmful under strong positive trend (Script 02)

**Finding:** a controlled trend-sign sweep (5 conditions, identical volatility, only
trend direction/magnitude varied) showed the *ungated* hybrid's advantage over naive
shrinks monotonically as trend turns strongly positive (R² = 0.882, p = 0.0178) — the
ML correction is actively harmful in that condition, not merely unhelpful.

**Fix:** a smooth sigmoid gate, driven by `rolling_trend_slope`
(`gate_weight()` in `src/gated_hybrid.py`), shrinks the residual correction toward zero
as trend rises past a threshold (4.6 for the synthetic data, calibrated from the 75th
percentile of trend-slope observed in training data under a no-trend condition —
decided before any post-gate outcome was examined). With the gate in place, performance
stays flat across the whole trend sweep instead of degrading.

### 2.4 The carrying-rate unit-conversion bug — the most consequential bug in the project

**Symptom:** `CARRYING_RATE = 0.25` was documented, in every earlier version of
`src/cost_model.py`, as a rate *per year* — but every cost function in the project
operates on *weekly* periods (one row per product-week), and the rate was being applied
directly with no annual-to-weekly conversion.

**Consequence:** this silently treated 25%-per-year as if it were 25%-per-week — a
~52× overstatement of the true weekly holding cost (C_o).

**Fix:** divide the annual rate by 52 before using it (`C_O = (CARRYING_RATE /
WEEKS_PER_YEAR) * UNIT_COST`). A straightforward unit-conversion fix, not a modelling
judgement call.

**Downstream effects, both required their own fix:**
- Correcting the rate moved **Q_STAR from ≈0.8945 to ≈0.9977** — holding a unit became
  ~52× cheaper relative to stocking out, so the newsvendor solution pushed the optimal
  fixed quantile much closer to 1.
- At that level, the *original* adaptive scheme — which interpolated the target
  quantile in **probability space** up to a fixed ceiling of 0.95 — had nowhere left to
  go: 0.95 < 0.9977, so every volatility regime collapsed to the identical quantile.
  This was fixed by redesigning the interpolation to happen in **z-space** instead
  (`assign_tau_adaptive()`), where equal steps still move real order quantities even
  near q=1, using `ADAPTIVE_Z_SPAN` (≈0.394) — the exact width, in z-units, that the
  original design swept between its two original documented values — so the fix
  preserves the original design's intended amount of adaptivity rather than being
  re-tuned to produce a particular outcome.
- **On real H&M data, this bug fix flipped the headline result's sign.** Before the
  fix, adaptive vs. fixed was significantly *negative* (−3.57%, 95% CI
  [−5.09%, −2.18%]) — the adaptive model was genuinely losing. After the fix, it flips
  to significantly *positive* (+7.15% to +7.22% pooled, 95% CI excluding zero, across
  1,197 real articles and 3,591 walk-forward folds).

**Why we're confident this is annual, not weekly:** the rate is declared "PER YEAR" in
the code's own comment and in the original documented business scenario; industry
benchmarks for fashion/apparel carrying cost sit at roughly 20–25% *per year*; and a
25%-*per-week* rate is economically implausible on its face — it would imply holding
one £50 garment costs £650 over a year, thirteen times the item's own value.

### 2.5 RQ3's rolling-CV-vs-true-regime confusion (Script 05)

**Symptom:** an early version of the RQ3 stockout-risk analysis grouped weeks by the
*noisy rolling CV estimate*, producing a confusing, non-monotonic pattern.

**Root cause:** rolling-window lag interacting with promo spikes meant the rolling
estimate didn't cleanly track which regime a product was actually in.

**Fix:** group by the generator's **ground-truth** CV regime (`true_cv_regime`,
tracked separately and never used as a model feature — only for evaluation) instead of
the noisy rolling estimate. With that fix, every product shows a clean, materially
higher stockout rate in its true volatile regime, and the adaptive policy reduces that
risk further in every product.

---

## 3. How Each Research Question's Conclusion Was Reached

**RQ1 — can the target quantile adapt to demand volatility?** Answered by the RQ1
utility diagram and `assign_tau_adaptive()`'s z-space redesign: the scheme produces a
real, distinct order quantity as volatility rises (285 → 296 units for a representative
product, calm to volatile), not just a theoretical probability shift. **Yes, cleanly.**

**RQ2 (synthetic) — does adaptation reduce cost?** Answered by Script 03's three
comparisons: vs. naive (+18.6% to +21.4%, all significant), vs. fixed target — the
actual thesis (+0.99% / +4.89% / +0.41%, only the most volatile product significant
alone), vs. standard MSE (+91.9% to +93.7%, decisive but mostly reflecting "any
cost-aware target beats none," not the adaptive mechanism specifically). **Yes, but
modest and product-dependent.**

**RQ2 (real data) — does it hold up outside synthetic data?** Answered by Script 04,
after the carrying-rate fix: +7.15–7.22% pooled adaptive-vs-fixed, 95% CI excluding
zero, holding across 5 of 6 H&M product categories (Sport, n=39, does not survive a
higher-replicate bootstrap). **Yes — the strongest, most defensible evidence in the
project.**

**RQ3 — is demand volatility a usable stockout-risk signal?** Answered by Script 05,
using the ground-truth regime label: every product shows materially higher stockout
risk in its true volatile regime (+15% to +135% relative), and the adaptive policy
reduces that risk further (−3% to −9%). **Yes** — though this validation is descriptive
(large-n simulation), not a formal significance test, which is an explicitly flagged
limitation and a natural next step (see below).

---

## 4. Headline Validated Results (Summary)

| Result | Value | Source |
|---|---|---|
| RQ1: order quantity, calm → volatile (representative product) | 285 → 296 units | RQ1 utility diagram |
| RQ2 synthetic, adaptive vs. fixed (Comparison B) | +0.99% / +4.89% / +0.41% | `scripts/03_comparison_A_and_B_gated_hybrid.py` |
| RQ2 synthetic, adaptive vs. standard MSE (Comparison C) | +91.9% to +93.7% | `scripts/03_comparison_A_and_B_gated_hybrid.py` |
| RQ2 real data, adaptive vs. fixed (post-fix) | +7.15% to +7.22%, 95% CI excludes zero | `scripts/04_real_data_validation.py` |
| RQ2 real data, adaptive vs. fixed (pre-fix, for contrast) | −3.57%, 95% CI [−5.09%, −2.18%] | superseded — carrying-rate bug |
| RQ3: stockout risk increase, calm → volatile | +15% to +135% relative | `scripts/05_rq3_stockout_risk.py` |
| RQ3: risk reduction from adaptive policy | −3% to −9% | `scripts/05_rq3_stockout_risk.py` |

---

## 5. Known Limitations (carried forward as future work)

- The MSE comparison overstates the adaptive-specific effect — most of the +81–94% gap
  is "any cost-asymmetric target beats none," not the adaptive mechanism specifically.
- Real-data £ costs use assumed, not H&M's actual, unit economics (3× markup, 25%
  carrying rate applied uniformly).
- The Sport category result does not survive a higher-replicate bootstrap.
- The corrected carrying rate and the adaptive z-span are principled but **frozen**
  constants — not yet sensitivity-tested against reasonable variation. This is the
  clearest next step before calling the headline result robust, not just correct.
- RQ3's validation is descriptive, not a formal significance test.

*Last updated alongside the pipeline code in this repository — see individual script
docstrings and `src/cost_model.py`, `src/gated_hybrid.py`, and `src/synthetic_data.py`
for the full technical detail behind each item above.*
