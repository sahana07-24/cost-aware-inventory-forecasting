"""
Newsvendor cost model -- shared by every experiment in this project, synthetic
and real-data alike.

THE SCENARIO (see Project_Explainer.docx, Section 1): an online fashion
retailer selling direct to consumers, one distribution centre, one
representative garment at a 3x markup (£50 unit cost / £150 retail price).
Holding cost is 25% of unit cost per year. Stockout cost is the lost margin
(£100) plus a £6 expedited-restock surcharge (£106/unit total) -- NOT just
the margin alone.

HARMONIZATION NOTE (flagged, not silent): the real-data script this project
ran first (phase5_hm_validation.py) computed c_u as margin-only
(price - unit_cost), omitting the £6-equivalent restock surcharge that the
documented scenario includes. That gave q*=8/9=0.8889 for every H&M article
(since the ratio only depends on the 3x markup and 25% carrying rate, both
fixed). This module applies the SAME proportional surcharge convention used
for the synthetic products (a fixed fraction of unit cost) to real articles
too, so both pipelines are computing q* the same way.

CORRECTION (this update -- was a confirmed bug, now fixed): CARRYING_RATE is
documented above, and in every earlier version of this file, as a rate PER
YEAR -- but every cost function in this project operates on WEEKLY periods
(one row per product-week), and the rate was being applied directly with no
annual-to-weekly conversion. That silently treated 25%-per-year as if it
were 25%-per-week, a ~52x overstatement of the true weekly holding cost.
Fixed here by dividing by 52. This is a straightforward unit-conversion fix,
not a modeling judgment call -- the annual figure and the weekly evaluation
grain were never in question, only their reconciliation.

CONSEQUENCE, AND THE SECOND FIX IT REQUIRED: correcting the rate moves
Q_STAR from ~0.8945 to ~0.9977 (holding a unit got ~52x cheaper relative to
stocking out, so the newsvendor solution pushes the optimal fixed quantile
much closer to 1). At that level, the *original* adaptive scheme -- which
interpolated the target quantile in PROBABILITY space, up to a fixed
ALPHA_CEILING=0.95 -- had nowhere left to go: 0.95 < 0.9977, so every regime
collapsed to the same quantile (see the deprecated function below). This
isn't a case for picking a new probability ceiling by hand -- any constant
close enough to 1 to leave room is arbitrary, and the same problem
resurfaces for any q* close enough to 1. The structural fix is to stop
interpolating in probability space at all: near q=1, probability space is
saturated (bounded above by 1), but the corresponding order quantity
(via the inverse-normal z-score) is not -- equal steps in z-space still move
real stocking decisions even when they barely move the quantile number.
ADAPTIVE_Z_SPAN below is the width, in z-units, that the ORIGINAL design
swept between q*=0.8945 (calm) and ALPHA_CEILING=0.95 (fully volatile),
recovered once from those two original, documented numbers and then applied
unchanged at the corrected q* -- so the fix preserves the original design's
intended amount of adaptivity, rather than being re-tuned against the
corrected results to produce any particular outcome.
"""
import numpy as np
from scipy.stats import norm

# ---- Synthetic-product scenario (the dissertation's stated illustrative economics) ----
UNIT_COST = 50.0
RETAIL_PRICE = 150.0
CARRYING_RATE = 0.25                 # holding cost, fraction of unit cost PER YEAR
RESTOCK_SURCHARGE_RATE = 0.12        # expedited-restock surcharge, fraction of unit cost (-> £6 on a £50 unit)
WEEKS_PER_YEAR = 52

C_O = (CARRYING_RATE / WEEKS_PER_YEAR) * UNIT_COST   # weekly holding cost -- was CARRYING_RATE*UNIT_COST (bug)
C_U = (RETAIL_PRICE - UNIT_COST) + RESTOCK_SURCHARGE_RATE * UNIT_COST
Q_STAR = C_U / (C_U + C_O)

# ---- Adaptive-quantile scheme: shared thresholds/shape across every dataset ----
CALM_CV_THRESHOLD = 0.35
SBC_CV_THRESHOLD = 0.70              # rolling CV level treated as "volatile" (SBC = "shift to bigger cover")

# Original (pre-correction) design constants -- kept as named constants, not
# inlined, so the provenance of ADAPTIVE_Z_SPAN stays auditable.
_Q_STAR_ORIGINAL_DESIGN = 0.8945147679324894      # C_U/(C_U+C_O) under the pre-fix (buggy) C_O
ALPHA_CEILING = 0.95                              # original probability ceiling; kept for the deprecated fn below
ADAPTIVE_Z_SPAN = norm.ppf(ALPHA_CEILING) - norm.ppf(_Q_STAR_ORIGINAL_DESIGN)  # ~0.3940, frozen from the original design


def cost_params_from_price(price: float, markup_ratio: float = 3.0,
                            carrying_rate: float = CARRYING_RATE,
                            restock_surcharge_rate: float = RESTOCK_SURCHARGE_RATE,
                            weeks_per_year: int = WEEKS_PER_YEAR) -> tuple:
    """Derive (c_o, c_u, q_star) from a real observed price, using the same
    3x-markup / 25%-per-year-carrying / proportional-restock-surcharge
    convention as the synthetic scenario, with the annual carrying rate
    converted to a weekly cost. Used for real-data articles, where we only
    have sale PRICE (not the retailer's confidential unit cost) -- see
    Project_Explainer.docx Section 1 and phase5_hm_validation.py's own note
    on this same assumption.
    """
    unit_cost = price / markup_ratio
    c_o = (carrying_rate / weeks_per_year) * unit_cost
    c_u = (price - unit_cost) + restock_surcharge_rate * unit_cost
    q_star = c_u / (c_u + c_o)
    return c_o, c_u, q_star


def assign_tau_adaptive_probability_space_DEPRECATED(cv: float, q_star: float,
                         cv_threshold: float = SBC_CV_THRESHOLD,
                         alpha_ceiling: float = ALPHA_CEILING) -> float:
    """SUPERSEDED -- kept for the audit trail and for anyone reproducing the
    pre-correction results, not used anywhere in this pipeline anymore.
    Interpolated the target quantile directly in probability space, up to a
    fixed ceiling. Degenerates to a no-op (returns q_star for every regime)
    whenever q_star >= alpha_ceiling, which is exactly what the corrected
    Q_STAR (~0.9977) does against the original ALPHA_CEILING (0.95). See the
    module docstring and assign_tau_adaptive() below for the fix.
    """
    span = alpha_ceiling - q_star
    if span <= 0:
        return q_star
    return q_star + span / (1 + np.exp(-8.0 * (cv - cv_threshold)))


def assign_tau_adaptive(cv: float, q_star: float = Q_STAR,
                         cv_threshold: float = SBC_CV_THRESHOLD,
                         z_span: float = ADAPTIVE_Z_SPAN) -> float:
    """Smooth sigmoid mapping from a CV level to a target quantile, done in
    z-space (order-quantity units) rather than probability space: sits at
    q_star when calm, and moves smoothly toward z(q_star) + z_span as CV
    passes cv_threshold, before converting back to a probability. Same
    sigmoid shape and 8.0 steepness constant as the original design; only
    the space the interpolation happens in has changed, which is what keeps
    the scheme from degenerating as q_star approaches 1. z_span is fixed at
    the width the original (pre-correction) design used -- see module
    docstring.
    """
    z_star = norm.ppf(q_star)
    z = z_star + z_span / (1 + np.exp(-8.0 * (cv - cv_threshold)))
    return float(norm.cdf(z))


def tau_bins_for(q_star: float = Q_STAR) -> dict:
    """The three quantile targets the adaptive scheme switches between,
    keyed by regime name. bins['calm'] always equals q_star exactly -- this
    is why 'fixed' and 'adaptive' predictions are identical whenever a test
    row falls in the calm bin (same tau, same trained model)."""
    return {
        "calm": q_star,
        "moderate": assign_tau_adaptive(0.5, q_star),
        "volatile": assign_tau_adaptive(1.0, q_star),
    }


def cv_to_bin(cv: float, calm_threshold: float = CALM_CV_THRESHOLD,
              volatile_threshold: float = SBC_CV_THRESHOLD) -> str:
    if cv < calm_threshold:
        return "calm"
    if cv < volatile_threshold:
        return "moderate"
    return "volatile"


def business_cost(y_true, y_pred, c_o: float = C_O, c_u: float = C_U) -> float:
    """Average per-period newsvendor cost: overage (leftover stock) costed
    at c_o, underage (stockout) costed at c_u."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    o = np.clip(y_pred - y_true, 0, None)
    u = np.clip(y_true - y_pred, 0, None)
    return float((c_o * o.sum() + c_u * u.sum()) / len(y_true))
