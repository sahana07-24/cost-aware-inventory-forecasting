"""
Cost-Aware Inventory Forecasting: interactive demo of an MSc Data Science dissertation
(University of Birmingham, 2026) by Sahana Sampath Saraswathy.

Run locally:  streamlit run app/streamlit_app.py
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from model_core import (  # noqa: E402
    ADAPTIVE_Z_SPAN, METHOD_LABELS, METHODS, PRODUCT_CONFIG, SBC_CV_THRESHOLD,
    CostSettings, run_scenario, tau_bins_for, tau_curve,
)
from bootstrap import bootstrap_ratio_ci_clustered  # noqa: E402
from synthetic_data import VOLATILE_BLOCK_BOUNDS, N_WEEKS  # noqa: E402

RESULTS = os.path.join(ROOT, "results")

# Colours: validated categorical palette, fixed per method (never by rank)
COLORS = {"adaptive": "#2a78d6", "fixed": "#eb6834", "naive": "#1baf7a", "mse": "#e87ba4"}
DEMAND_COLOR = "#8a8984"
SHADE = "rgba(235,104,52,0.10)"

PRODUCT_NAMES = {
    "product_A": "Product A: Steady grower",
    "product_B": "Product B: Declining item",
    "product_C": "Product C: Fast grower",
}

st.set_page_config(page_title="Cost-Aware Inventory Forecasting", page_icon="📦", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  .small-note {color: #6b6a65; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------
# Cached computations
# --------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_scenario(product, seed, unit_cost, price, carrying, surcharge, window):
    costs = CostSettings(unit_cost, price, carrying, surcharge)
    r = run_scenario(product, seed, costs, evaluate_on=window)
    return dict(df=r["df"], test=r["test"], preds=r["preds"], mask=r["mask"],
                summary=r["summary"], n_eval_weeks=r["n_eval_weeks"])


@st.cache_data(show_spinner=False)
def cached_batch(product, n_seeds, unit_cost, price, carrying, surcharge):
    costs = CostSettings(unit_cost, price, carrying, surcharge)
    rows = []
    for s in range(n_seeds):
        r = run_scenario(product, s, costs, evaluate_on="volatile")
        if r["n_eval_weeks"] < 4:
            continue
        sm = r["summary"]
        rows.append(dict(seed=s, **{f"cost_{m}": sm[m]["cost_per_week"] for m in METHODS}))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_default_scenarios():
    return pd.read_csv(os.path.join(HERE, "data", "default_scenarios.csv"))


@st.cache_data(show_spinner=False)
def load_real_folds():
    return pd.read_csv(os.path.join(RESULTS, "04_real_data_folds_N1500.csv"))


@st.cache_data(show_spinner=False)
def real_pooled_cis():
    fdf = load_real_folds()
    out = {}
    for den in ["cost_fixed", "cost_naive", "cost_mse"]:
        out[den] = bootstrap_ratio_ci_clustered(fdf, "cost_adaptive", den)
    return out


def fmt_money(x):
    return f"£{x:,.0f}" if abs(x) >= 100 else f"£{x:,.2f}"


def base_layout(fig, height=420, ytitle=None, xtitle=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      hovermode="x unified", yaxis_title=ytitle, xaxis_title=xtitle)
    fig.update_xaxes(showgrid=False)
    return fig


# --------------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------------
st.title("Cost-aware inventory forecasting")
st.markdown(
    "Running out of stock costs a retailer far more than holding one extra unit. This app shows a "
    "forecasting method that **orders for the cost, not for the average**, and raises its safety "
    "margin only when demand turns volatile. "
    "It is the interactive version of my MSc dissertation, and it runs the dissertation's own "
    "validated code."
)

tab_sim, tab_results, tab_how = st.tabs(["🧪 Live simulator", "📊 Validated results", "⚙️ How it works"])

# --------------------------------------------------------------------------------------------
# Tab 1: simulator
# --------------------------------------------------------------------------------------------
with tab_sim:
    c1, c2, c3 = st.columns([1.3, 1, 1.2])
    with c1:
        product = st.selectbox("Product", list(PRODUCT_CONFIG), index=1, format_func=PRODUCT_NAMES.get)
    with c2:
        if "seed" not in st.session_state:
            st.session_state.seed = 0
        seed = st.number_input("Scenario number (random seed)", 0, 999, key="seed", step=1)
    with c3:
        window_label = st.radio("Score the methods on", ["Volatile weeks only", "All forecast weeks"],
                                horizontal=False, help="The dissertation scores synthetic products on the "
                                "volatile weeks of the forecast period, where the methods differ.")
        window = "volatile" if window_label.startswith("Volatile") else "all"

    with st.expander("Business costs (change these and watch the orders move)", expanded=False):
        k1, k2, k3, k4 = st.columns(4)
        unit_cost = k1.number_input("Unit cost (£)", 1.0, 1000.0, 50.0, 5.0)
        price = k2.number_input("Retail price (£)", 1.0, 3000.0, 150.0, 10.0)
        carrying = k3.slider("Holding cost (% of unit cost per year)", 5, 60, 25) / 100
        surcharge = k4.slider("Expedited restock surcharge (% of unit cost)", 0, 50, 12) / 100
        if price <= unit_cost:
            st.warning("Retail price must be above unit cost for a stockout to cost anything.")

    costs = CostSettings(unit_cost, price, carrying, surcharge)
    with st.spinner("Training the LightGBM residual models..."):
        r = cached_scenario(product, int(seed), unit_cost, price, carrying, surcharge, window)

    sm = r["summary"]

    # Typical result across many scenarios (precomputed at default costs)
    defaults = CostSettings()
    using_defaults = (unit_cost, price, carrying, surcharge) == (
        defaults.unit_cost, defaults.retail_price, defaults.carrying_rate_per_year, defaults.restock_surcharge_rate)
    if using_defaults and window == "volatile":
        d = load_default_scenarios()
        d = d[d["product"] == product]
        saving = 100 * (d["cost_fixed"] - d["cost_adaptive"]) / d["cost_fixed"]
        pooled = 100 * (1 - d["cost_adaptive"].sum() / d["cost_fixed"].sum())
        vs_naive = 100 * (1 - d["cost_adaptive"].sum() / d["cost_naive"].sum())
        st.markdown(f"##### Typical result for {PRODUCT_NAMES[product].split(': ')[1].lower()}, across {len(d)} scenarios")
        t1, t2, t3 = st.columns(3)
        t1.metric("Adaptive total saving vs fixed", f"{pooled:+.1f}%")
        t2.metric("Adaptive total saving vs naive", f"{vs_naive:+.1f}%")
        t3.metric("Scenarios where adaptive is cheaper than fixed", f"{(saving > 0).mean():.0%}")
        st.caption("Adaptation usually changes little, then saves a lot in the scenarios with the worst volatility "
                   "spikes. Pick any scenario number below to explore one in detail.")
        st.markdown(f"##### Scenario {int(seed)} in detail")

    st.caption(
        f"Holding one unsold unit costs **£{costs.c_o:.2f}/week**; a lost sale costs **£{costs.c_u:.2f}**, "
        f"so the cost-optimal service level is **q\\* = {costs.q_star:.2%}**. "
        f"Scored on {r['n_eval_weeks']} weeks."
    )

    # KPI cards
    m_cols = st.columns(4)
    fixed_cost = sm["fixed"]["cost_per_week"]
    for col, m in zip(m_cols, ["adaptive", "fixed", "naive", "mse"]):
        cost = sm[m]["cost_per_week"]
        delta = None
        if m != "fixed" and fixed_cost > 0:
            delta = f"{100 * (cost - fixed_cost) / fixed_cost:+.1f}% vs fixed"
        col.metric(METHOD_LABELS[m].split(" (")[0], f"{fmt_money(cost)} / week", delta, delta_color="inverse")

    # Main chart
    df, test, preds = r["df"], r["test"], r["preds"]
    show_naive = st.checkbox("Also show naive and MSE forecasts", value=False)
    fig = go.Figure()
    for lo, hi in VOLATILE_BLOCK_BOUNDS:
        fig.add_vrect(x0=int(N_WEEKS * lo), x1=int(N_WEEKS * hi), fillcolor=SHADE, line_width=0, layer="below")
    split = int(len(df) * 0.7)
    fig.add_vline(x=split, line_dash="dot", line_color=DEMAND_COLOR)
    fig.add_annotation(x=split, y=1.0, yref="paper", text="  forecast starts", showarrow=False,
                       xanchor="left", font=dict(size=11, color=DEMAND_COLOR))
    fig.add_annotation(x=int(N_WEEKS * 0.25), y=1.0, yref="paper", text="volatile period", showarrow=False,
                       font=dict(size=11, color="#b2502a"))
    fig.add_trace(go.Scatter(x=df["week"], y=df["demand"], name="Actual demand", mode="lines",
                             line=dict(color=DEMAND_COLOR, width=1.5)))
    order = ["fixed", "adaptive"] + (["naive", "mse"] if show_naive else [])
    for m in order:
        fig.add_trace(go.Scatter(x=test["week"], y=preds[m], name=METHOD_LABELS[m], mode="lines",
                                 line=dict(color=COLORS[m], width=2)))
    base_layout(fig, height=440, ytitle="Units per week", xtitle="Week")
    st.plotly_chart(fig, width="stretch")

    # Breakdown table
    rows = []
    for m in METHODS:
        s = sm[m]
        rows.append({"Method": METHOD_LABELS[m], "Cost per week": fmt_money(s["cost_per_week"]),
                     "Stockout weeks": f"{s['stockout_rate']:.0%}",
                     "Avg order (units)": f"{s['avg_order']:,.0f}",
                     "Units left over": f"{s['leftover_units']:,.0f}",
                     "Units short": f"{s['shortfall_units']:,.0f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        '<p class="small-note">One scenario can be lucky or unlucky. Run many to see the typical result. '
        "The standard MSE forecast aims at average demand, so it is short roughly half the time, and "
        "every shortfall costs the full lost margin.</p>", unsafe_allow_html=True)

    with st.expander("Run 30 scenarios with your own cost settings", expanded=False):
        if st.button("Run 30 scenarios (about 10 seconds)"):
            with st.spinner("Training 30 x 4 models..."):
                b = cached_batch(product, 30, unit_cost, price, carrying, surcharge)
            b["saving_vs_fixed"] = 100 * (b["cost_fixed"] - b["cost_adaptive"]) / b["cost_fixed"]
            pooled = 100 * (1 - b["cost_adaptive"].sum() / b["cost_fixed"].sum())
            wins = (b["saving_vs_fixed"] > 0).mean()
            a, bcol, ccol = st.columns(3)
            a.metric("Scenarios", len(b))
            bcol.metric("Adaptive cheaper than fixed in", f"{wins:.0%} of scenarios")
            ccol.metric("Total cost saving vs fixed", f"{pooled:+.1f}%")
            hist = go.Figure(go.Histogram(x=b["saving_vs_fixed"], nbinsx=20, marker_color=COLORS["adaptive"]))
            hist.add_vline(x=0, line_color=DEMAND_COLOR, line_dash="dot")
            base_layout(hist, height=280, ytitle="Scenarios", xtitle="Adaptive saving vs fixed (%), per scenario")
            hist.update_layout(hovermode="closest", showlegend=False)
            st.plotly_chart(hist, width="stretch")
            st.caption("The dissertation's validated figures use 600 scenarios per product (see Validated results).")

    with st.expander("How the adaptive target moves with volatility", expanded=False):
        cvs = np.linspace(0, 1.5, 151)
        z_star = norm.ppf(costs.q_star)
        z_curve = norm.ppf(np.clip(tau_curve(costs.q_star, cvs), 1e-12, 1 - 1e-12))
        zf = go.Figure()
        zf.add_trace(go.Scatter(x=cvs, y=np.full_like(cvs, z_star), name="Fixed target", mode="lines",
                                line=dict(color=COLORS["fixed"], width=2)))
        zf.add_trace(go.Scatter(x=cvs, y=z_curve, name="Adaptive curve", mode="lines",
                                line=dict(color=COLORS["adaptive"], width=1.5, dash="dot")))
        bins_ = tau_bins_for(costs.q_star)
        z_step = [norm.ppf(bins_["calm"] if cv < 0.35 else bins_["moderate"] if cv < SBC_CV_THRESHOLD
                           else bins_["volatile"]) for cv in cvs]
        zf.add_trace(go.Scatter(x=cvs, y=z_step, name="Used by the model (3 regimes)", mode="lines",
                                line=dict(color=COLORS["adaptive"], width=2.5, shape="hv")))
        zf.add_vline(x=SBC_CV_THRESHOLD, line_dash="dot", line_color=DEMAND_COLOR)
        base_layout(zf, height=300, ytitle="Safety factor z (standard deviations of cover)",
                    xtitle="Recent demand volatility (rolling coefficient of variation)")
        st.plotly_chart(zf, width="stretch")
        bins = tau_bins_for(costs.q_star)
        st.caption(
            f"Target service level: calm {bins['calm']:.3%}, moderate {bins['moderate']:.3%}, volatile "
            f"{bins['volatile']:.3%}. Near 100% the percentages barely change, but the safety factor rises "
            f"by up to {ADAPTIVE_Z_SPAN:.2f} standard deviations, which adds real units to the order. "
            "That is why the adaptation is done in z-space."
        )

# --------------------------------------------------------------------------------------------
# Tab 2: validated results
# --------------------------------------------------------------------------------------------
with tab_results:
    fdf = load_real_folds()
    with st.spinner("Cluster-bootstrapping 1,197 articles..."):
        cis = real_pooled_cis()

    st.subheader("Real H&M data: 1,197 articles, 3,591 walk-forward folds")
    a, b, c = st.columns(3)
    for col, key, label in [(a, "cost_fixed", "vs fixed target (the thesis)"),
                            (b, "cost_naive", "vs naive safety stock"),
                            (c, "cost_mse", "vs standard MSE forecast")]:
        p, lo, hi, _ = cis[key]
        col.metric(f"Adaptive saving {label}", f"{p:+.1f}%")
        col.caption(f"95% CI {lo:+.1f}% to {hi:+.1f}%")
    st.caption("Pooled £-weighted saving: total cost of the comparison method vs total cost of the adaptive "
               "method across all folds. Confidence intervals resample whole articles (cluster bootstrap), "
               "because an article's folds are not independent.")

    st.markdown("#### Where the saving comes from")
    fdf = fdf.copy()
    fdf["decile"] = pd.qcut(fdf["cost_fixed"].rank(method="first"), 10, labels=False) + 1
    g = fdf.groupby("decile").agg(fixed=("cost_fixed", "sum"), adapt=("cost_adaptive", "sum"),
                                  win=("adaptive_vs_fixed_pct", lambda s: (s > 0).mean()))
    g["share"] = 100 * g["fixed"] / g["fixed"].sum()
    g["saving"] = 100 * (1 - g["adapt"] / g["fixed"])
    dec_fig = go.Figure(go.Bar(
        x=[f"D{i}" for i in g.index], y=g["saving"],
        marker_color=[COLORS["adaptive"] if v > 0 else "#b9b8b1" for v in g["saving"]],
        text=[f"{s:.0f}% of £" if s >= 1 else ("<0.1% of £" if s < 0.1 else f"{s:.1f}% of £") for s in g["share"]],
        textposition="outside",
        customdata=np.stack([g["share"], 100 * g["win"]], axis=1),
        hovertemplate="Decile %{x}<br>Saving vs fixed: %{y:.1f}%<br>Share of total cost: %{customdata[0]:.1f}%"
                      "<br>Adaptive wins: %{customdata[1]:.0f}% of folds<extra></extra>"))
    dec_fig.add_hline(y=0, line_color=DEMAND_COLOR)
    base_layout(dec_fig, height=360, ytitle="Adaptive saving vs fixed (%)",
                xtitle="Folds grouped by cost, D1 = cheapest 10%, D10 = most expensive 10%")
    dec_fig.update_layout(hovermode="closest", showlegend=False)
    dec_fig.update_yaxes(range=[min(g["saving"].min() * 1.25, -10), max(g["saving"].max() * 1.6, 10)])
    st.plotly_chart(dec_fig, width="stretch")
    top = g.loc[10]
    st.markdown(
        f"The most expensive 10% of folds carry **{top['share']:.0f}% of all inventory cost**. There, the adaptive "
        f"target wins **{top['win']:.0%}** of folds and saves **{top['saving']:.1f}%**. On cheap, low-volume "
        "articles it slightly over-stocks, which costs almost nothing in total. So the saving is real in pounds, "
        "but an equal-weighted average across folds is negative (adaptive wins 37% of all folds). "
        "**The practical rule: use adaptive targets for high-value, high-volume products.**"
    )
    st.caption("H&M publishes prices on a scaled unit, so absolute £ costs per fold are tiny; percentages "
               "are unaffected because every method is costed on the same scale.")

    st.divider()
    st.subheader("Synthetic products: 400–600 scenarios each")
    syn = pd.read_csv(os.path.join(RESULTS, "03_comparison_A_and_B_summary.csv"))
    table = pd.DataFrame({
        "Product": [PRODUCT_NAMES[p] for p in syn["product"]],
        "vs naive": [f"{m:+.1f}%  {ci}" for m, ci in zip(syn["comparison_A_mean"], syn["comparison_A_ci"])],
        "vs fixed (thesis)": [f"{m:+.1f}%  {ci}" + ("" if s else "  (not significant)")
                              for m, ci, s in zip(syn["comparison_B_mean"], syn["comparison_B_ci"],
                                                  syn["comparison_B_significant"])],
        "vs standard MSE": [f"{m:+.1f}%  {ci}" for m, ci in zip(syn["comparison_C_mean"], syn["comparison_C_ci"])],
    })
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Mean cost saving of the adaptive gated hybrid with 95% bootstrap CI. On synthetic data the gain "
               "over a fixed target is small and only significant for the most volatile product (B). Most of the "
               "gap to the MSE forecast comes from being cost-aware at all, not from adapting.")

    st.markdown("#### Does volatility predict stockout risk? (RQ3)")
    rq3 = pd.read_csv(os.path.join(RESULTS, "05_rq3_stockout_risk_summary.csv"))
    rq3["regime"] = np.where(rq3.groupby("product")["true_cv_regime"].rank() == 1, "Calm", "Volatile")
    rq3["x"] = [f"{p.split('_')[1].upper()} · {g_}" for p, g_ in zip(rq3["product"], rq3["regime"])]
    rq_fig = go.Figure()
    for col_rate, lo, hi, name, key in [("fixed_rate", "fixed_lo", "fixed_hi", "Fixed target", "fixed"),
                                        ("adapt_rate", "adapt_lo", "adapt_hi", "Adaptive target", "adaptive")]:
        rq_fig.add_trace(go.Bar(
            x=rq3["x"], y=100 * rq3[col_rate], name=name, marker_color=COLORS[key],
            error_y=dict(type="data", symmetric=False, array=100 * (rq3[hi] - rq3[col_rate]),
                         arrayminus=100 * (rq3[col_rate] - rq3[lo]), color=DEMAND_COLOR, thickness=1.2)))
    base_layout(rq_fig, height=340, ytitle="Weeks with a stockout (%)", xtitle="Product · true demand regime")
    rq_fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06, hovermode="x")
    st.plotly_chart(rq_fig, width="stretch")
    st.caption("400 scenarios per product, 95% Wilson intervals. Stockouts rise in volatile periods; the adaptive "
               "target lowers them there and leaves calm periods unchanged, as designed.")

    st.markdown("#### Why a hybrid at all?")
    lg = pd.read_csv(os.path.join(RESULTS, "01_lgbm_vs_naive_validated.csv"))
    st.markdown(
        "A pure LightGBM model that predicts demand directly **lost to a simple rolling-mean rule by "
        f"{-lg['mean_pct'].max():.0f}% to {-lg['mean_pct'].min():.0f}%** (800 scenarios per product). A model "
        "trained once cannot keep up with a rule that updates every week. So LightGBM was redesigned to correct "
        "the simple rule's error instead of replacing it."
    )

# --------------------------------------------------------------------------------------------
# Tab 3: how it works
# --------------------------------------------------------------------------------------------
with tab_how:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("The method in five steps")
        st.markdown("""
1. **Start from a simple, live rule.** Rolling 12-week mean plus a safety stock. It updates every week.
2. **Let LightGBM correct it.** A quantile LightGBM model predicts the *error* of the simple rule
   from seasonality, lags, promotions and recent trend.
3. **Gate the correction.** When recent trend is strong, a sigmoid gate shrinks the ML correction toward zero,
   because tests showed the correction hurts under strong positive trend.
4. **Order for the cost.** The newsvendor model turns costs into a target service level:
   `q* = Cu / (Cu + Co)`. With a £106 lost sale and a £0.24/week holding cost, q\\* ≈ 99.8%.
5. **Adapt to volatility.** When recent demand volatility (coefficient of variation) rises, the target moves
   up smoothly in z-space, adding safety stock only when it is needed.
""")
        st.subheader("What went wrong, and how it was fixed")
        st.markdown("""
- **Volatile weeks only appeared in the test period.** Tree models can't extrapolate, so the model had never
  seen volatility. Fixed by cycling volatility through the series three times.
- **Pure ML lost to a simple rule** by 68–242%. This led to the hybrid design.
- **The ML correction hurt under strong upward trend.** This led to the trend gate.
- **A holding-cost unit bug:** a 25% *annual* rate was applied *weekly*, overstating holding cost by 52×.
  Fixing it moved q\\* from 89.5% to 99.8% and flipped the real-data result from −3.6% to +7.2%.
""")
    with right:
        st.subheader("Validation discipline")
        st.markdown("""
- Stopping rules and sample sizes fixed **before** looking at results
- Forced minimum of 400 scenarios, after an early stop once gave a false null
- Walk-forward folds on real data; every feature uses only past weeks
- **Cluster bootstrap** over articles, since folds of one article aren't independent
- Pooled £ and per-fold results both reported
""")
        st.subheader("Limitations")
        st.markdown("""
- One retailer; unit costs assumed (3× markup), not H&M's real figures
- Real-data gain is concentrated in high-cost articles
- Carrying rate and adaptive span not yet sensitivity-tested
- RQ3 is descriptive, not a formal significance test
""")
    st.divider()
    st.markdown(
        "**Sahana Sampath Saraswathy** · MSc Data Science, University of Birmingham · "
        "[LinkedIn](https://www.linkedin.com/in/sahana-ss-a357863a3) · "
        "[GitHub](https://github.com/sahana07-24) · "
        "Data: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)"
    )
