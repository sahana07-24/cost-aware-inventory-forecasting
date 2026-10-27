"""
RQ3 diagram: realized stockout rate by TRUE demand-CV regime (calm vs.
volatile), fixed vs. adaptive policy, per synthetic product.

Source data: results/05_rq3_stockout_risk_summary.csv (computed by
scripts/05_rq3_stockout_risk.py, 400 seeds/product, Wilson 95% CIs).

This is deliberately NOT the RQ1 cost-curve diagram (ppt/charts/utility_diagram.py,
kept as a presentation asset -- see 05's docstring). That one answers "where does
adaptation move you on the cost curve." RQ3's own question is a risk-RATE question
(does demand-CV regime predict stockout risk, and does acting on it help?), so its
diagram is a grouped bar chart of stockout rate, not a cost curve.

Run: python 07_rq3_stockout_risk_chart.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

HERE = os.path.dirname(__file__)
IN_PATH = os.path.join(HERE, "..", "results", "05_rq3_stockout_risk_summary.csv")
OUT_DIR = os.path.join(HERE, "..", "results", "figures")
OUT_PATH = os.path.join(OUT_DIR, "07_rq3_stockout_risk_by_regime.png")

df = pd.read_csv(IN_PATH)

PRODUCT_LABELS = {
    "product_A": "Product A\n(Steady Grower)",
    "product_B": "Product B\n(Declining Item)",
    "product_C": "Product C\n(Fast Grower)",
}

# Categorical pair -- fixed (muted slate) vs adaptive (teal), colorblind-checked pairing
COLOR_FIXED = "#5B6B84"
COLOR_ADAPT = "#1B8A8A"

fig, axes = plt.subplots(1, 3, figsize=(11, 5.0), sharey=True)

for ax, product in zip(axes, ["product_A", "product_B", "product_C"]):
    sub = df[df["product"] == product].sort_values("true_cv_regime").reset_index(drop=True)
    regimes = ["Calm", "Volatile"]
    x = np.arange(len(regimes))
    width = 0.32

    fixed_rate = sub["fixed_rate"].values * 100
    fixed_err = np.vstack([
        (sub["fixed_rate"].values - sub["fixed_lo"].values) * 100,
        (sub["fixed_hi"].values - sub["fixed_rate"].values) * 100,
    ])
    adapt_rate = sub["adapt_rate"].values * 100
    adapt_err = np.vstack([
        (sub["adapt_rate"].values - sub["adapt_lo"].values) * 100,
        (sub["adapt_hi"].values - sub["adapt_rate"].values) * 100,
    ])

    b1 = ax.bar(x - width / 2, fixed_rate, width, yerr=fixed_err, capsize=3,
                color=COLOR_FIXED, label="Fixed q*", zorder=3,
                error_kw=dict(elinewidth=1, ecolor="#33261b"))
    b2 = ax.bar(x + width / 2, adapt_rate, width, yerr=adapt_err, capsize=3,
                color=COLOR_ADAPT, label="Adaptive q*", zorder=3,
                error_kw=dict(elinewidth=1, ecolor="#33261b"))

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color="#2a2a2a")

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_title(PRODUCT_LABELS[product], fontsize=10.5, fontweight="bold", pad=10)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#999")
    ax.spines["bottom"].set_color("#999")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.tick_params(axis="both", labelsize=9, colors="#444")

axes[0].set_ylabel("Realized stockout rate", fontsize=10)
axes[0].set_ylim(0, max(df["adapt_hi"].max(), df["fixed_hi"].max()) * 100 * 1.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.965), fontsize=10)

fig.suptitle("RQ3 — Does demand-CV regime predict stockout risk, and does adapting on it help?",
             fontsize=12.5, fontweight="bold", y=1.10)
fig.text(0.5, 1.015,
          "Stockout rate by TRUE demand-volatility regime (400 seeds/product, error bars = Wilson 95% CI)",
          ha="center", fontsize=9.5, color="#555")

plt.tight_layout(rect=[0, 0, 1, 0.86])

os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {OUT_PATH}")

# Print the "answer" numbers this chart demonstrates
print("\nCalm -> volatile relative jump in risk under FIXED policy, and adaptive's reduction of that:")
for product in ["product_A", "product_B", "product_C"]:
    sub = df[df["product"] == product].sort_values("true_cv_regime")
    calm, vol = sub.iloc[0], sub.iloc[1]
    delta = (vol.fixed_rate / calm.fixed_rate - 1) * 100
    reduction = (1 - vol.adapt_rate / vol.fixed_rate) * 100
    print(f"  {product}: calm {calm.fixed_rate:.2%} -> volatile {vol.fixed_rate:.2%} ({delta:+.0f}% relative)  "
          f"| adaptive cuts volatile-regime risk by {reduction:.0f}%")
