import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["PingFang HK", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("white")

parsers = ["P0\nRule-based", "P1\nRegex", "P3\ngpt-4.1-mini", "P3\nhaiku-4-5", "P3\nsonnet-4-6", "P3\ngemma-4-31b"]
exact   = [47.2, 63.2, 61.8, 66.1, 66.7, 68.3]
slot_f1 = [95.0, 96.4, 96.6, 97.2, 97.2, 97.4]
neg_f1  = [47.5, 78.6, 92.7, 96.8, 98.2, 92.8]

x = np.arange(len(parsers))
w = 0.25

b1 = ax.bar(x - w, exact,   width=w, color="#4f6ef7", label="ExactMatch", zorder=3, edgecolor="white")
b2 = ax.bar(x,     slot_f1, width=w, color="#74b9ff", label="SlotF1",     zorder=3, edgecolor="white")
b3 = ax.bar(x + w, neg_f1,  width=w, color="#f4a261", label="NegF1",      zorder=3, edgecolor="white")

# 每個 bar 標數值
for bars, vals in [(b1, exact), (b2, slot_f1), (b3, neg_f1)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=8, color="#1a1a2e")

ax.set_xticks(x)
ax.set_xticklabels(parsers, fontsize=11)
ax.set_ylim(0, 115)
ax.set_ylabel("%", fontsize=11)
ax.yaxis.grid(True, color="#f0f0f0", zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=11, frameon=False, loc="lower right")
for sp in ax.spines.values():
    sp.set_color("#dee2e6")

ax.set_title("Parser 各指標比較（ExactMatch / SlotF1 / NegF1）",
             fontsize=13, fontweight="bold", color="#1a1a2e", pad=12)
ax.annotate("eval_conditions_v1 · 2,160q  ·  NegF1 n=500（has_negation=True）",
            xy=(0.5, -0.14), xycoords="axes fraction",
            ha="center", fontsize=9.5, color="#888")

fig.tight_layout()
fig.savefig("docs/chart_parser_scores.png", dpi=150, bbox_inches="tight")
print("saved: docs/chart_parser_scores.png")
