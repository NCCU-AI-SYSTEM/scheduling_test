import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["PingFang HK", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("white")

parsers  = ["gpt-4.1-mini", "claude-haiku-4-5", "gemma-4-31b", "claude-sonnet-4-6"]
speed_ms = [223, 194, 450, 591]
colors   = ["#4f6ef7", "#74b9ff", "#f4a261", "#e17055"]

x = np.arange(len(parsers))
bars = ax.bar(x, speed_ms, color=colors, width=0.5, edgecolor="white", zorder=3)

for bar, v in zip(bars, speed_ms):
    ax.text(bar.get_x() + bar.get_width()/2, v + 8,
            f"{v} ms", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#1a1a2e")

ax.set_xticks(x)
ax.set_xticklabels(parsers, fontsize=11)
ax.set_ylim(0, 700)
ax.set_ylabel("ms / query", fontsize=11)
ax.yaxis.grid(True, color="#f0f0f0", zorder=0)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_color("#dee2e6")

ax.set_title("LLM Parser 推論速度（P3 各模型）", fontsize=13,
             fontweight="bold", color="#1a1a2e", pad=12)
ax.annotate("10 workers 並行 · 2,160q · Trend Micro endpoint",
            xy=(0.5, -0.14), xycoords="axes fraction",
            ha="center", fontsize=9.5, color="#888")

fig.tight_layout()
fig.savefig("docs/chart_parser_speed.png", dpi=150, bbox_inches="tight")
print("saved: docs/chart_parser_speed.png")
