"""畫 filter 實驗結果圖表，輸出 PNG。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 設定中文字型 ──────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = ["PingFang HK", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ── 資料 ──────────────────────────────────────────────────────────────────────
variants   = ["no_filter", "oracle", "p1 regex"]
r10        = [0.2431, 0.3722, 0.2222]
avg_pool   = [2795.0, 808.7, 850.3]
ms_per_q   = [499.0, 54.3, 47.7]

COLORS = ["#adb5bd", "#4f6ef7", "#f4a261"]

dim_labels = [
    "unit（系所）",
    "lmt_kind（通識細分）",
    "lang+unit+point",
    "lang+weekday+hour",
    "weekday",
    "lang（單獨）",
]
dim_pools  = [20, 35, 15, 151, 554, 1649]
dim_colors = ["#4f6ef7","#4f6ef7","#4f6ef7","#74b9ff","#f4a261","#e17055"]

# ── 畫布：2×2 + 底部橫條 ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 11))
fig.patch.set_facecolor("#f8f9fa")

gs = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.38,
                      top=0.91, bottom=0.06, left=0.07, right=0.97)

ax1 = fig.add_subplot(gs[0, 0])   # R@10
ax2 = fig.add_subplot(gs[0, 1])   # avg_pool
ax3 = fig.add_subplot(gs[1, 0])   # ms/query
ax4 = fig.add_subplot(gs[1, 1])   # oracle donut
ax5 = fig.add_subplot(gs[2, :])   # dim pool 橫條


def _style_ax(ax, title, desc=""):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("#dee2e6")
    ax.tick_params(colors="#555", labelsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1a1a2e", pad=8)
    if desc:
        ax.annotate(desc, xy=(0.5, 1.01), xycoords="axes fraction",
                    ha="center", fontsize=9, color="#888")


# ── (1) R@10 ──────────────────────────────────────────────────────────────────
x = np.arange(len(variants))
bars = ax1.bar(x, r10, color=COLORS, edgecolor="#fff", linewidth=0.8,
               width=0.55, zorder=3)
ax1.axhline(r10[0], color="#adb5bd", lw=1, ls="--", zorder=2)
ax1.set_xticks(x); ax1.set_xticklabels(variants, fontsize=10)
ax1.set_ylim(0, 0.55)
ax1.set_ylabel("R@10", fontsize=10)
ax1.yaxis.grid(True, color="#f0f0f0", zorder=0)
ax1.set_axisbelow(True)
for bar, v in zip(bars, r10):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.008,
             f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
             color="#1a1a2e")
# 差距標注
ax1.annotate("", xy=(1, r10[1]), xytext=(1, r10[0]),
             arrowprops=dict(arrowstyle="<->", color="#2d4ecf", lw=1.5))
ax1.text(1.28, (r10[0]+r10[1])/2, "+14.9pp", color="#2d4ecf", fontsize=9, va="center")
_style_ax(ax1, "R@10 比較（三種 filter 設定）", "↑ 越高越好  ·  no rerank  ·  2,160q")


# ── (2) avg_pool ──────────────────────────────────────────────────────────────
bars2 = ax2.bar(x, avg_pool, color=COLORS, edgecolor="#fff", linewidth=0.8,
                width=0.55, zorder=3)
ax2.axhline(2795, color="#dee2e6", lw=1, ls="--", zorder=2)
ax2.set_xticks(x); ax2.set_xticklabels(variants, fontsize=10)
ax2.set_ylim(0, 3300)
ax2.set_ylabel("avg pool（門）", fontsize=10)
ax2.yaxis.grid(True, color="#f0f0f0", zorder=0)
ax2.set_axisbelow(True)
ax2.text(0.98, 2795+40, "全庫 2,795", ha="right", fontsize=8.5, color="#aaa",
         transform=ax2.get_yaxis_transform())
for bar, v in zip(bars2, avg_pool):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 50,
             f"{v:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
             color="#1a1a2e")
_style_ax(ax2, "平均候選池大小（avg_pool）", "↓ 越小代表 filter 縮減越有效")


# ── (3) ms/query ──────────────────────────────────────────────────────────────
bars3 = ax3.bar(x, ms_per_q, color=COLORS, edgecolor="#fff", linewidth=0.8,
                width=0.55, zorder=3)
ax3.set_xticks(x); ax3.set_xticklabels(variants, fontsize=10)
ax3.set_ylim(0, 580)
ax3.set_ylabel("ms / query", fontsize=10)
ax3.yaxis.grid(True, color="#f0f0f0", zorder=0)
ax3.set_axisbelow(True)
for bar, v in zip(bars3, ms_per_q):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 8,
             f"{v:.0f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold",
             color="#1a1a2e")
# 9x 標注
ax3.annotate("", xy=(0, ms_per_q[0]), xytext=(1, ms_per_q[1]),
             arrowprops=dict(arrowstyle="<->", color="#2d4ecf", lw=1.5))
ax3.text(0.5, (ms_per_q[0]+ms_per_q[1])/2 + 20, "9× 加速", color="#2d4ecf",
         fontsize=9, ha="center")
_style_ax(ax3, "推論速度（ms / query）", "↓ 越低越快  ·  候選池縮小後矩陣乘積更快")


# ── (4) Oracle filter 品質 donut ──────────────────────────────────────────────
sizes  = [2157, 3]
clrs   = ["#4f6ef7", "#f4a261"]
wedges, texts = ax4.pie(sizes, colors=clrs, startangle=90,
                         wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
ax4.text(0, 0, "99.86%", ha="center", va="center",
         fontsize=16, fontweight="bold", color="#1a1a2e")
ax4.text(0, -0.22, "gold in pool", ha="center", va="center",
         fontsize=9, color="#666")
patches = [
    mpatches.Patch(color="#4f6ef7", label=f"gold 在候選池內（2,157）"),
    mpatches.Patch(color="#f4a261", label=f"被錯誤排除（3）"),
]
ax4.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.14),
           fontsize=9, frameon=False)
_style_ax(ax4, "Oracle Filter 召回品質")
ax4.set_facecolor("#f8f9fa")


# ── (5) 各維度 pool size 橫條 ─────────────────────────────────────────────────
y = np.arange(len(dim_labels))
h = 0.5
bars5 = ax5.barh(y, dim_pools, color=dim_colors, edgecolor="#fff",
                 linewidth=0.8, height=h, zorder=3)
ax5.axvline(2795, color="#dee2e6", lw=1.2, ls="--", zorder=2)
ax5.set_yticks(y); ax5.set_yticklabels(dim_labels, fontsize=10)
ax5.set_xlim(0, 3200)
ax5.set_xlabel("avg pool size（門）", fontsize=10)
ax5.xaxis.grid(True, color="#f0f0f0", zorder=0)
ax5.set_axisbelow(True)
ax5.text(2795+30, ax5.get_ylim()[1]*0.95, "全庫 2,795", fontsize=8.5, color="#aaa", va="top")
for bar, v in zip(bars5, dim_pools):
    pct = (1 - v/2795) * 100
    ax5.text(v + 40, bar.get_y() + bar.get_height()/2,
             f"{v} 門（縮 {pct:.0f}%）",
             va="center", fontsize=9.5, fontweight="bold", color="#1a1a2e")
_style_ax(ax5, "Oracle Filter 各維度候選池縮減效果", "↓ pool size 越小篩選力越強")


# ── 總標題 ────────────────────────────────────────────────────────────────────
fig.suptitle("Filter 實驗結果 — eval_conditions_v1 · 2,160q · no rerank",
             fontsize=14, fontweight="bold", color="#1a1a2e", y=0.975)

out = "docs/filter_experiment_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"saved: {out}")
