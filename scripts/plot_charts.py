import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["PingFang HK", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ── 完整資料（只有 P3 LLM，含新 Round 2）────────────────────────────────────
# 依 ExactMatch 排序
models = [
    {"name": "gpt-4o",        "em": 58.5, "neg_f1": 95.2, "ms": 300},
    {"name": "gpt-4.1-mini",  "em": 61.8, "neg_f1": 92.7, "ms": 223},
    {"name": "haiku-4-5",     "em": 66.1, "neg_f1": 96.8, "ms": 194},
    {"name": "sonnet-4-6",    "em": 66.7, "neg_f1": 98.2, "ms": 591},
    {"name": "gpt-5.4-mini",  "em": 66.6, "neg_f1": 93.0, "ms": 613},
    {"name": "gemma-4-31b",   "em": 68.0, "neg_f1": 93.3, "ms": 548},
    {"name": "qwen3-235b",    "em": 68.7, "neg_f1": 98.2, "ms": 429},
]
# 按 EM 升序排（圖從左到右由低到高）
models.sort(key=lambda x: x["em"])

names   = [m["name"]   for m in models]
em      = [m["em"]     for m in models]
neg_f1  = [m["neg_f1"] for m in models]
ms      = [m["ms"]     for m in models]

x = np.arange(len(models))
w = 0.38

# ── 顏色：用 NegF1 深淺區分高低 ──────────────────────────────────────────────
EM_C   = "#4f6ef7"
NEG_C  = "#f4a261"

# ── 圖1：成績（EM + NegF1） ───────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 5.2))
fig1.patch.set_facecolor("#ffffff")
ax1.set_facecolor("white")

b1 = ax1.bar(x - w/2, em,     width=w, color=EM_C,  label="ExactMatch", zorder=3, edgecolor="white", linewidth=0.8)
b2 = ax1.bar(x + w/2, neg_f1, width=w, color=NEG_C, label="NegF1",      zorder=3, edgecolor="white", linewidth=0.8)

# 數值標注
for bar, v in zip(b1, em):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.6,
             f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1a1a2e")
for bar, v in zip(b2, neg_f1):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.6,
             f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1a1a2e")

# 最佳 EM 標注（qwen3）
best_em_idx = em.index(max(em))
ax1.annotate("EM 最佳", xy=(x[best_em_idx] - w/2, em[best_em_idx] + 4),
             ha="center", fontsize=8.5, color="#2d4ecf",
             arrowprops=dict(arrowstyle="-", color="#2d4ecf", lw=1),
             xytext=(x[best_em_idx] - w/2, em[best_em_idx] + 8))

# 最佳 NegF1 標注（sonnet / qwen3 並列）
for i, (name, v) in enumerate(zip(names, neg_f1)):
    if v >= 98.0:
        ax1.annotate("NegF1 最佳", xy=(x[i] + w/2, v + 4),
                     ha="center", fontsize=8.5, color="#c0662a",
                     xytext=(x[i] + w/2, v + 8))

ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=10.5)
ax1.set_ylim(0, 115)
ax1.set_ylabel("%", fontsize=11)
ax1.yaxis.grid(True, color="#f0f0f0", zorder=0)
ax1.set_axisbelow(True)
ax1.legend(fontsize=11, frameon=False, loc="lower right")
for sp in ax1.spines.values():
    sp.set_color("#dee2e6")
ax1.tick_params(colors="#555")

ax1.set_title("P3 LLM Parser 成績（ExactMatch vs NegF1）",
              fontsize=13, fontweight="bold", color="#1a1a2e", pad=12)
ax1.annotate("eval_conditions_v1 · 2,160q · NegF1 n=500 · 依 ExactMatch 升序",
             xy=(0.5, -0.13), xycoords="axes fraction",
             ha="center", fontsize=9.5, color="#888")

fig1.tight_layout()
fig1.savefig("docs/chart_parser_scores.png", dpi=150, bbox_inches="tight")
print("saved: docs/chart_parser_scores.png")


# ── 圖2：速度（依速度排序，快→慢） ──────────────────────────────────────────
models_spd = sorted(models, key=lambda x: x["ms"])
spd_names  = [m["name"]   for m in models_spd]
spd_ms     = [m["ms"]     for m in models_spd]
spd_em     = [m["em"]     for m in models_spd]
spd_neg    = [m["neg_f1"] for m in models_spd]

# 顏色梯度：越快越藍，越慢越橘
spd_max = max(spd_ms)
spd_min = min(spd_ms)
def _spd_color(ms):
    t = (ms - spd_min) / (spd_max - spd_min)   # 0=快, 1=慢
    r = int(79  + t * (225 - 79))
    g = int(110 + t * (107 - 110))
    b = int(247 + t * (85  - 247))
    return f"#{r:02x}{g:02x}{b:02x}"

spd_colors = [_spd_color(v) for v in spd_ms]

fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor("#ffffff")
ax2.set_facecolor("white")

x2   = np.arange(len(spd_names))
bars = ax2.bar(x2, spd_ms, color=spd_colors, width=0.52,
               edgecolor="white", linewidth=0.8, zorder=3)

# ms 數值（bar 上方）
for bar, v in zip(bars, spd_ms):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 6,
             f"{v} ms", ha="center", va="bottom",
             fontsize=10.5, fontweight="bold", color="#1a1a2e")

# EM / NegF1 小字（bar 內）
for bar, v, em_v, neg_v in zip(bars, spd_ms, spd_em, spd_neg):
    if v > 100:   # bar 夠高才放
        ax2.text(bar.get_x() + bar.get_width()/2, v * 0.48,
                 f"EM {em_v:.1f}%\nNeg {neg_v:.1f}%",
                 ha="center", va="center",
                 fontsize=8.5, color="white", fontweight="bold")

ax2.set_xticks(x2)
ax2.set_xticklabels(spd_names, fontsize=10.5)
ax2.set_ylim(0, 720)
ax2.set_ylabel("ms / query", fontsize=11)
ax2.yaxis.grid(True, color="#f0f0f0", zorder=0)
ax2.set_axisbelow(True)
for sp in ax2.spines.values():
    sp.set_color("#dee2e6")
ax2.tick_params(colors="#555")

ax2.set_title("P3 LLM Parser 推論速度（快 → 慢）",
              fontsize=13, fontweight="bold", color="#1a1a2e", pad=12)
ax2.annotate("10 workers 並行 · Trend Micro endpoint · bar 內顯示 EM / NegF1",
             xy=(0.5, -0.13), xycoords="axes fraction",
             ha="center", fontsize=9.5, color="#888")

fig2.tight_layout()
fig2.savefig("docs/chart_parser_speed.png", dpi=150, bbox_inches="tight")
print("saved: docs/chart_parser_speed.png")
