"""Dumbbell chart for poster section 04: two-lead AUROC by training approach
(Fixed I+II vs Random leads, matched seed 42) plus the shared full-12-lead
reference model. Record-level test metrics (n=2,198 ECGs).

Source : results/runpod-s42/all_model_metrics.csv
Output : docs/visuals/assets/training-strategy-tradeoff.png (1620x570, for 540px column)
Usage  : python docs/visuals/assets/plot_tradeoff.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Record-level AUROC from all_model_metrics.csv (n=2,198 test ECGs).
DIAGNOSES = [
    "AFIB-or-AFLT",
    "Myocardial infarction",
    "ST\u2013T changes",
    "Conduction disorders",
    "Normal superclass",
]
FIXED = [0.9287, 0.8409, 0.8991, 0.8496, 0.9112]     # Fixed I+II, on I+II
RANDOM = [0.9483, 0.8053, 0.8918, 0.8359, 0.9023]    # Random-lead training, on I+II
FULL12 = [0.9134, 0.8837, 0.9144, 0.8813, 0.9232]    # Shared fixed12 model, on 12 leads

NAVY, MIDBLUE, GREEN = "#002145", "#0077C8", "#0a8a5f"
GREY_GRID, GREY_TICK = "#e4e4e4", "#666666"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": "#111111",
        "axes.unicode_minus": False,
    }
)

FIG_W, FIG_H, DPI = 5.4, 1.9, 300  # -> 1620x570px for the ~540px poster column
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
ax = fig.add_axes([0.27, 0.20, 0.68, 0.60])  # room left for names, top for legend

ys = list(range(len(DIAGNOSES) - 1, -1, -1))  # AFIB on top
ax.set_xlim(0.795, 0.955)
ax.set_ylim(-0.55, 4.75)
ax.set_yticks(ys)
ax.set_yticklabels(DIAGNOSES, fontsize=8, color="#111111")
ax.set_xticks([0.80, 0.85, 0.90, 0.95])
ax.set_xticklabels(["0.80", "0.85", "0.90", "0.95"], fontsize=7, color=GREY_TICK)
ax.xaxis.grid(True, color=GREY_GRID, linewidth=0.8, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#d4d4d4")
ax.tick_params(left=False, bottom=False)

series = [
    ("Fixed I + II", FIXED, NAVY, 0.22, "bottom"),
    ("Random leads", RANDOM, MIDBLUE, -0.22, "top"),
    ("Full 12-lead", FULL12, GREEN, 0.22, "bottom"),
]

label_artists = []
for y, f, r, g in zip(ys, FIXED, RANDOM, FULL12):
    vals = [f, r, g]
    ax.hlines(y, min(vals), max(vals), colors="#111111", linewidth=1.2, zorder=3)
    for (name, data, color, dy, va), v in zip(series, vals):
        is_max = v == max(vals)
        ax.scatter(
            [v], [y], s=36, c=color, edgecolors="white",
            linewidths=0.9, zorder=4,
        )
        label_artists.append(
            ax.text(
                v, y + dy, f"{v:.3f}", ha="center", va=va, fontsize=7,
                color=color, weight="bold" if is_max else "normal", zorder=5,
            )
        )
    # horizontal hairline across the plot so each name tracks to its row
    ax.hlines(y, 0.795, 0.955, colors=GREY_GRID, linewidth=0.8, zorder=1)

ax.legend(
    handles=[
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               markerfacecolor=c, markeredgecolor="white", label=n)
        for n, _, c, _, _ in series
    ],
    loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=3, frameon=False,
    fontsize=7, handletextpad=0.4, columnspacing=1.6,
)

fig.canvas.draw()
boxes = [t.get_window_extent() for t in label_artists]
worst = 0.0
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        ox = min(boxes[i].x1, boxes[j].x1) - max(boxes[i].x0, boxes[j].x0)
        oy = min(boxes[i].y1, boxes[j].y1) - max(boxes[i].y0, boxes[j].y0)
        if ox > 0 and oy > 0:
            worst = max(worst, ox * oy)
            print(f"OVERLAP: {label_artists[i].get_text()} x {label_artists[j].get_text()}")
print("overlap check:", "FAIL" if worst else "OK (no label overlaps)")

OUTPUT = Path(__file__).with_name("training-strategy-tradeoff.png")
fig.savefig(OUTPUT, dpi=DPI, facecolor="white")
print(f"wrote {OUTPUT}")
