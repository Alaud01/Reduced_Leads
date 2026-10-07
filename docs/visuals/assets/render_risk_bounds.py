"""Render the poster's validation rule-out error plot with Matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT = Path(__file__).with_name("ruleout-risk-bounds.png")

labels = ["Fixed I + II", "Random leads", "12-lead rescue\n(full-input model)"]
values = [2.60, 2.40, 3.35]
colors = ["#002145", "#0077C8", "#0A8A5F"]
positions = [2, 1, 0]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, ax = plt.subplots(figsize=(10.8, 3.8), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_position([0.29, 0.20, 0.66, 0.65])

ax.set_xlim(0, 4.4)
ax.set_ylim(-0.55, 2.65)
ax.set_xticks([0, 1, 2, 3, 4], ["0%", "1%", "2%", "3%", "4%"])
ax.set_yticks(positions, labels)
ax.tick_params(axis="x", colors="#606870", labelsize=13, length=0, pad=12)
ax.tick_params(axis="y", colors="#20242a", labelsize=14, length=0, pad=16)

ax.grid(axis="x", color="#dfe3e7", linewidth=1.1, zorder=0)
for y in positions:
    ax.axhline(y, color="#e6e9ec", linewidth=1.1, zorder=0)

ax.barh(positions, values, height=0.40, color=colors, zorder=3)
for y, value, color in zip(positions, values, colors):
    ax.text(value + 0.08, y, f"{value:.2f}%", va="center", ha="left",
            color=color, fontsize=14, fontweight="bold")

ax.axvline(2, color="#b92332", linewidth=2, linestyle=(0, (4, 3)), zorder=2)
ax.text(2, 1.04, "2% limit", transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", color="#b92332", fontsize=13)

for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#c9ced2")
ax.spines["bottom"].set_linewidth(1.1)

fig.savefig(OUTPUT, dpi=150, facecolor="white")
plt.close(fig)
