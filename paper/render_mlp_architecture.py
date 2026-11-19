"""Render the paper's width-256 MLP at its final single-column size.

The canvas is 3.25 inches wide, matching the paper column. All labels use
10 pt or larger Times text at native size, so including the PDF at column
width does not shrink its typography below the body text.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{times}",
    "font.family": "serif",
    "font.serif": ["Times"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

fig = plt.figure(figsize=(3.25, 2.25), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 3.25)
ax.set_ylim(0, 2.25)
ax.set_aspect("equal")
ax.axis("off")

ink = "#243746"
edge = "#8797a4"
xs = (0.49, 1.27, 2.03, 2.80)
ys = (0.52, 0.81, 1.40, 1.69)
output_y = 1.10
r = 0.095

# Each drawn neuron connects to every drawn neuron in the following layer.
# The vertical dots mark omitted input and hidden units.
for left, right, src, dst in (
    (xs[0], xs[1], ys, ys),
    (xs[1], xs[2], ys, ys),
    (xs[2], xs[3], ys, (output_y,)),
):
    for y0 in src:
        for y1 in dst:
            ax.plot((left + r, right - r), (y0, y1),
                    color=edge, linewidth=0.48, alpha=0.47, zorder=1)

for x, fill, stroke in (
    (xs[0], "#DCEAF6", "#5F91BC"),
    (xs[1], "#F8DFAE", "#D2A657"),
    (xs[2], "#E6DBF3", "#A789C4"),
):
    for y in ys:
        ax.add_patch(Circle((x, y), r, facecolor=fill,
                            edgecolor=stroke, linewidth=0.9, zorder=3))
    ax.text(x, 1.105, r"$\vdots$", ha="center", va="center",
            fontsize=10, color=ink, zorder=4)

ax.add_patch(Circle((xs[3], output_y), 0.12, facecolor="#D7EEDD",
                    edgecolor="#72A982", linewidth=1.0, zorder=3))
ax.text(xs[3], output_y, r"$z$", ha="center", va="center",
        fontsize=10, color=ink, zorder=4)

for y, label in zip(reversed(ys),
                    (r"$\tilde{x}_1$", r"$\tilde{x}_2$", r"$\tilde{x}_{d-1}$", r"$\tilde{x}_d$")):
    ax.text(0.33, y, label, ha="right", va="center", fontsize=10, color=ink)

for x, title in zip(xs, ("Input", "Hidden 1", "Hidden 2", "Output")):
    ax.text(x, 2.04, title, ha="center", va="center", fontsize=10, color=ink)
for x, detail in zip(xs, (r"$d=20,30,40$", "256 ReLU", "256 ReLU", "1 logit")):
    ax.text(x, 0.17, detail, ha="center", va="center", fontsize=10, color=ink)

# Deliberately avoid bbox_inches='tight': the PDF's natural width is exactly
# one paper column, so its 10 pt lettering stays 10 pt in the manuscript.
for extension in ("pdf", "png"):
    fig.savefig(OUT / f"mlp_architecture.{extension}", dpi=320,
                facecolor="white")
plt.close(fig)
