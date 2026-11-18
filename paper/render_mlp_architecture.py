"""Draw the member C MLP as a conventional neuron-and-connection diagram."""

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
from matplotlib.patches import Circle, FancyArrowPatch


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(11.6, 6.0))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 11.1)
ax.set_ylim(0, 5.7)
ax.set_aspect("equal")
ax.axis("off")

ink = "#253746"
muted = "#586B78"
connection = "#8A9AA7"
x_input, x_hidden1, x_hidden2, x_output = 1.25, 3.55, 5.90, 8.25
y_input = (1.15, 1.90, 3.60, 4.35)
y_hidden = (0.70, 1.35, 2.00, 3.50, 4.15, 4.80)
y_output = 2.75


def neuron(x, y, radius, fill, stroke, label=None, label_color=ink):
    ax.add_patch(Circle((x, y), radius, facecolor=fill, edgecolor=stroke,
                        linewidth=1.35, zorder=3))
    if label is not None:
        ax.text(x, y, label, ha="center", va="center", color=label_color,
                fontsize=10.2, zorder=4)


def connect(x_left, ys_left, x_right, ys_right, r_left, r_right, opacity):
    for y_left in ys_left:
        for y_right in ys_right:
            ax.plot((x_left + r_left, x_right - r_right), (y_left, y_right),
                    color=connection, linewidth=0.75, alpha=opacity,
                    solid_capstyle="round", zorder=1)


# Lines sit behind the nodes. Each visible neuron connects to every visible
# neuron in the following layer; the vertical dots denote omitted neurons.
connect(x_input, y_input, x_hidden1, y_hidden, 0.23, 0.18, 0.43)
connect(x_hidden1, y_hidden, x_hidden2, y_hidden, 0.18, 0.18, 0.38)
connect(x_hidden2, y_hidden, x_output, (y_output,), 0.18, 0.26, 0.50)

for y, label in zip(y_input, (r"$\tilde{x}_d$", r"$\tilde{x}_{d-1}$",
                              r"$\tilde{x}_2$", r"$\tilde{x}_1$")):
    neuron(x_input, y, 0.23, "#DCEAF6", "#5F91BC", label)
for y in y_hidden:
    neuron(x_hidden1, y, 0.18, "#F8DFAE", "#D2A657")
    neuron(x_hidden2, y, 0.18, "#E6DBF3", "#A789C4")
neuron(x_output, y_output, 0.27, "#D7EEDD", "#72A982", r"$z$")

for x in (x_input, x_hidden1, x_hidden2):
    ax.text(x, 2.75, r"$\vdots$", ha="center", va="center",
            fontsize=19, color=muted)

for x, title in ((x_input, "Input layer"), (x_hidden1, "Hidden layer 1"),
                 (x_hidden2, "Hidden layer 2"), (x_output, "Output layer")):
    ax.text(x, 5.35, title, ha="center", va="center", fontsize=12.2,
            fontweight="bold", color=ink)

ax.text(x_input, 0.18, "standardized features\n$d = 20, 30,$ or $40$",
        ha="center", va="top", fontsize=10.3, color=muted, linespacing=1.5)
ax.text(x_hidden1, 0.18, r"256 units $\cdot$ ReLU", ha="center", va="top",
        fontsize=10.3, color=muted)
ax.text(x_hidden2, 0.18, r"256 units $\cdot$ ReLU", ha="center", va="top",
        fontsize=10.3, color=muted)
ax.text(x_output, 0.18, r"1 logit $\cdot$ no activation", ha="center", va="top",
        fontsize=10.3, color=muted)

ax.add_patch(FancyArrowPatch((8.58, y_output), (9.15, y_output),
                             arrowstyle="-|>", mutation_scale=13,
                             linewidth=1.3, color=ink, zorder=2))
ax.text(9.30, y_output, "$z>0$: class 1\notherwise: class 0",
        ha="left", va="center", fontsize=10.4, color=ink,
        linespacing=1.55)

fig.subplots_adjust(left=0.025, right=0.99, top=0.98, bottom=0.045)
for extension in ("png", "pdf"):
    fig.savefig(OUT / f"mlp_architecture.{extension}", dpi=280,
                facecolor="white", bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
