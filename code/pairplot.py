import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# Style settings
# ==========================================================
plt.style.use("seaborn-v0_8-ticks")

print("=" * 70)
print("EDA – Pairplo – FINAL PAPER-READY VERSION")
print("=" * 70)

# ==========================================================
# 1. Load Data
# ==========================================================
while True:
    base_path = input("\nEnter the full folder path containing data.csv:\n> ")
    base_path = base_path.strip().replace('"', '').replace("'", "")
    data_path = os.path.join(base_path, "data.csv")

    if os.path.exists(data_path):
        print(f"File found: {data_path}")
        break
    else:
        print("Error: data.csv not found.")

data = pd.read_csv(data_path)

# ==========================================================
# 2. Categorical Ordering (draw order control)
# ==========================================================
# Background → middle → foreground
hue_order = ["N5", "N3", "N4"]

data["Pile_Group"] = pd.Categorical(
    data["Pile_Group"],
    categories=hue_order,
    ordered=True
)

data = data.sort_values("Pile_Group")

# ==========================================================
# 3. Pairplot definition
# ==========================================================
custom_markers = ["x", "_", "|"]

print("Generating pairplot...")

vars_list = ["Days", "S/D", "L/D", "IF"]
n = len(vars_list)

graph = sns.pairplot(
    data=data,
    vars=vars_list,
    hue="Pile_Group",
    hue_order=hue_order,
    markers=custom_markers,
    corner=True,
    diag_kind="kde",
    palette="colorblind",
    plot_kws={
        "s": 50,
        "alpha": 0.5,
        "linewidth": 0.5
    }
)

# ==========================================================
# 4. RESTORE VERTICAL AXES
# ==========================================================
for i in range(n):

    # Diagonal axes (KDE)
    ax_diag = graph.axes[i, i]
    if ax_diag is not None:
        ax_diag.spines["left"].set_visible(True)
        ax_diag.yaxis.set_visible(True)
        ax_diag.tick_params(axis="y", which="major", labelleft=True)

    # First column axes (scatter)
    ax_firstcol = graph.axes[i, 0]
    if ax_firstcol is not None:
        ax_firstcol.spines["left"].set_visible(True)
        ax_firstcol.yaxis.set_visible(True)
        ax_firstcol.tick_params(axis="y", which="major", labelleft=True)

# Safety net: ensure all left spines are visible
for ax in graph.axes.flatten():
    if ax is not None:
        ax.spines["left"].set_visible(True)

# ==========================================================
# 5. FIX SEMANTIC ERROR:
#    KDE of Days → Y axis must be "Density", NOT "Days"
# ==========================================================
ax_days_diag = graph.axes[0, 0]

if ax_days_diag is not None:
    ax_days_diag.set_ylabel(
        "Density",
        fontsize=11,
        fontweight="bold",
        labelpad=10
    )

# ==========================================================
# 6. Title & Layout
# ==========================================================
graph.fig.suptitle(
    "Pairwise Relationships Between Input Variables and Interaction Factor",
    fontsize=15,
    y=1.02
)

graph.fig.subplots_adjust(
    top=0.93,
    left=0.08,
    right=0.98,
    bottom=0.08
)

# ==========================================================
# 7. Save output
# ==========================================================
output_filename = "Pairplot.png"
output_path = os.path.join(base_path, output_filename)

plt.savefig(output_path, dpi=600, bbox_inches="tight")
plt.show()
plt.close()
print("=" * 70)
print(f"Figure saved at:\n{output_path}")
print("Pairplot generation completed successfully.")
print("=" * 70)
