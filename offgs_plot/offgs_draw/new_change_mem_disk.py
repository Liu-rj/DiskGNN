import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np

import pandas as pd
from io import StringIO

# CSV data as string
csv_data = """
dataset,fanout,batchsize,10% cache,15% cache,20% cache,30% cache,50% cache,70% cache
Friendster,"10,15,20",1024,5.397,4.418,2.965,1.614,0.561,0.164
IGB-HOM,"10,15,20",1024,10.487,4.927,1.927,0.149,0.02,0.02
"""

# Using pandas to read this data from a string
data = pd.read_csv(StringIO(csv_data))

# Extract cache columns names
cache_columns = [
    "10% cache",
    "15% cache",
    "20% cache",
    "30% cache",
    "50% cache",
    "70% cache",
]

# Extract datasets names
datasets = data["dataset"]
print(datasets)
# Extract numerical data for caches
cache_data = data[cache_columns].values
cache_columns = ["10\%", "15\%", "20\%", "30\%", "50\%", "70\%"]

# Show extracted information
print("Cache Columns:", cache_columns)
print("Datasets:", datasets.tolist())
print("Cache Data:\n", cache_data)

# Plot settings
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()
font_size = 16

fig.set_size_inches(10, 3)
# plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 2, 2
group = 6
width = total_width * 0.7 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["tomato", "slategray", "skyblue", "silver", "silver"]
hatches = ["", "//", "|||", "|||", "---", "...", "||", "xx", "oo", ".."]

labels = ["Friendster", "IGB-HOM"]

x_labels = cache_columns

yticks = np.arange(0, 13, 2)
# Setting labels and title
val_limit = yticks[-1]

loc = []
read_data = []
for i, (label, dataset) in enumerate(zip(datasets, cache_data)):
    # Plot line
    # axes.plot(x_positions + i * 0.35 - 0.05, dataset, label=label, color=colors[i], linestyle='--')
    dataset = [round(x, 2) for x in dataset]
    # Plot bars
    container = ax.bar(
        exit_idx_x + i * width,
        dataset,
        width=width * 0.8,
        color="w",
        edgecolor="black",
        hatch=hatches[i],
        linewidth=1,
        label=labels[i],
        zorder=10,
    )
    ax.bar_label(
        container, dataset, fontsize=font_size - 2, zorder=200, fontweight="bold"
    )
    ax.grid(axis="y", linestyle="--")


ax.set_xticks(exit_idx_x + 0.45)
ax.set_xticklabels(cache_columns, fontsize=font_size)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=font_size)
ax.set_ylim(0, val_limit)
ax.set_ylabel("Disk Size Blowup", fontsize=font_size)
ax.set_xlabel("Cache Sizes", fontsize=font_size)
ax.legend(
    fontsize=font_size,
    edgecolor="black",
    facecolor="white",
    framealpha=1,
    shadow=False,
    # fancybox=False,
    handlelength=1.5,
    handletextpad=0.6,
    columnspacing=0.8,
    prop={"weight": "bold", "size": font_size},
)
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
# plt.tight_layout()
SAVE_PTH = "../figures"

file_name = f"{SAVE_PTH}/new_cache_disk.pdf"
plt.savefig(file_name, bbox_inches="tight", dpi=300)
## print save
print(f"Save to {file_name}")
# %%
