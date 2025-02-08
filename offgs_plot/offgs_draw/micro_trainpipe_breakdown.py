import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from draw_utils import *

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True
SAVE_PTH = "../figures"
font_size = 16

# Load the CSV file
file_path = "../offgs_data/trainpipe_breakdown.csv"
df = pd.read_csv(file_path)

# Drop the DGL-onDisk row
df = df[df["system"] != "DGL-onDisk"]

# Separate the datasets
papers_df = df[df["dataset"] == "papers100M"]
mag_df = df[df["dataset"] == "mag240M"]

# Define the stages to be compared
stages = [
    "graph load",
    "feature load",
    "graph sample",
    "feature assemble",
    "model train",
]


# Function to create a stacked bar plot for a single dataset
def plot_single_dataset(ax, dataset_df, dataset_name, draw_xlabels=False):
    systems = dataset_df["system"].values
    x = np.arange(len(systems))
    width = 0.5
    x_limit = 700 if dataset_name == "Ogbn-papers100M" else 600

    # Initialize bottom for stacking
    bottom = np.zeros(len(systems))

    # Define colors and hatching for each segment type
    colors = ["black", "grey", "white", "lightgrey", "white"]
    hatches = [None, None, None, None, "..."]

    # Plot each stage as a stacked bar segment
    for idx, stage in enumerate(stages):
        times = dataset_df[stage].fillna(0).values
        ax.barh(
            x,
            times,
            height=width,
            color=colors[idx % len(colors)],
            hatch=hatches[idx % len(hatches)],
            left=bottom,
            label=stage,
            edgecolor="black",
            zorder=10,
        )
        bottom += times

    # Add total time labels at the right of each stacked bar
    total_times = dataset_df["total time"].values
    pos = [700 if total > 700 else total for total in total_times]
    for i, total in enumerate(total_times):
        ax.text(pos[i] + 5, i, f"{total:.1f}", va="center", fontsize=font_size)

    ax.set_xlim(0, x_limit)
    ax.set_xticks(np.arange(0, x_limit + 1, 100))
    ax.set_xticklabels(np.arange(0, x_limit + 1, 100), fontsize=font_size)
    ax.set_yticks(x)
    ax.set_yticklabels(systems, fontsize=font_size)
    # ax.set_title(title)
    if draw_xlabels:
        ax.set_xlabel("Runing Time (sec)", fontsize=font_size)
    ax.set_ylabel(dataset_name, fontsize=font_size)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)


# Create subplots for the two datasets
fig, axes = plt.subplots(2, 1, figsize=(10, 5))
plt.subplots_adjust(wspace=0, hspace=0.4)

# Plot for ogbn-papers100M
plot_single_dataset(axes[0], papers_df, "Ogbn-papers100M", draw_xlabels=True)

# Plot for mag240m
plot_single_dataset(axes[1], mag_df, "MAG240M", draw_xlabels=True)

# Add legend to the bottom plot
axes[0].legend(
    bbox_to_anchor=(1.05, 1),
    ncol=5,
    loc="lower right",
    labelspacing=0.1,
    edgecolor="black",
    facecolor="white",
    framealpha=1,
    shadow=False,
    # fancybox=False,
    handlelength=1.8,
    handletextpad=0.5,
    columnspacing=0.5,
    prop={"weight": "bold", "size": font_size - 2},
).set_zorder(100)

file_name = f"{SAVE_PTH}/micro_trainpipe_breakdown.pdf"
plt_save_and_final(file_name)
