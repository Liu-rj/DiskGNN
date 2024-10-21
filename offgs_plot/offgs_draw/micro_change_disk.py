import csv

import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True
SAVE_PTH = "../figures"
font_size = 16


file_path = "../offgs_data/change_disk.csv"

with open(file_path, mode="r", newline="") as file:
    reader = csv.reader(file)
    data = list(reader)

for row in data:
    print(row)

for i, row in enumerate(data[1:]):
    normalize = float(row[2])
    for j in range(2, len(row)):
        row[j] = str(round(float(row[j]) / normalize, 2))
    data[i + 1] = row

##
print("after normalization")
speedup_list = []
for it, row in enumerate(data):
    print(row)
    if it > 0:
        speedup_list += row[3:]
print("Average Speedup:", np.mean([float(x) for x in speedup_list]))


models = ["SAGE"]
datasets = ["Ogbn-papers100M", "MAG240M", "Friendster", "IGB-HOM"]
data = [[float(cell) if cell.isdigit() else cell for cell in row] for row in data[1:]]
for k, model in enumerate(models):

    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches(10, 3)
    plt.subplots_adjust(wspace=0, hspace=0)

    total_width, n = 4, 4
    group = 1
    width = total_width * 0.9 / n
    x = np.arange(group) * n
    exit_idx_x = x + (total_width - width) / n
    hatches = ["", "//", "x", ".."]
    labels = [
        "DiskGNN",
        "MariusGNN",
        "Ginex",
        "DGL-OnDisk",
    ]
    colorlist = ["white", "white", "white", "k"]
    if k == 0:
        x_labels = "GraphSAGE"
    else:
        x_labels = "GAT"

    yticks = np.arange(0, 4, 1)

    for i in range(4):
        val_limit = yticks[-1]
        axes[i].set_yticks(yticks)
        axes[i].set_ylim(0, val_limit)

        axes[i].tick_params(axis="y", labelsize=10)  # 设置 y 轴刻度标签的字体大小

        axes[i].set_xticks([])
        # axes[i].set_xticklabels()
        axes[i].set_xlabel(datasets[i], fontsize=font_size)
        axes[i].grid(axis="y", linestyle="--")
        for j in range(n):
            ##TODO add label
            num = round(float(data[i + k][j + 2]), 2)
            plot_label = [num]
            if num == 0:
                plot_label = ["N/A"]
            container = axes[i].bar(
                exit_idx_x + j * width,
                [num] if num < val_limit else [val_limit],
                width=width * 0.8,
                color="white",
                # edgecolor=edgecolors[j],
                edgecolor="k",
                hatch=hatches[j],
                linewidth=1.0,
                label=labels[j],
                zorder=10,
            )
            axes[i].bar_label(
                container,
                plot_label,
                fontsize=font_size - 2,
                zorder=200,
                fontweight="bold",
            )

    if k == 0:
        axes[0].legend(
            bbox_to_anchor=(3.7, 1.08),
            ncol=4,
            loc="lower right",
            # fontsize=font_size,
            # markerscale=3,
            labelspacing=0.1,
            edgecolor="black",
            facecolor="white",
            framealpha=1,
            shadow=False,
            # fancybox=False,
            handlelength=1.5,
            handletextpad=0.6,
            columnspacing=0.8,
            prop={"weight": "bold", "size": font_size},
        ).set_zorder(100)

    axes[0].set_ylabel("Normalized Runtime", fontsize=font_size, fontweight="bold")
    axes[0].set_yticklabels(yticks, fontsize=font_size)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    axes[3].set_yticklabels([])

    plt.savefig(
        f"{SAVE_PTH }/micro_change_disk_{model}.pdf", bbox_inches="tight", dpi=300
    )
    ## print save
    print(f"{SAVE_PTH }/micro_change_disk_{model}.pdf")
