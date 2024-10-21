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


file_path = "../Graph_NN_Benchmarks.csv"

with open(file_path, mode="r", newline="") as file:
    reader = csv.reader(file)
    data = list(reader)

# 打印原始数据
for row in data:
    print(",".join(row))

# 修改数据：将第六列的值加到第五列
for it, row in enumerate(data):
    data[it] = row[:-1]
    data[it][7] = "Ginex+Preprocess" if it == 0 else str(float(row[7]) + float(row[6]))


# 打印修改后的数据
for row in data:
    print(",".join(row))

## save the new data
# new_file_path = "../Graph_NN_Benchmarks_new.csv"
# with open(new_file_path, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerows(data)

for row in data[1:]:
    normalize = float(row[2])
    for j in range(2, len(row)):
        row[j] = str(round(float(row[j]) / normalize, 2))
##
print("after normalization")
speedup_list = []
for it, row in enumerate(data):
    print(",".join(row))
    if it > 0:
        if it == 7 or it == 8:
            speedup_list += row[4:-2]
        else:
            speedup_list += row[4:-1]
print("Average Speedup:", np.mean([float(x) for x in speedup_list]))


models = ["SAGE", "GAT"]
datasets = ["Ogbn-papers100M (5GB)", "MAG240M (10GB)", "Friendster (3GB)", "IGB-HOM (15GB)"]
data = [[float(cell) if cell.isdigit() else cell for cell in row] for row in data[1:]]
for k, model in enumerate(models):

    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches(20, 3)
    plt.subplots_adjust(wspace=0, hspace=0)

    total_width, n = 7, 7
    group = 1
    width = total_width * 0.9 / n
    x = np.arange(group) * n
    exit_idx_x = x + (total_width - width) / n
    edgecolors = ["dimgrey", "lightseagreen", "tomato", "slategray", "silver"]
    hatches = ["", "\\\\", "//", "||", "x", "--", ".."]

    labels = [
        "DiskGNN",
        "DiskGNN+Preprocess",
        "MariusGNN",
        "MariusGNN+Preprocess",
        "Ginex",
        "Ginex+Preprocess",
        "DGL-OnDisk",
    ]
    if k == 0:
        x_labels = "GraphSAGE"
    else:
        x_labels = "GAT"

    yticks = np.arange(0, 11, 2)
    val_limit = 11

    for i in range(4):
        # val_limit = yticks[-1]
        axes[i].set_yticks(yticks)
        axes[i].set_ylim(0, val_limit)

        axes[i].tick_params(axis="y", labelsize=10)  # 设置 y 轴刻度标签的字体大小

        axes[i].set_xticks([])
        # axes[i].set_xticklabels()
        axes[i].set_xlabel(datasets[i], fontsize=font_size)
        axes[i].grid(axis="y", linestyle="--")
        for j in range(n):
            ##TODO add label

            num = float(data[i * 2 + k][j + 2])
            plot_label = [num]
            if j == 6 and i == 3:
                plot_label = ["N/A"]
                num = 0
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
            bbox_to_anchor=(3.95, 1.08),
            ncol=7,
            loc="lower right",
            # fontsize=font_size,
            # markerscale=3,
            labelspacing=0.2,
            edgecolor="black",
            facecolor="white",
            framealpha=1,
            shadow=False,
            # fancybox=False,
            handlelength=2,
            handletextpad=0.5,
            columnspacing=0.5,
            prop={"weight": "bold", "size": font_size},
        ).set_zorder(100)

    axes[0].set_ylabel("Normalized Runtime", fontsize=font_size, fontweight="bold")
    axes[0].set_yticklabels(yticks, fontsize=font_size)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    axes[3].set_yticklabels([])

    plt.savefig(f"{SAVE_PTH }/speed_{model}_revised.pdf", bbox_inches="tight", dpi=300)
    ## print save
    print(f"{SAVE_PTH }/speed_{model}_revised.pdf")
