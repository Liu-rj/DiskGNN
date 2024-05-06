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

# 修改数据：将第六列的值加到第五列
for row in data[1:]:
    row[4] = str(round(float(row[4]) + float(row[5]), 2))
    row[4], row[5], row[6], row[7] = row[7], row[5], row[4], row[8]
    ##remove row[8]
    row = row[:-2]


# 打印修改后的数据
for row in data:
    print(",".join(row))

## save the new data
new_file_path = "../Graph_NN_Benchmarks_new.csv"
with open(new_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

for row in data[1:]:
    normalize = float(row[2])
    for j in range(2, 8):
        row[j] = str(round(float(row[j]) / normalize, 2))
##
print("after normalization")
speedup_list = []
marius_preprocess = []
for it, row in enumerate(data):
    print(",".join(row))
    if it > 0:
        marius_preprocess.append(row[3])
        if it == 7 or it == 8:
            speedup_list += row[4:-2]
        else:
            speedup_list += row[4:-1]
print("Average Speedup:", np.mean([float(x) for x in speedup_list]))


models = ["SAGE", "GAT"]
datasets = ["Ogbn-papers100M", "MAG240M", "Friendster", "IGB-HOM"]
data = [[float(cell) if cell.isdigit() else cell for cell in row] for row in data[1:]]
for k, model in enumerate(models):

    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches(20, 3)
    plt.subplots_adjust(wspace=0, hspace=0)

    total_width, n = 6, 6
    group = 1
    width = total_width * 0.9 / n
    x = np.arange(group) * n
    exit_idx_x = x + (total_width - width) / n
    edgecolors = ["dimgrey", "lightseagreen", "tomato", "slategray", "silver"]
    hatches = ["", "\\\\", "//", "x", "--", "..", "xx", "oo", ".."]
    labels = [
        "DiskGNN",
        "DiskGNN+Preprocess",
        "MariusGNN",
        "Ginex",
        "Ginex+Sample",
        "DGL-OnDisk",
    ]
    colorlist = ["white", "white", "white", "white", "white", "k"]
    if k == 0:
        x_labels = "GraphSAGE"
    else:
        x_labels = "GAT"

    yticks = np.arange(0, 11, 2)

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

            num = float(data[i * 2 + k][j + 2])
            plot_label = [num]
            if j == 5 and i == 3:
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
            if j == 5 and i == 3:
                plot_label = ["N/A"]
            axes[i].bar_label(
                container,
                plot_label,
                fontsize=font_size - 2,
                zorder=200,
                fontweight="bold",
            )

    if k == 0:
        axes[0].legend(
            bbox_to_anchor=(3.6, 1.08),
            ncol=6,
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
            handletextpad=1,
            columnspacing=0.8,
            prop={"weight": "bold", "size": font_size},
        ).set_zorder(100)

    axes[0].set_ylabel("Normalized Runtime", fontsize=font_size, fontweight="bold")
    axes[0].set_yticklabels(yticks, fontsize=font_size)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    axes[3].set_yticklabels([])

    plt.savefig(f"{SAVE_PTH }/speed_{model}.pdf", bbox_inches="tight", dpi=300)
    ## print save
    print(f"{SAVE_PTH }/speed_{model}.pdf")
