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


fanout_file_path = "../offgs_data/fanout_blowup.csv"
batchsize_file_path = "../offgs_data/batchsize_blowup.csv"

with open(fanout_file_path, mode="r", newline="") as file:
    reader = csv.reader(file)
    fanout_data = list(reader)

for row in fanout_data:
    print(row)

with open(batchsize_file_path, mode="r", newline="") as file:
    reader = csv.reader(file)
    batchsize_data = list(reader)

for row in batchsize_data:
    print(row)

fig, axes = plt.subplots(1, 2)
fig.set_size_inches(10, 3)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 2, 2
group = 3
width = total_width * 0.7 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["tomato", "slategray", "skyblue", "silver", "silver"]
hatches = ["", "//", "|||", "|||", "---", "...", "||", "xx", "oo", ".."]

labels = ["Friendster", "IGB-HOM"]

x_labels = ["Fanout", "Batch Size"]
fanout_x_labels = fanout_data[0][2:]
batchsize_x_labels = batchsize_data[0][2:]

yticks = np.arange(0, 13, 2)
# Setting labels and title
val_limit = yticks[-1]

for i, data in enumerate([fanout_data, batchsize_data]):
    val_limit = yticks[-1]
    axes[i].set_yticks(yticks)
    axes[i].set_ylim(0, val_limit)

    axes[i].tick_params(axis="y", labelsize=10)  # 设置 y 轴刻度标签的字体大小

    axes[i].set_xticks(exit_idx_x + width / 2)
    axes[i].set_xticklabels(data[0][2:], fontsize=font_size)
    axes[i].set_xlabel(x_labels[i], fontsize=font_size)
    axes[i].grid(axis="y", linestyle="--")
    for j in range(n):
        ##TODO add label
        num = [round(float(ele), 2) for ele in data[j + 1][2:]]
        container = axes[i].bar(
            exit_idx_x + j * width,
            num,
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
            num,
            fontsize=font_size - 2,
            zorder=200,
            fontweight="bold",
        )

axes[0].legend(
    # bbox_to_anchor=(1.45, 1),
    # ncol=2,
    loc="best",
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

axes[0].set_ylabel("Disk Size Blowup", fontsize=font_size, fontweight="bold")
axes[0].set_yticklabels(yticks, fontsize=font_size)
axes[1].set_yticklabels([])

plt.savefig(f"{SAVE_PTH }/main_fanout_batchsize_blowup.pdf", bbox_inches="tight", dpi=300)
## print save
print(f"{SAVE_PTH }/main_fanout_batchsize_blowup.pdf")
