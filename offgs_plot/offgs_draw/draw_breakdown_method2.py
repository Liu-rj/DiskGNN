import numpy as np
import matplotlib.pyplot as plt
import csv
from draw_utils import *

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True
SAVE_PTH = "../figures"
font_size = 16


import csv

file_path = "../offgs_data/breakdown_method2.csv"

with open(file_path, mode="r", newline="") as file:
    reader = csv.reader(file)
    data = list(reader)
print(data)


config = []
all_normalized_runtime = []
all_speedup = []
all_disk_blowup = []
dataset = ["Friendster", "IGB-HOM"]
for i, row in enumerate(data):
    if len(row) > 0 and i > 0:
        if row[1] == "Time":
            statistics = row[2:]
            ## convert to float

            statistics = [float(stat) for stat in statistics]
            speedup = [round(statistics[0] / float(stat), 2) for stat in statistics]
            normalized_runtime = [
                round(float(stat) / statistics[0], 2) for stat in statistics
            ]
            all_normalized_runtime.append(normalized_runtime)
            all_speedup.append(speedup)
        else:
            all_disk_blowup.append([float(ele[:-1]) for ele in row[2:]])
print("finish preprocessing")
print(all_normalized_runtime)
print(all_speedup)
print(all_disk_blowup)

total_width, n = 5, 5
group = 1
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "lightseagreen", "tomato", "slategray", "silver"]
hatches = ["x", "//", "\\\\", "--", ""]
hatches = hatches[::-1]
labels = ["Raw", "Raw+G", "Raw+G+C", "Raw+G+C+P", "Raw+G+C+P+D"]

x_labels = dataset

for it, ydata in enumerate([all_normalized_runtime, all_speedup]):
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 3)
    plt.subplots_adjust(wspace=0, hspace=0)

    if it == 0:
        yticks = np.arange(0, 1.3, 0.2)
        yticks = [round(ele, 2) for ele in yticks]
        tx_yticks = np.arange(0, 13, 2)
    else:
        yticks = np.arange(0, 46, 9)
        tx_yticks = np.arange(-5, 21, 5)

    for i in range(2):
        val_limit = yticks[-1]
        axes[i].set_yticks(yticks)
        axes[i].set_ylim(0, val_limit)
        axes[i].tick_params(axis="y", labelsize=10)  # 设置 y 轴刻度标签的字体大小

        axes[i].set_xticklabels([])
        axes[i].set_xticks([])
        # axes[i].set_xticklabels([], fontsize=8)
        axes[i].set_xlabel(x_labels[i], fontsize=font_size)
        axes[i].grid(axis="y", linestyle="--")
        for j in range(n):
            ##TODO add label
            # plot_label= [data[j] for data in all_new_normalized_statistics[(line_num-1)*6+i*group:(line_num-1)*6+i*group+3]]
            plot_label = [ydata[i][j]]
            container = axes[i].bar(
                exit_idx_x + j * width,
                ydata[i][j],
                width=width * 0.8,
                color="white",
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

        ax2 = axes[i].twinx()
        # plot_label = all_disk_blowup[i]
        x = [exit_idx_x + j * width for j in range(n)]
        container = ax2.plot(
            x,
            all_disk_blowup[i],
            label=f"Disk Blowup",
            linestyle="--",
            color="black",
            marker="o",
            markersize=10,
            markerfacecolor="none",
            linewidth=1,
        )
        ax2.set_yticks(tx_yticks)
        ax2.set_yticklabels([])
        # ax2.set_ylim(0, val_limit)
        ax2.tick_params(axis="y", labelsize=10)  # 设置 y 轴刻度标签的字体大小
        # axes[i].bar_label(
        #     container, plot_label, fontsize=font_size - 2, zorder=200, fontweight="bold"
        # )

    axes[0].plot(
        [],
        [],
        linestyle="--",
        color="black",
        marker="o",
        markersize=5,
        markerfacecolor="none",
        linewidth=1,
        label="Disk Blowup",
    )
    axes[0].legend(
        bbox_to_anchor=(2.07, 1.02),
        ncol=6,
        loc="lower right",
        # fontsize=font_size,
        # markerscale=3,
        labelspacing=0.1,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
        shadow=False,
        # fancybox=False,
        handlelength=1.3,
        handletextpad=0.5,
        columnspacing=0.5,
        prop={"weight": "bold", "size": font_size - 2},
    ).set_zorder(100)

    title_label = "Normalized Runtime" if it == 0 else "Speedup"
    axes[0].set_ylabel(title_label, fontsize=font_size)
    axes[0].set_yticklabels(yticks, fontsize=font_size)
    axes[1].set_yticklabels([])
    ax0_tx, ax1_tx = axes[0].twinx(), axes[1].twinx()
    ax1_tx.set_ylabel("Disk Space Blowup", fontsize=font_size)
    ax1_tx.set_yticks(tx_yticks)
    ax1_tx.set_yticklabels(tx_yticks, fontsize=font_size)
    ax0_tx.set_yticks(tx_yticks)
    ax0_tx.set_yticklabels([])
    file_label = "runtime" if it == 0 else "speedup"
    file_name = f"{SAVE_PTH}/breakdown_method2_{file_label}.pdf"
    plt_save_and_final(file_name)
