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


import csv

line_nums = [1, 2]
file_path = "../offgs_data/vary_cache.csv"

with open(file_path, mode="r", newline="") as file:
    reader = csv.reader(file)
    data = list(reader)

# Skipping the header row and ensuring all data is float or int as needed
data = [[float(cell) if cell.isdigit() else cell for cell in row] for row in data[1:]]

config = []
statistics = []
all_configs = []
all_statistics = []
all_normalized_statistics = []
for i, row in enumerate(data):
    if i % 3 == 0:
        config = []
        statistics = []
        config.append(row[0])
        config.append(row[1])
        statistics.append(row[-2])
    if i % 3 == 1:

        statistics.append(
            float(row[-2]) + float(row[-1])
        )  # Assumes row[-2] and row[-1] are now numbers
        statistics.append(row[-1])
    if i % 3 == 2:

        statistics.append(row[-1])
        statistics[1], statistics[2], statistics[3] = (
            statistics[3],
            statistics[2],
            statistics[1],
        )
        statistics = [float(stat) for stat in statistics]
        statistics_normalized = [round(stat / statistics[0], 2) for stat in statistics]
        all_configs.append(config)
        all_statistics.append(statistics)
        all_normalized_statistics.append(statistics_normalized)
        print(config)
        print(statistics)
        print(statistics_normalized)
all_new_normalized_statistics = []
for i, stats in enumerate(all_statistics):
    if i % 3 == 0:
        pivot = stats[0]
    statistics_normalized = [round(stat / pivot, 2) for stat in stats]
    all_new_normalized_statistics.append(statistics_normalized)

for line_num in line_nums:

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 3)
    plt.subplots_adjust(wspace=0, hspace=0)

    total_width, n = 3, 3
    group = 3
    width = total_width * 0.9 / n
    x = np.arange(group) * n
    exit_idx_x = x + (total_width - width) / n
    edgecolors = ["dimgrey", "lightseagreen", "tomato", "slategray", "silver"]
    # hatches = ["/////", "\\\\\\\\\\"]
    hatches = ["", "//", "x", "|||", "---", "...", "||", "xx", "oo", ".."]

    labels = ["Our", "MariusGNN", "Ginex", "Ginex+Sample"]
    if line_num == 1:
        x_labels = [all_configs[0][0], all_configs[3][0]]
    else:
        x_labels = [all_configs[6][0], all_configs[9][0]]

    yticks = np.arange(0, 11, 2)

    for i in range(2):
        val_limit = yticks[-1]
        axes[i].set_yticks(yticks)
        axes[i].set_ylim(0, val_limit)

        axes[i].tick_params(axis="y", labelsize=10)  # 设置 y 轴刻度标签的字体大小

        axes[i].set_xticks([1.6, 4.6, 7.6])
        axes[i].set_xticklabels(["10\%", "30\%", "50\%"], fontsize=font_size)
        axes[i].set_xlabel(x_labels[i], fontsize=font_size)
        axes[i].grid(axis="y", linestyle="--")
        for j in range(n):
            ##TODO add label
            plot_label = [
                data[j]
                for data in all_new_normalized_statistics[
                    (line_num - 1) * 6 + i * group : (line_num - 1) * 6 + i * group + 3
                ]
            ]
            # if j==1:
            #     print(                exit_idx_x + j * width,)

            container = axes[i].bar(
                exit_idx_x + j * width,
                [
                    data[j] if data[j] < val_limit else val_limit
                    for data in all_new_normalized_statistics[
                        (line_num - 1) * 6
                        + i * group : (line_num - 1) * 6
                        + i * group
                        + 3
                    ]
                ],
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
                fontsize=font_size - 4,
                zorder=200,
                fontweight="bold",
            )

    if line_num == 1:
        axes[0].legend(
            bbox_to_anchor=(1.5, 1.02),
            ncol=4,
            loc="lower right",
            # fontsize=font_size + 4,
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

    axes[0].set_ylabel("Normalized Runtime", fontsize=font_size)
    axes[0].set_yticklabels(yticks, fontsize=font_size)
    axes[1].set_yticklabels([])
    plt.savefig(f"{SAVE_PTH }/vary_mem_{line_num}.pdf", bbox_inches="tight", dpi=300)
    ## print save
    print(f"Save to {SAVE_PTH }/vary_mem_{line_num}.pdf")
    # %%
