import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True
SAVE_PTH = "../figures"
font_size = 16


import csv

file_path = "../offgs_data/feat_packing.csv"

with open(file_path, mode="r", newline="") as file:
    reader = csv.reader(file)
    data = list(reader)


config = []
statistics = []
all_configs = []
all_statistics = []
all_normalized_statistics = []
dataset = ["Ogbn-papers100M", "MAG240M", "Friendster", "IGB-HOM"]
for i, row in enumerate(data):
    if len(row) == 0:
        continue
    if i == 0:
        all_configs.append(row[1:])
    else:
        statistics = row[1:]
        ## convert to float

        statistics = [float(stat) for stat in statistics]
        statistics[0], statistics[1], statistics[2], statistics[3] = (
            statistics[2],
            statistics[0],
            statistics[3],
            statistics[1],
        )
        print(statistics)
        all_statistics.append(statistics)
        optimized_preprocessing = statistics[-2]
        tmp_statistics = [
            round(data / optimized_preprocessing, 2) for data in statistics
        ]
        # tmp_statistics=statistics
        # tmp_statistics[0]=round(tmp_statistics[0]/tmp_statistics[2],2)
        # tmp_statistics[1]=round(tmp_statistics[1]/tmp_statistics[2],2)
        # tmp_statistics[2]=round(tmp_statistics[2]/tmp_statistics[2],2)
        # tmp_statistics[3]=round(tmp_statistics[3]/tmp_statistics[2],2)
        # normalized_statistics=[round(float(stat)/float(statistics[-3]),2) for stat in statistics[:-1]]
        all_normalized_statistics.append(tmp_statistics)
print("finish preprocessing")
print(all_normalized_statistics)
fig, axes = plt.subplots()
fig.set_size_inches(10, 3)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 2, 2
group = 4
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["tomato", "slategray", "skyblue", "silver", "silver"]
hatches = ["//", "\\\\"]
labels = [
    "Primitive Preprocessing",
    "Primitive Read",
    "Optimized Preprocessing",
    "Optimized Read",
]

x_labels = dataset

yticks = np.arange(0, 11, 2)

loc = []
read_data = []
for j in range(4):
    ##TODO add label
    # plot_label= [data[j] for data in all_new_normalized_statistics[(line_num-1)*6+i*group:(line_num-1)*6+i*group+3]]
    if j == 1 or j == 3:
        continue
    print("pos ", int(j / 2))
    hatch = ""
    alpa = 1
    zorder = 10
    if j == 0 or j == 1:
        color = "lightseagreen"
        if j == 1:
            hatch = "//"
            color = "white"
            alpa = 0.3
            zorder = 2

    elif j == 2 or j == 3:
        color = "tomato"
        if j == 3:
            hatch = "//"
            color = "white"
            alpa = 0.3
            zorder = 2

    label_data = [data[j] for data in all_normalized_statistics]
    container = axes.bar(
        exit_idx_x + (1 - (int(j / 2))) * width,
        [data[j] for data in all_normalized_statistics],
        width=width * 0.8,
        color=color,
        edgecolor="black",
        hatch=hatch,
        linewidth=0.8,
        label=labels[j],
        alpha=0.7,
        zorder=10,
    )
    loc = loc + list(exit_idx_x + (1 - (int(j / 2))) * width)
    read_data = read_data + [data[j + 1] for data in all_normalized_statistics]
    axes.bar_label(
        container, label_data, fontsize=font_size, zorder=200, fontweight="bold"
    )
    axes.grid(axis="y", linestyle="--")


container = axes.bar(
    loc,
    read_data,
    width=width * 0.8,
    color="none",
    edgecolor="black",
    hatch="///",
    # linewidth=0.3,
    label="Read Time",
    alpha=0.4,
    zorder=10,
)


axes.set_yticks(yticks)
axes.set_ylim(0, 11)
axes.set_yticklabels(yticks, fontsize=font_size)
axes.set_xticks((exit_idx_x + width + exit_idx_x) / 2)
axes.set_xticklabels(x_labels, fontsize=font_size)
# axes.set_xlabel('Dataset Name', fontsize=12)


axes.legend(
    bbox_to_anchor=(1.01, 0.99),
    ncol=3,
    loc="lower right",
    # fontsize=10,
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

axes.set_ylabel("Normalized Runtime", fontsize=font_size)
axes.set_yticklabels(yticks, fontsize=font_size)
file_name = f"{SAVE_PTH}/preprocess_decompose.pdf"
plt.savefig(file_name, bbox_inches="tight", dpi=300)
## print save
print(f"Save to {file_name}")
# %%
