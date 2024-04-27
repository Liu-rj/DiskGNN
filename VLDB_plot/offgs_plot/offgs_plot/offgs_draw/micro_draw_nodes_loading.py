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

# file_path = "../offgs_data/read_io.csv"
file_path="/home/ubuntu/OfflineSampling/VLDB_plot/offgs_plot/offgs_plot/offgs_data/read_io.csv"
with open(file_path, mode="r", newline="") as file:
    reader = csv.reader(file)
    data = list(reader)


config = []
statistics = []
all_configs = []
all_statistics = []
all_normalized_statistics = []
dataset = ["Friendster", "IGB-HOM"]
for i, row in enumerate(data):
    if len(row) == 0:
        continue
    if i == 0:
        all_configs.append(row[1:])
    else:
        statistics = row[2:]
        ## convert to float
        statistics=[round(float(stat),2) for stat in statistics]
        statistics[-2],statistics[-1]=statistics[-1],statistics[-2]
        statistics = [float(stat) for stat in statistics]
        all_statistics.append(statistics)
        
print("finish preprocessing")
print(all_statistics)
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(10, 3)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 3, 3
group = 3
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "lightseagreen", "tomato", "slategray", "silver"]
hatches = ["", "//", "xx", "--", ""]
labels = all_configs[0][1:]
labels[1],labels[2]=labels[2],labels[1]

x_labels = dataset

log_ticks = np.logspace(0, 4, num=5, base=10)  # 1, 10, 100, 1000, 10000
from matplotlib.ticker import LogFormatterSciNotation

for i in range(2):
 
    axes[i].set_yscale('log')

    # 设置y轴刻度
    axes[i].set_yticks(log_ticks)

    # 第一个图使用标量格式显示刻度标签
    if i == 0:
        formatter = LogFormatterSciNotation()
        axes[i].get_yaxis().set_major_formatter(formatter)

    # 第二个图不显示刻度标签
    if i == 1:
        axes[i].get_yaxis().set_tick_params(labelleft=False)
    
    # 设置y轴的显示范围
    axes[i].set_ylim(1, 13000)

    # 设置刻度标签大小
    axes[i].tick_params(axis="y", labelsize=14)
    axes[i].set_xticks(exit_idx_x+0.9)
    
    
    axes[i].set_xticklabels(["10\%","30\%","50\%"], fontsize=font_size-2)

    # axes[i].set_xticklabels([], fontsize=8)
    axes[i].set_xlabel(dataset[i], fontsize=font_size)
    axes[i].grid(axis="y", linestyle="--")
    for j in range(n):
        ##TODO add label
        # plot_label= [data[j] for data in all_new_normalized_statistics[(line_num-1)*6+i*group:(line_num-1)*6+i*group+3]]
        true_num=[ percent[j] for percent in all_statistics[i*3:i*3+3]]
        plot_label = true_num
        container = axes[i].bar(
            exit_idx_x + j * width,
            [true_num_ if true_num_>1 else 1 for true_num_ in true_num],
            width=width * 0.7,
            color="white",
            edgecolor="k",
            hatch=hatches[j],
            linewidth=1.0,
            label=labels[j],
            zorder=10,
        )
        axes[i].bar_label(
            container, plot_label, fontsize=font_size - 6, zorder=200, fontweight="bold"
        )


axes[0].legend(
    bbox_to_anchor=(1.6, 1.0),
    ncol=3,
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
    handletextpad=0.6,
    columnspacing=0.8,
    prop={"weight": "bold", "size": font_size},
).set_zorder(100)

axes[0].set_ylabel("Normalized Runtime", fontsize=font_size)
# axes[0].set_yticklabels(yticks, fontsize=font_size)
# axes[1].set_yticklabels([])
file_name = f"{SAVE_PTH}/read_io.pdf"
plt.savefig(file_name, bbox_inches="tight", dpi=300)
## print save
print(f"Save to {file_name}")
# %%
