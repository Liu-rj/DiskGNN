
import numpy as np
import matplotlib.pyplot as plt
import csv
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True
SAVE_PTH='/home/ubuntu/offgs_plot/offgs_plot/figures'


import csv

file_path = '/home/ubuntu/offgs_plot/offgs_plot/offgs_data/disk_blowup.csv'

with open(file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    data = list(reader)


config = []
statistics = []
all_configs = []
all_statistics = []
all_normalized_statistics = []
dataset=['Friendster','IGB-HOM']
for i, row in enumerate(data):
    if len(row)==0:
        continue
    if i==0:
        all_configs.append(row[1:])
    else:
        statistics=row[1:]
        ## convert to float
        
        statistics=[float(stat) for stat in statistics]
        all_statistics.append(statistics)
        normalized_statistics=[round(float(stat)/float(statistics[-3]),2) for stat in statistics[:-1]]
        ## put the last one into the first
        normalized_statistics.insert(0,normalized_statistics.pop())
        ## revert the list
        normalized_statistics=normalized_statistics[::-1]
        all_normalized_statistics.append(normalized_statistics)
print('finish preprocessing')      
print(all_normalized_statistics)
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(6, 1.9)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 5, 5
group = 1
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "lightseagreen", "tomato", "slategray", "silver"]
hatches = [ "|||","///","\\\\\\","---",""]
hatches=hatches[::-1]
labels = ["Ginex","3x", "5x", "7x","Unlimited"]
labels=labels[::-1]

x_labels=dataset

yticks=np.arange(0, 11, 1)

for i in range(2):
    val_limit=yticks[-1]
    axes[i].set_yticks(yticks)
    axes[i].set_ylim(0, val_limit)

    axes[i].tick_params(axis='y', labelsize=10)  # 设置 y 轴刻度标签的字体大小
    axes[i].set_xticklabels([])

    axes[i].set_xticks([])
    # axes[i].set_xticklabels([], fontsize=8)
    axes[i].set_xlabel(x_labels[i], fontsize=12)
    for j in range(n):
        ##TODO add label
        # plot_label= [data[j] for data in all_new_normalized_statistics[(line_num-1)*6+i*group:(line_num-1)*6+i*group+3]]
        plot_label=[all_normalized_statistics[i][j]]
        container=axes[i].bar(
            exit_idx_x + j * width,
            all_normalized_statistics[i][j],
            width=width * 0.8,
            color="white",
            edgecolor='k',
            hatch=hatches[j ],
            linewidth=1.0,
            label=labels[j],
            zorder=10,
        )
        axes[i].bar_label(container, plot_label, fontsize=8, zorder=200, fontweight="bold")


axes[0].legend(
    bbox_to_anchor=(1.71, 0.99),
    ncol=5,
    loc="lower right",
    # fontsize=10,
    # markerscale=3,
    labelspacing=0.1,
    edgecolor="black",
    facecolor="white",
    framealpha=1,
    shadow=False,
    fancybox=False,
    handlelength=1.3,
    handletextpad=0.6,
    columnspacing=0.8,
    prop={"weight": "bold", "size": 10},
).set_zorder(100)

axes[0].set_ylabel("Normalized Runtime", fontsize=10)
axes[0].set_yticklabels(yticks, fontsize=12)
axes[1].set_yticklabels([])
file_name=f"{SAVE_PTH}/vary_disk.pdf"
plt.savefig(file_name, bbox_inches="tight", dpi=300)
## print save
print(f"Save to {file_name}")
#%%