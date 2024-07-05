
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

file_path = '/home/ubuntu/offgs_plot/offgs_plot/offgs_data/feat_packing.csv'

with open(file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    data = list(reader)


config = []
statistics = []
all_configs = []
all_statistics = []
all_normalized_statistics = []
dataset=['Friendster','IGB-HOM','Ogbn-papers100M','MAG240M']
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
        tmp_statistics=statistics
        tmp_statistics[0]=round(tmp_statistics[0]/tmp_statistics[3],2)
        tmp_statistics[1]=round(tmp_statistics[1]/tmp_statistics[3],2)
        tmp_statistics[2]=round(tmp_statistics[2]/tmp_statistics[3],2)
        tmp_statistics[3]=round(tmp_statistics[3]/tmp_statistics[3],2)
        # normalized_statistics=[round(float(stat)/float(statistics[-3]),2) for stat in statistics[:-1]]
        all_normalized_statistics.append(tmp_statistics)
print('finish preprocessing')      
print(all_normalized_statistics)
fig, axes = plt.subplots()
fig.set_size_inches(8, 2.5)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 4, 4
group = 4
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "lightseagreen", "tomato", "slategray", "silver"]
hatches = ["/////", "\\\\\\\\\\"]
labels = ['Primitive Read','Optimized Read','Primitive Preprocessing','Optimized Preprocessing']

x_labels=dataset

yticks=np.arange(0, 10, 1)


for j in range(n):
    ##TODO add label
    # plot_label= [data[j] for data in all_new_normalized_statistics[(line_num-1)*6+i*group:(line_num-1)*6+i*group+3]]

    container=axes.bar(
        exit_idx_x + j * width,
        [data[j] for data in all_normalized_statistics],
        width=width * 0.8,
        color="white",
        edgecolor=edgecolors[j],
        hatch=hatches[j % 2],
        linewidth=1.0,
        label=labels[j],
        zorder=10,
    )
    # axes[i].bar_label(container, plot_label, fontsize=5, zorder=200, fontweight="bold")

axes.set_yticks(yticks)
# axes.set_ylim(0, 10)
axes.set_yticklabels(yticks, fontsize=12)
axes.set_xticks((exit_idx_x + width + exit_idx_x) / 2)
axes.set_xticklabels(x_labels, fontsize=12)
axes.set_xlabel('Dataset Name', fontsize=12)


axes.legend(
    bbox_to_anchor=(1, 0.99),
    ncol=4,
    loc="lower right",
    # fontsize=10,
    # markerscale=3,
    labelspacing=0.1,
    edgecolor="black",
    facecolor="white",
    framealpha=1,
    shadow=False,
    fancybox=False,
    handlelength=0.8,
    handletextpad=0.6,
    columnspacing=0.8,
    prop={"weight": "bold", "size": 10},
).set_zorder(100)

axes.set_ylabel("Normalized Runtime", fontsize=10)
axes.set_yticklabels(yticks, fontsize=12)
file_name=f"{SAVE_PTH}/preprocess.pdf"
plt.savefig(file_name, bbox_inches="tight", dpi=300)
## print save
print(f"Save to {file_name}")
#%%