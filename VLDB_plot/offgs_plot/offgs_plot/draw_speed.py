#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /livesr.py
# \brief:
# Author: raphael hao
import sys
#%%
sys.path.append('/home/ubuntu/offgs_plot/offgs_plot')

from script.settings import DATA_PATH, FIGURE_PATH
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True

#%%
import numpy as np

livesr_subnet = np.genfromtxt(
    DATA_PATH / "livesr/e2e_subnet.csv",
    delimiter=",",
    dtype=(int, int, float, float, float),
)
livesr_channel = np.genfromtxt(
    DATA_PATH / "livesr/e2e_channel.csv",
    delimiter=",",
    dtype=(int, int, float, float, float),
)
livesr_e2e = [livesr_subnet, livesr_channel]
#%%

fig, axes = plt.subplots(1, 2)
fig.set_size_inches(6, 1.5)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 3, 3
group = 4
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "lightseagreen", "tomato", "slategray", "silver"]
hatches = ["/////", "\\\\\\\\\\"]
labels = ["BRT", "BRT+VF", "BRT+HF"]
x_labels = ["Number of branches", "Number of channels"]
for i in range(2):
    for j in range(n):
        axes[i].bar(
            exit_idx_x + j * width,
            livesr_e2e[i][:][f"f{j + 2}"],
            width=width * 0.8,
            color="white",
            edgecolor=edgecolors[j],
            hatch=hatches[j % 2],
            linewidth=1.0,
            label=labels[j],
            zorder=10,
        )
    axes[i].set_yticks([0, 40, 80, 120])
    axes[i].tick_params(axis='y', labelsize=12)  # 设置 y 轴刻度标签的字体大小

    axes[i].set_xticks([1.6, 4.6, 7.6, 10.6])
    axes[i].set_xticklabels(livesr_e2e[i][:][f"f{i}"], fontsize=12)
    axes[i].set_xlabel(x_labels[i], fontsize=14)
axes[0].legend(
    bbox_to_anchor=(1.5, 0.98),
    ncol=3,
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
axes[0].set_ylabel("Time (\\textit{ms})", fontsize=14)
axes[0].set_yticklabels([0, 40, 80, 120], fontsize=12)
axes[1].set_yticklabels([])
plt.savefig(FIGURE_PATH / "livesr_e2e.pdf", bbox_inches="tight", dpi=300)
#%%
