import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np

import pandas as pd
from io import StringIO

# CSV data as string
csv_data = """
dataset,fanout,batchsize,10% cache,15% cache,20% cache,30% cache,50% cache,70% cache
Friendster,"10,15,20",1024,5.397,4.418,2.965,1.614,0.561,0.164
IGB-HOM,"10,15,20",1024,10.487,4.927,1.927,0.149,0.02,0.02
"""

# Using pandas to read this data from a string
data = pd.read_csv(StringIO(csv_data))

# Extract cache columns names
cache_columns = ['10% cache', '15% cache', '20% cache', '30% cache', '50% cache', '70% cache']

# Extract datasets names
datasets = data['dataset']
print(datasets)
# Extract numerical data for caches
cache_data = data[cache_columns].values
cache_columns = ['10\%', '15\%', '20\%', '30\%', '50\%', '70\%']

# Show extracted information
print("Cache Columns:", cache_columns)
print("Datasets:", datasets.tolist())
print("Cache Data:\n", cache_data)

# Plot settings
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True
yticks=np.arange(0, 12, 1)
# Create a new figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 5))

# Line and bar plot
colors = ['blue', 'green']  # Line colors
bar_colors = ['lightblue', 'lightgreen']  # Bar colors

# X positions of cache percentages, ensuring bars don't overlap
x_positions = np.arange(len(cache_columns))  # [0, 1, 2, 3, 4, 5]
hatches = ["", "///","|||","|||","---","...","||","xx","oo",'..']
    
labels = ["Our", "MariusGNN", "Ginex","Ginex+Sample"]

for i, (label, dataset) in enumerate(zip(datasets, cache_data)):
    # Plot line
    ax.plot(x_positions + i * 0.35 - 0.05, dataset, label=label, color=colors[i], linestyle='--')

    # Plot bars
    container=ax.bar(x_positions + i*0.35 - 0.05, dataset, width=0.3, 
           
           color=bar_colors[i], align='center')
    ax.bar_label(container, dataset, fontsize=9, zorder=200, fontweight="bold")


# Setting labels and title
val_limit=yticks[-1]

ax.set_yticks(yticks)
ax.set_ylim(0, val_limit)
ax.set_xticks(x_positions+0.13)
ax.set_xticklabels(cache_columns, fontsize=12)
ax.set_ylabel('Disk Size Blow up',fontsize=12)
ax.set_xlabel('Cache Sizes',fontsize=14)
ax.legend(title="Dataset",fontsize=25,
          
           edgecolor="black",
            facecolor="white",
            
            framealpha=1,
            shadow=False,
            fancybox=False,
            handlelength=1.5,
            handletextpad=0.6,
            columnspacing=0.8,
            prop={"weight": "bold", "size": 15},
          
          )
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
# plt.tight_layout()
SAVE_PTH='/home/ubuntu/offgs_plot/offgs_plot/figures'

file_name=f"{SAVE_PTH}/cache_disk.pdf"
plt.savefig(file_name, bbox_inches="tight", dpi=300)
## print save
print(f"Save to {file_name}")
#%%