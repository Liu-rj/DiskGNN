import dgl
import numpy as np
import matplotlib.pyplot as plt
import argparse
from load_graph import *
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="ogbn-products",
    help="which dataset to load for training",
)
args = parser.parse_args()
print(args)

name = args.dataset
if args.dataset.startswith("ogbn"):
    dataset = load_ogb(name, "/home/ubuntu/dataset")
elif args.dataset == "friendster":
    dataset = load_dglgraph("/home/ubuntu/dataset/friendster/friendster.bin")
g, features, labels, n_classes, splitted_idx = dataset

degrees = g.in_degrees().detach().numpy()
# counts, bins = np.histogram(degrees)
# plt.stairs(counts, bins)
a = sorted(degrees, reverse=True)
font_size = 10
plt.plot(np.arange(len(a)) / g.num_nodes(), a)
plt.xlabel(
    "proportion of nodes (sorted in descending order by in-degrees)",
    fontsize=font_size,
    fontweight="bold",
)
plt.ylabel("in-degrees", fontsize=font_size, fontweight="bold")
plt.title(name + "-degree", fontsize=font_size, fontweight="bold")
plt.grid(linestyle="-.")
plt.savefig(name + "-degree.pdf", bbox_inches="tight")
plt.clf()

degrees = g.in_degrees().detach().numpy()
# counts, bins = np.histogram(degrees)
# plt.stairs(counts, bins)
a = sorted(degrees, reverse=True)
b = np.cumsum(a) / g.num_edges()
font_size = 10
plt.plot(np.arange(b.shape[0]) / g.num_nodes(), b)
plt.xlabel(
    "proportion of nodes (sorted in descending order by in-degrees)",
    fontsize=font_size,
    fontweight="bold",
)
plt.ylabel(
    "proportion of accumulated in-degrees", fontsize=font_size, fontweight="bold"
)
plt.title(name + "-accumulated-degree", fontsize=font_size, fontweight="bold")
plt.grid(linestyle="-.")
plt.savefig(name + "-degree-distribution.pdf", bbox_inches="tight")
plt.clf()


degrees = g.in_degrees()
idx = torch.argsort(degrees)
num_neighbors = []
stamps = np.arange(0, 1.1, 0.1)
for stamp in tqdm(stamps):
    u = g.in_edges(idx[: int(idx.shape[0] * stamp)])[0]
    neighbors = torch.unique(u)
    num_neighbors.append(neighbors.numel() / g.num_nodes())
font_size = 10
plt.plot(stamps, num_neighbors)
plt.xlabel(
    "proportion of nodes (sorted in ascending order by in-degrees)",
    fontsize=font_size,
    fontweight="bold",
)
plt.ylabel(
    "proportion of neighbors over all nodes", fontsize=font_size, fontweight="bold"
)
plt.title(name + "-neighbor-distribution", fontsize=font_size, fontweight="bold")
plt.grid(linestyle="-.")
plt.savefig(name + "-neighbor-distribution.pdf", bbox_inches="tight")
