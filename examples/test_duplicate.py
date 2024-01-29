import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import psutil
import time
from offgs.dataset import OffgsDataset
import os

import offgs


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--dataset", type=str, default="friendster")
parser.add_argument("--batchsize", type=int, default=1024)
parser.add_argument("--fanout", type=str, default="10,10,10")
parser.add_argument("--store-path", default="/nvme1n1/offgs_dataset")
parser.add_argument("--feat-cache-size", type=int, default=10000000000)
parser.add_argument("--log", type=str, default="logs/pack_decompose.csv")
parser.add_argument("--ratio", type=float, default=1.0)
args = parser.parse_args()
print(args)

# --- load data --- #
dataset_path = f"{args.store_path}/{args.dataset}-offgs"
dataset = OffgsDataset(dataset_path)

process = psutil.Process(os.getpid())
mem = process.memory_info().rss / (1024 * 1024 * 1024)
print("Memory Occupation:", mem, "GB")

device = torch.device(f"cuda:{args.device}")

dataset_path = f"{args.store_path}/{args.dataset}-offgs"
output_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
output_dir = os.path.join(args.store_path, output_dir)
aux_dir = f"{output_dir}/cache-size-{args.feat_cache_size}"

features = dataset.mmap_features

node_counts = torch.load(f"{output_dir}/node_counts.pt")
sorted_idx = torch.load(f"{output_dir}/meta_node_popularity.pt").cpu()

feat_load_time, nid_load_time, difference_time, save_time = 0, 0, 0, 0

start = time.time()

train_idx = (
    dataset.split_idx["train"]
    if args.ratio == 1.0
    else torch.load(f"{dataset_path}/train_idx_{args.ratio}.pt")
)
num_batches = (train_idx.numel() + args.batchsize - 1) // args.batchsize

tic = time.time()
table_size = 4 * dataset.num_nodes
num_entries = min(
    (args.feat_cache_size - table_size) // (4 * features.shape[1]),
    dataset.num_nodes,
)
# Maximum 400GB cache size for int32 (aligned with Ginex)
if num_entries > torch.iinfo(torch.int32).max:
    raise ValueError
cache_indices = sorted_idx[:num_entries]
key, value = torch.ops.offgs._CAPI_BuildHashMap(cache_indices.to(device))
cache_init_time = time.time() - tic
print(
    f"#Cached Entries: {num_entries} / {dataset.num_nodes}",
    f"Ratio: {num_entries / dataset.num_nodes}",
    f"Access Cache Ratio: {node_counts[cache_indices].sum().item() / node_counts.sum().item()}",
)

sampled_batch = torch.randperm(num_batches)[:2].tolist()

overlap_rate = []
for bid in sampled_batch:
    overlap_rate.append([])
    input_nodes: torch.Tensor = torch.load(f"{output_dir}/in-nid-{bid}.pt").to(device)
    bid_cold, _, _, _ = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)

    for i in trange(num_batches, ncols=100):
        if i == bid:
            continue
        input_nodes: torch.Tensor = torch.load(f"{output_dir}/in-nid-{i}.pt").to(device)

        (
            cold_nodes,
            rev_cold_idx,
            hot_nodes,
            rev_hot_idx,
        ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)

        overlap_rate[-1].append(
            np.intersect1d(bid_cold.cpu(), cold_nodes.cpu()).shape[0]
            / bid_cold.shape[0]
        )


for i, bid in enumerate(sampled_batch):
    plt.plot(
        range(num_batches - 1),
        overlap_rate[i],
        label=f"Batch {bid}",
    )
plt.xlabel("Minibatch ID")
plt.ylabel("Intersection / Size of Batch")
plt.title(f"{args.dataset} Intersection")
plt.grid()
plt.legend()
plt.savefig(f"figs/{args.dataset}-{args.feat_cache_size}-intersection.png")
plt.close()
