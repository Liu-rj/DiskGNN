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
parser.add_argument("--feat-cache-size", type=float, default=1e10)
parser.add_argument("--disk-cache-num", type=float, default=1e6)
parser.add_argument("--segment-size", type=int, default=100)
parser.add_argument("--ratio", type=float, default=1.0)
args = parser.parse_args()
print(args)
args.feat_cache_size = int(args.feat_cache_size)
args.disk_cache_num = int(args.disk_cache_num)

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
num_entries = min(
    args.feat_cache_size // (4 * features.shape[1]),
    dataset.num_nodes,
)
cache_indices = sorted_idx[:num_entries]
key, value = torch.ops.offgs._CAPI_BuildHashMap(cache_indices.to(device))
cache_init_time = time.time() - tic
print(
    f"#Cached Entries: {num_entries} / {dataset.num_nodes}",
    f"Ratio: {num_entries / dataset.num_nodes}",
    f"Access Cache Ratio: {node_counts[cache_indices].sum().item() / node_counts.sum().item()}",
)
assert args.disk_cache_num + num_entries < dataset.num_nodes
num_segments = (num_batches + args.segment_size - 1) // args.segment_size
print(f"Num Segments: {num_segments}, Segment Size: {args.segment_size}")

sampled_batch = torch.randperm(num_batches)[:2].tolist()

overlap_rate = []
for bid in sampled_batch:
    segid = bid // args.segment_size
    startid = segid * args.segment_size
    endid = min((segid + 1) * args.segment_size, num_batches)

    popularity = torch.zeros(dataset.num_nodes, dtype=torch.int32, device=device)
    for i in trange(startid, endid, ncols=100):
        input_nodes = torch.load(f"{output_dir}/in-nid-{i}.pt").to(device)
        cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
        popularity[cold_nodes] += 1

    seg_sorted_idx = torch.argsort(popularity, descending=True)
    disk_cache = seg_sorted_idx[: args.disk_cache_num]
    disk_key, disk_value = torch.ops.offgs._CAPI_BuildHashMap(disk_cache)
    print(
        f"Disk Cache Entries: {args.disk_cache_num} / {dataset.num_nodes}",
        f"Ratio: {args.disk_cache_num / dataset.num_nodes}",
        f"Access Cache Ratio: {popularity[disk_cache].sum().item() / popularity.sum().item()}",
    )

    overlap_rate.append([])
    input_nodes: torch.Tensor = torch.load(f"{output_dir}/in-nid-{bid}.pt").to(device)
    bid_cold = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
    bid_disk = torch.ops.offgs._CAPI_QueryHashMap(bid_cold, disk_key, disk_value)[0]

    for segid in trange(num_segments, ncols=100):
        startid = segid * args.segment_size
        endid = min((segid + 1) * args.segment_size, num_batches)

        popularity = torch.zeros(dataset.num_nodes, dtype=torch.int32, device=device)
        for i in range(startid, endid):
            input_nodes = torch.load(f"{output_dir}/in-nid-{i}.pt").to(device)
            cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
            popularity[cold_nodes] += 1

        seg_sorted_idx = torch.argsort(popularity, descending=True)
        disk_cache = seg_sorted_idx[: args.disk_cache_num]
        disk_key, disk_value = torch.ops.offgs._CAPI_BuildHashMap(disk_cache)

        src, dst = [], []
        for i in range(startid, endid):
            if i == bid:
                continue
            input_nodes = torch.load(f"{output_dir}/in-nid-{i}.pt").to(device)
            cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
            disk_cold = torch.ops.offgs._CAPI_QueryHashMap(
                cold_nodes, disk_key, disk_value
            )[0]

            overlap_rate[-1].append(
                np.intersect1d(bid_disk.cpu(), disk_cold.cpu()).shape[0]
                / bid_disk.shape[0]
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
plt.savefig(f"figs/{args.dataset}-{args.feat_cache_size:g}-intersection.png")
plt.close()
