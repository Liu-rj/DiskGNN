import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from load_graph import *
import psutil
import time
import json
from offgs.dataset import OffgsDataset
import csv
from collections import defaultdict
from joblib import Parallel, delayed

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

table_size = 4 * dataset.num_nodes
num_entries = min(
    (args.feat_cache_size - table_size) // (4 * features.shape[1]),
    dataset.num_nodes,
)
# Maximum 400GB cache size for int32 (aligned with Ginex)
if num_entries > torch.iinfo(torch.int32).max:
    raise ValueError
cache_indices = sorted_idx[:num_entries].to(device)
key, value = torch.ops.offgs._CAPI_BuildHashMap(cache_indices)


def simulate(disk_cache_num):
    print(f"Disk Cache Num: {disk_cache_num}")
    assert disk_cache_num + num_entries < dataset.num_nodes

    disk_cache = sorted_idx[num_entries : num_entries + disk_cache_num].to(device)
    disk_table = torch.full((dataset.num_nodes,), -1, dtype=torch.int32, device=device)
    disk_table[disk_cache] = torch.arange(disk_cache_num, dtype=torch.int32).to(device)
    disk_key, disk_value = torch.ops.offgs._CAPI_BuildHashMap(disk_cache)
    print(
        f"#Cached Entries: {num_entries} / {dataset.num_nodes}",
        f"Ratio: {num_entries / dataset.num_nodes}",
        f"Access Cache Ratio: {node_counts[cache_indices].sum().item() / node_counts.sum().item()}",
    )
    print(
        f"Disk Cache Entries: {disk_cache_num} / {dataset.num_nodes}",
        f"Ratio: {disk_cache_num / dataset.num_nodes}",
        f"Access Cache Ratio: {node_counts[disk_cache].sum().item() / node_counts.sum().item()}",
    )

    original_cold_nodes = 0
    total_cold_nodes = 0
    io_traffic = 0
    src, dst = [], []
    for i in trange(num_batches, ncols=100):
        input_nodes: torch.Tensor = torch.load(f"{output_dir}/in-nid-{i}.pt").to(device)

        (
            cold_nodes,
            rev_cold_idx,
            hot_nodes,
            rev_hot_idx,
        ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)
        original_cold_nodes += cold_nodes.numel()

        (
            disk_cold,
            _,
            disk_hot,
            _,
        ) = torch.ops.offgs._CAPI_QueryHashMap(cold_nodes, disk_key, disk_value)

        disk_loc = disk_table[disk_hot]
        assert (disk_loc != -1).all()
        src += disk_loc.cpu().tolist()
        dst += [i] * disk_loc.numel()

        page_id = disk_loc // 8
        unique_page_id = torch.unique(page_id)

        io_traffic += disk_cold.numel() + unique_page_id.numel() * 8
        total_cold_nodes += disk_cold.numel()

    disk_size = total_cold_nodes + disk_cache_num
    print("+++++++++++++++++++++++++Before Reorder+++++++++++++++++++++++++")
    print(f"Oringinal Cold Nodes: {original_cold_nodes}")
    print(
        f"Packed Nodes: {total_cold_nodes}, Ratio: {total_cold_nodes / original_cold_nodes}"
    )
    print(f"IO Traffic: {io_traffic}, Ratio: {io_traffic / original_cold_nodes}")
    print(f"Disk Size: {disk_size}, Ratio: {disk_size / original_cold_nodes}")
    record_before = (io_traffic / original_cold_nodes, disk_size / original_cold_nodes)

    tensor_src = torch.tensor(src, dtype=torch.int64).pin_memory()
    tensor_dst = torch.tensor(dst, dtype=torch.int64).pin_memory()
    num_src = disk_cache_num
    num_dst = num_batches
    h = torch.randperm(num_batches, device=device)
    h_res, degree, counts = torch.ops.offgs._CAPI_SegmentedMinHash(
        tensor_src, tensor_dst, h, num_src, num_dst
    )
    # node_info = zip(h_res.cpu().tolist(), degree.cpu().tolist())
    h_res_cpu, degree_cpu = h_res.cpu().tolist(), degree.cpu().tolist()
    indices = sorted(range(h_res.numel()), key=lambda k: (h_res_cpu[k], -degree_cpu[k]))
    # h_res_sorted, indices = torch.sort(h_res)
    disk_table[disk_cache[indices]] = torch.arange(
        disk_cache_num, dtype=torch.int32, device=device
    )

    original_cold_nodes = 0
    total_cold_nodes = 0
    io_traffic = 0
    for i in trange(num_batches, ncols=100):
        input_nodes: torch.Tensor = torch.load(f"{output_dir}/in-nid-{i}.pt").to(device)

        (
            cold_nodes,
            rev_cold_idx,
            hot_nodes,
            rev_hot_idx,
        ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)
        original_cold_nodes += cold_nodes.numel()

        (
            disk_cold,
            _,
            disk_hot,
            _,
        ) = torch.ops.offgs._CAPI_QueryHashMap(cold_nodes, disk_key, disk_value)

        disk_loc = disk_table[disk_hot]
        assert (disk_loc != -1).all()
        page_id = disk_loc // 8
        unique_page_id = torch.unique(page_id)

        io_traffic += disk_cold.numel() + unique_page_id.numel() * 8
        total_cold_nodes += disk_cold.numel()

    disk_size = total_cold_nodes + disk_cache_num
    print("+++++++++++++++++++++++++After Reorder+++++++++++++++++++++++++")
    print(f"Oringinal Cold Nodes: {original_cold_nodes}")
    print(
        f"Packed Nodes: {total_cold_nodes}, Ratio: {total_cold_nodes / original_cold_nodes}"
    )
    print(f"IO Traffic: {io_traffic}, Ratio: {io_traffic / original_cold_nodes}")
    print(f"Disk Size: {disk_size}, Ratio: {disk_size / original_cold_nodes}")
    return original_cold_nodes, total_cold_nodes, io_traffic, disk_size, record_before


io_ratio, disk_ratio, io_ratio_before = [], [], []
# disk_cache_num_list = [int(1e6)]
disk_cache_num_list = np.arange(0, 1e7, 1e6, dtype=np.int64)
for disk_cache_num in disk_cache_num_list:
    (
        original_cold_nodes,
        total_cold_nodes,
        io_traffic,
        disk_size,
        record_before,
    ) = simulate(disk_cache_num)
    io_ratio.append(io_traffic / original_cold_nodes)
    disk_ratio.append(disk_size / original_cold_nodes)
    io_ratio_before.append(record_before[0])

plt.plot(disk_cache_num_list, io_ratio, label="IO Traffic")
plt.plot(disk_cache_num_list, io_ratio_before, label="IO Traffic w/o Reorder")
plt.plot(disk_cache_num_list, disk_ratio, label="Disk Size")
plt.xlabel("Disk Cache Node Num")
plt.ylabel("Ratio")
plt.title(f"{args.dataset} Disk Cache v.s. IO and Storage")
plt.yticks(np.arange(0, max(np.max(io_ratio), np.max(disk_ratio)), 0.5))
plt.grid()
plt.legend()
plt.savefig(f"{args.dataset}-ratio.png")
plt.close()
