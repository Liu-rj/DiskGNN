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
parser.add_argument("--feat-cache-size", type=float, default=1e10)
parser.add_argument("--log", type=str, default="logs/pack_decompose.csv")
parser.add_argument("--ratio", type=float, default=1.0)
args = parser.parse_args()
print(args)
args.feat_cache_size = int(args.feat_cache_size)

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

src, dst = [], []
total_cold_nodes = 0
for i in trange(num_batches, ncols=100):
    tic = time.time()
    input_nodes: torch.Tensor = torch.load(f"{output_dir}/in-nid-{i}.pt").to(device)
    nid_load_time += time.time() - tic

    tic = time.time()
    (
        cold_nodes,
        rev_cold_idx,
        hot_nodes,
        rev_hot_idx,
    ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)
    total_cold_nodes += cold_nodes.numel()
    torch.cuda.synchronize()
    difference_time += time.time() - tic

    src += cold_nodes.cpu().tolist()
    dst += [i] * cold_nodes.numel()

tensor_src = torch.tensor(src, dtype=torch.int64).pin_memory()
tensor_dst = torch.tensor(dst, dtype=torch.int64).pin_memory()
num_src = dataset.num_nodes
num_dst = num_batches

h = torch.randperm(dataset.num_nodes, pin_memory=True)
h_res, _ = torch.ops.offgs._CAPI_SegmentedMinHash(
    tensor_dst, tensor_src, h, num_dst, num_src, False
)

bucket = defaultdict(list)
for i, ele in enumerate(h_res.cpu()):
    bucket[ele.item()].append(i)

print(
    "Before Second MinHash:",
    f"Number of Buckets: {len(bucket)}",
    f"Max Bucket Size: {max([len(v) for v in bucket.values()])}",
    f"Min Bucket Size: {min([len(v) for v in bucket.values()])}",
)

keys = list(bucket.keys())
for k in keys:
    if len(bucket[k]) > 10:
        values = bucket.pop(k)
        src, dst = [], []
        for i, v in enumerate(values):
            input_nodes = torch.load(f"{output_dir}/in-nid-{v}.pt").to(device)
            (
                cold_nodes,
                rev_cold_idx,
                hot_nodes,
                rev_hot_idx,
            ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)

            src += cold_nodes.cpu().tolist()
            dst += [i] * cold_nodes.numel()

        tensor_src = torch.tensor(src, dtype=torch.int64).pin_memory()
        tensor_dst = torch.tensor(dst, dtype=torch.int64).pin_memory()
        num_src = dataset.num_nodes
        num_dst = len(values)

        h = torch.randperm(dataset.num_nodes, pin_memory=True)
        h_res, _ = torch.ops.offgs._CAPI_SegmentedMinHash(
            tensor_dst, tensor_src, h, num_dst, num_src, False
        )

        for i, ele in enumerate(h_res.cpu()):
            bucket[k * dataset.num_nodes + ele.item()].append(values[i])

print(
    "After Second MinHash:",
    f"Number of Buckets: {len(bucket)}",
    f"Max Bucket Size: {max([len(v) for v in bucket.values()])}",
    f"Min Bucket Size: {min([len(v) for v in bucket.values()])}",
)

reduction = 0
for k, v in tqdm(bucket.items(), ncols=100):
    assert len(v) > 0
    for idx, i in enumerate(v):
        input_nodes: torch.Tensor = torch.load(f"{output_dir}/in-nid-{i}.pt")
        if idx == 0:
            intersection = input_nodes.numpy()
        else:
            intersection = np.intersect1d(intersection, input_nodes.numpy())
    reduction += intersection.shape[0] * (len(v) - 1)

# h_res_list = h_res.cpu().tolist()
# bucket_intersect = {}
# for i in trange(num_batches, ncols=100):
#     input_nodes: torch.Tensor = torch.load(f"{output_dir}/in-nid-{i}.pt")
#     bucket_id = h_res_list[i]
#     assert len(bucket[bucket_id]) > 0
#     if bucket_id not in bucket_intersect:
#         bucket_intersect[bucket_id] = input_nodes.numpy()
#     else:
#         bucket_intersect[bucket_id] = np.intersect1d(
#             bucket_intersect[bucket_id], input_nodes.numpy()
#         )

# reduction = 0
# for k, v in bucket_intersect.items():
#     assert len(bucket[k]) > 0
#     reduction += v.shape[0] * (len(bucket[k]) - 1)

print(f"Reduction Ratio: {reduction / total_cold_nodes}")
