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

node_counts = torch.load(f"{output_dir}/node_counts.pt").cpu()
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
num_entries = max(num_entries, 0)
# Maximum 400GB cache size for int32 (aligned with Ginex)
if num_entries > torch.iinfo(torch.int32).max:
    raise ValueError
cache_indices = sorted_idx[:num_entries]
key, value = torch.ops.offgs._CAPI_BuildHashMap(cache_indices.to(device))
print(
    f"#Cached Entries: {num_entries} / {dataset.num_nodes}",
    f"Ratio: {num_entries / dataset.num_nodes}",
    f"Access Cache Ratio: {node_counts[cache_indices].sum().item() / node_counts.sum().item()}",
)


def plot_page_num_ratio(bf_page_num_ratio, af_page_num_ratio):
    plt.plot(range(num_batches), bf_page_num_ratio, label="Before Reorder")
    plt.plot(range(num_batches), af_page_num_ratio, label="After Reorder")
    plt.xlabel("Minibatch ID")
    plt.ylabel("Number of Pages (before unique / after unique)")
    plt.title(f"{args.dataset} Page Num")
    plt.grid()
    plt.legend()
    plt.savefig(f"{args.dataset}-{args.feat_cache_size}-page-num-ratio.png")
    plt.close()


def plot_page_num(page_num):
    plt.plot(range(num_batches), page_num, label="Page Num")
    plt.xlabel("Minibatch ID")
    plt.ylabel("Number of Pages")
    plt.title(f"{args.dataset} Page Num")
    plt.grid()
    plt.legend()
    plt.savefig(f"{args.dataset}-{args.feat_cache_size}-page-num.png")
    plt.close()


def simulate(disk_cache_num, segment_size):
    assert disk_cache_num + num_entries < dataset.num_nodes

    num_segments = (num_batches + segment_size - 1) // segment_size
    print(f"Num Segments: {num_segments}, Segment Size: {segment_size}")

    # bf_page_num_ratio = []
    # af_page_num_ratio = []
    # page_num = []
    original_cold_nodes = [0, 0]
    total_cold_nodes = [0, 0]
    io_traffic = [0, 0]
    for segid in trange(num_segments, ncols=100):
        all_pid = []

        startid = segid * segment_size
        endid = min((segid + 1) * segment_size, num_batches)

        popularity = torch.zeros(dataset.num_nodes, dtype=torch.int32, device=device)
        for bid in range(startid, endid):
            input_nodes = torch.load(f"{output_dir}/in-nid-{bid}.pt").to(device)
            (
                cold_nodes,
                _,
                _,
                _,
            ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)
            popularity[cold_nodes] += 1

        seg_sorted_idx = torch.argsort(popularity, descending=True)
        disk_cache = seg_sorted_idx[:disk_cache_num]
        disk_table = torch.full(
            (dataset.num_nodes,), -1, dtype=torch.int32, device=device
        )
        disk_table[disk_cache] = torch.arange(
            disk_cache_num, dtype=torch.int32, device=device
        )
        disk_key, disk_value = torch.ops.offgs._CAPI_BuildHashMap(disk_cache)
        print(
            f"\nDisk Cache Entries: {disk_cache_num} / {dataset.num_nodes}",
            f"Ratio: {disk_cache_num / dataset.num_nodes:.3f}",
            f"Ratio In Seg: {disk_cache_num / (popularity > 0).sum().numel():.3f}",
            f"Access Cache Ratio: {popularity[disk_cache].sum().item() / popularity.sum().item():.3f}",
        )

        src, dst = [], []
        for bid in range(startid, endid):
            input_nodes = torch.load(f"{output_dir}/in-nid-{bid}.pt").to(device)

            (
                cold_nodes,
                _,
                _,
                _,
            ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)
            original_cold_nodes[0] += cold_nodes.numel()

            (
                disk_cold,
                _,
                disk_hot,
                _,
            ) = torch.ops.offgs._CAPI_QueryHashMap(cold_nodes, disk_key, disk_value)

            disk_loc = disk_table[disk_hot]
            assert (disk_loc != -1).all()
            src += disk_loc.cpu().tolist()
            dst += [bid] * disk_loc.numel()

            page_id = disk_loc // 8
            unique_page_id = torch.unique(page_id)
            # bf_page_num_ratio.append(page_id.numel() / unique_page_id.numel())

            io_traffic[0] += disk_cold.numel() + unique_page_id.numel() * 8
            total_cold_nodes[0] += disk_cold.numel()

        tensor_src = torch.tensor(src, dtype=torch.int64).pin_memory()
        tensor_dst = torch.tensor(dst, dtype=torch.int64).pin_memory()
        num_src = disk_cache_num
        num_dst = num_batches
        h1 = torch.randperm(num_batches, device=device)
        h1_res, degree = torch.ops.offgs._CAPI_SegmentedMinHash(
            tensor_src, tensor_dst, h1, num_src, num_dst, False
        )
        # h_res = h1_res
        h2 = torch.randperm(num_batches, device=device)
        h2_res, _ = torch.ops.offgs._CAPI_SegmentedMinHash(
            tensor_src, tensor_dst, h2, num_src, num_dst, False
        )
        h_res = h1_res * num_batches + h2_res
        # h3 = torch.randperm(num_batches, device=device)
        # h3_res, _, _ = torch.ops.offgs._CAPI_SegmentedMinHash(
        #     tensor_src, tensor_dst, h3, num_src, num_dst
        # )
        # h_res = h1_res * num_batches * num_batches + h2_res * num_batches + h3_res

        h_res_cpu, degree_cpu = h_res.cpu().tolist(), degree.cpu().tolist()
        indices = sorted(
            range(h_res.numel()), key=lambda k: (h_res_cpu[k], -degree_cpu[k])
        )
        disk_table[disk_cache[indices]] = torch.arange(
            disk_cache_num, dtype=torch.int32, device=device
        )

        for bid in range(startid, endid):
            input_nodes = torch.load(f"{output_dir}/in-nid-{bid}.pt").to(device)

            (
                cold_nodes,
                _,
                _,
                _,
            ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)
            original_cold_nodes[1] += cold_nodes.numel()

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
            # page_num.append(unique_page_id.numel())
            # af_page_num_ratio.append(page_id.numel() / unique_page_id.numel())
            all_pid.append(unique_page_id.cpu().numpy())

            io_traffic[1] += disk_cold.numel() + unique_page_id.numel() * 8
            total_cold_nodes[1] += disk_cold.numel()

        sampled_batch = torch.randperm(segment_size)[:2].tolist()

        overlap_rate = []
        for bid in sampled_batch:
            overlap_rate.append([])
            page_id = all_pid[bid]
            for i, pid in enumerate(all_pid):
                if i == bid:
                    continue
                overlap_rate[-1].append(
                    np.intersect1d(page_id, pid).shape[0] / page_id.shape[0]
                )

        for i, bid in enumerate(sampled_batch):
            plt.plot(
                range(segment_size - 1),
                overlap_rate[i],
                label=f"Batch {bid}",
            )
        plt.xlabel("Minibatch ID")
        plt.ylabel("Intersection / #Page Reads of Batch")
        plt.title(f"{args.dataset} Page Intersection")
        plt.grid()
        plt.legend()
        plt.savefig(
            f"figs/{args.dataset}-{args.feat_cache_size:g}-{disk_cache_num:g}-{segment_size}-page-intersection.png"
        )
        plt.close()

        exit()

    assert original_cold_nodes[0] == original_cold_nodes[1]

    disk_size = total_cold_nodes[0] + disk_cache_num * num_segments
    print(
        "+++++++++++++++++++++++++Before Reorder+++++++++++++++++++++++++\n",
        f"Oringinal Cold Nodes: {original_cold_nodes[0]}, blowup: {original_cold_nodes[0] / dataset.num_nodes}\n",
        f"Packed Nodes: {total_cold_nodes[0]}, Ratio: {total_cold_nodes[0] / original_cold_nodes[0]}\n",
        f"IO Traffic: {io_traffic[0]}, Ratio: {io_traffic[0] / original_cold_nodes[0]}\n",
        f"Disk Size: {disk_size}, Ratio: {disk_size / original_cold_nodes[0]}",
    )
    record_before = (
        io_traffic[0] / original_cold_nodes[0],
        disk_size / original_cold_nodes[0],
    )

    disk_size = total_cold_nodes[1] + disk_cache_num * num_segments
    print(
        "+++++++++++++++++++++++++After Reorder+++++++++++++++++++++++++\n",
        f"Oringinal Cold Nodes: {original_cold_nodes[1]}, blowup: {original_cold_nodes[1] / dataset.num_nodes}\n",
        f"Packed Nodes: {total_cold_nodes[1]}, Ratio: {total_cold_nodes[1] / original_cold_nodes[1]}\n",
        f"IO Traffic: {io_traffic[1]}, Ratio: {io_traffic[1] / original_cold_nodes[1]}\n",
        f"Disk Size: {disk_size}, Ratio: {disk_size / original_cold_nodes[1]}",
    )

    # plot_page_num_ratio(
    #     sorted(bf_page_num_ratio, reverse=True),
    #     sorted(af_page_num_ratio, reverse=True),
    # )
    return (
        original_cold_nodes[1],
        total_cold_nodes[1],
        io_traffic[1],
        disk_size,
        record_before,
    )


f = open("logs/disk_cache.csv", "a")
writer = csv.writer(f, lineterminator="\n")

io_ratio, disk_ratio, io_ratio_before = [], [], []
disk_cache_num_list = [int(6e6)]
segment_size_list = [400]
# if args.dataset == "friendster":
#     disk_cache_num_list = np.arange(0, 10e6 + 2e6, 2e6, dtype=np.int64)
#     segment_size_list = [100, 200, 300, 400]
# elif args.dataset == "igb-full":
#     if args.feat_cache_size == 3e10:
#         disk_cache_num_list = np.arange(0, 12e6 + 2e6, 2e6, dtype=np.int64)
#         segment_size_list = [500, 1000, 2000, 3000]
#     elif args.feat_cache_size == 1e10:
#         disk_cache_num_list = np.arange(0, 20e6 + 4e6, 4e6, dtype=np.int64)
#         segment_size_list = [100, 200, 400, 800]
for segment_size in segment_size_list:
    for disk_cache_num in disk_cache_num_list:
        (
            original_cold_nodes,
            total_cold_nodes,
            io_traffic,
            disk_size,
            record_before,
        ) = simulate(disk_cache_num, segment_size)
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            f"{args.feat_cache_size:g}",
            segment_size,
            f"{disk_cache_num:g}",
            round(record_before[0], 3),
            round(io_traffic / original_cold_nodes, 3),
            round(disk_size / original_cold_nodes, 3),
        ]
        writer.writerow(log_info)

f.close()
