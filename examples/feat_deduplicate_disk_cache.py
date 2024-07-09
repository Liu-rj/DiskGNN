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

import offgs


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--dataset", type=str, default="friendster")
parser.add_argument("--batchsize", type=int, default=1024)
parser.add_argument("--fanout", type=str, default="10,10,10")
parser.add_argument("--store-path", default="/nvme1n1/offgs_dataset")
parser.add_argument("--feat-cache-size", type=float, default=1e10)
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
print(f"Train Node Ratio: {train_idx.numel() / dataset.num_nodes}")
num_batches = (train_idx.numel() + args.batchsize - 1) // args.batchsize

num_entries = min(
    args.feat_cache_size // (4 * features.shape[1]),
    dataset.num_nodes,
)
cache_indices = sorted_idx[:num_entries]
key, value = torch.ops.offgs._CAPI_BuildHashMap(cache_indices.to(device))
print(
    f"#Cached Entries: {num_entries} / {dataset.num_nodes}",
    f"Ratio: {num_entries / dataset.num_nodes}",
    f"Access Cache Ratio: {node_counts[cache_indices].sum().item() / node_counts.sum().item()}",
)

all_input_nodes = [
    torch.load(f"{output_dir}/in-nid-{bid}.pt") for bid in range(num_batches)
]
all_cold_nodes = [
    torch.ops.offgs._CAPI_QueryHashMap(input_nodes.to(device), key, value)[0].to("cpu")
    for input_nodes in all_input_nodes
]


def plot_page_num_ratio(page_num_ratio):
    plt.plot(range(num_batches), page_num_ratio)
    plt.xlabel("Minibatch ID")
    plt.ylabel("IO Traffic Reduction")
    # plt.title(f"{args.dataset} Page Num")
    plt.grid(axis="y", linestyle="--")
    plt.savefig(f"figs/{args.dataset}-{args.feat_cache_size}-page-num-ratio.pdf")
    plt.close("all")


def plot_page_num(page_num):
    plt.plot(range(num_batches), page_num, label="Page Num")
    plt.xlabel("Minibatch ID")
    plt.ylabel("Number of Pages")
    plt.title(f"{args.dataset} Page Num")
    plt.grid()
    plt.legend()
    plt.savefig(f"{args.dataset}-{args.feat_cache_size}-page-num.png")
    plt.close()


def save_page_num(pages_bf_reorder, pages_af_reorder, segment_size, disk_cache_num):
    minibatch_pages = [
        [pages_bf_reorder[i], pages_af_reorder[i]] for i in range(num_batches)
    ]
    f = open(
        f"logs/{args.dataset}-{args.feat_cache_size:g}-{segment_size}-{disk_cache_num:g}-disk-cache-io-decompose.csv",
        "w",
    )
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(minibatch_pages)


def simulate(disk_cache_num, segment_size):
    assert disk_cache_num + num_entries < dataset.num_nodes

    num_segments = (num_batches + segment_size - 1) // segment_size
    print(f"Num Segments: {num_segments}, Segment Size: {segment_size}")

    pages_bf_reorder = []
    pages_af_reorder = []
    original_cold_nodes = [0, 0]
    total_cold_nodes = [0, 0]
    io_traffic = [0, 0, 0]
    disk_size = 0
    cache_ratio, access_ratio, all_cache_num = [], [], []
    for segid in trange(num_segments, ncols=100):
        page_cache = np.array([], dtype=np.int64)

        startid = segid * segment_size
        endid = min((segid + 1) * segment_size, num_batches)

        popularity = torch.zeros(dataset.num_nodes, dtype=torch.int32, device=device)
        for bid in range(startid, endid):
            cold_nodes = all_cold_nodes[bid].to(device)
            popularity[cold_nodes] += 1

        seg_sorted_idx = torch.argsort(popularity, descending=True)
        cache_num = min(disk_cache_num, (popularity > 0).sum().item())
        disk_cache = seg_sorted_idx[:cache_num]
        disk_table = torch.full(
            (dataset.num_nodes,), -1, dtype=torch.int32, device=device
        )
        disk_table[disk_cache] = torch.arange(
            cache_num, dtype=torch.int32, device=device
        )
        disk_key, disk_value = torch.ops.offgs._CAPI_BuildHashMap(disk_cache)
        tqdm.write(
            f"\nDisk Cache Entries: {cache_num} / {dataset.num_nodes}\t"
            f"Ratio: {cache_num / dataset.num_nodes:.3f}\t"
            f"Ratio In Seg: {cache_num / (popularity > 0).sum().item():.3f}\t"
            f"Access Cache Ratio: {popularity[disk_cache].sum().item() / popularity.sum().item():.3f}"
        )
        all_cache_num.append(cache_num)
        cache_ratio.append(cache_num / (popularity > 0).sum().item())
        access_ratio.append(
            popularity[disk_cache].sum().item() / popularity.sum().item()
        )

        disk_size += cache_num

        src, dst = [], []
        for bid in range(startid, endid):
            cold_nodes = all_cold_nodes[bid].to(device)
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
            pages_bf_reorder.append(unique_page_id.numel())

            io_traffic[0] += disk_cold.numel() + unique_page_id.numel() * 8
            total_cold_nodes[0] += disk_cold.numel()

        tensor_src = torch.tensor(src, dtype=torch.int64).pin_memory()
        tensor_dst = torch.tensor(dst, dtype=torch.int64).pin_memory()
        num_src = cache_num
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

        # h_res_cpu, degree_cpu = h_res.cpu().tolist(), degree.cpu().tolist()
        # indices = sorted(
        #     range(h_res.numel()), key=lambda k: (h_res_cpu[k], -degree_cpu[k])
        # )
        # indices = sorted(range(h_res.numel()), key=lambda k: h_res_cpu[k])
        indices = torch.argsort(h_res)
        disk_table[disk_cache[indices]] = torch.arange(
            cache_num, dtype=torch.int32, device=device
        )

        for bid in range(startid, endid):
            cold_nodes = all_cold_nodes[bid].to(device)
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
            pages_af_reorder.append(unique_page_id.numel())

            io_traffic[1] += disk_cold.numel() + unique_page_id.numel() * 8
            total_cold_nodes[1] += disk_cold.numel()

            page_diff = np.setdiff1d(unique_page_id.cpu(), page_cache)
            io_traffic[2] += disk_cold.numel() + page_diff.shape[0] * 8
            page_cache = unique_page_id.cpu()

    assert original_cold_nodes[0] == original_cold_nodes[1]
    assert total_cold_nodes[0] == total_cold_nodes[1]

    disk_size += total_cold_nodes[0]
    print(
        "+++++++++++++++++++++++++Before Reorder+++++++++++++++++++++++++\n",
        f"Oringinal Cold Nodes: {original_cold_nodes[0]}, blowup: {original_cold_nodes[0] / dataset.num_nodes}\n",
        f"Packed Nodes: {total_cold_nodes[0]}, Ratio: {total_cold_nodes[0] / original_cold_nodes[0]}\n",
        f"IO Traffic: {io_traffic[0]}, Ratio: {io_traffic[0] / original_cold_nodes[0]}\n",
        f"Disk Size: {disk_size}, Ratio: {disk_size / original_cold_nodes[0]}, Blowup: {disk_size / dataset.num_nodes}",
    )
    record_before = (
        io_traffic[0] / original_cold_nodes[0],
        disk_size / original_cold_nodes[0],
    )
    print(
        "+++++++++++++++++++++++++After Reorder+++++++++++++++++++++++++\n",
        f"Oringinal Cold Nodes: {original_cold_nodes[1]}, blowup: {original_cold_nodes[1] / dataset.num_nodes}\n",
        f"Packed Nodes: {total_cold_nodes[1]}, Ratio: {total_cold_nodes[1] / original_cold_nodes[1]}\n",
        f"IO Traffic: {io_traffic[1]}, Ratio: {io_traffic[1] / original_cold_nodes[1]}\n",
        f"IO Traffic Opt.: {io_traffic[2]}, Ratio: {io_traffic[2] / original_cold_nodes[1]}\n",
        f"Disk Size: {disk_size}, Ratio: {disk_size / original_cold_nodes[1]}, Blowup: {disk_size / dataset.num_nodes}",
    )

    # plot_page_num_ratio(
    #     sorted([pages_af_reorder[i] / pages_bf_reorder[i] for i in range(num_batches)])
    # )
    save_page_num(pages_bf_reorder, pages_af_reorder, segment_size, disk_cache_num)
    return (
        original_cold_nodes[0],
        total_cold_nodes[1],
        io_traffic[1],
        io_traffic[2],
        disk_size,
        record_before,
        np.mean(cache_ratio),
        np.mean(access_ratio),
        np.mean(all_cache_num),
    )


# micro
f = open("logs/disk_cache_fs_ig.csv", "a")
# motivate
# f = open("logs/global_disk_cache_motivate.csv", "a")
writer = csv.writer(f, lineterminator="\n")

io_ratio, disk_ratio, io_ratio_before = [], [], []
if args.dataset == "friendster":
    # motivate io traffic
    # disk_cache_num_list = np.arange(0, 10e6 + 2e6, 2e6, dtype=np.int64)
    # segment_size_list = [50, num_batches]

    # motivate io decompose
    disk_cache_num_list = [int(10e6)]
    segment_size_list = [50]

    # micro exp
    # disk_cache_num_list = np.arange(0, 20e6 + 4e6, 4e6, dtype=np.int64)
    # segment_size_list = [50, 100, 150]
elif args.dataset == "igb-full":
    disk_cache_num_list = np.arange(0, 20e6 + 4e6, 4e6, dtype=np.int64)
    segment_size_list = [100, 200, 400, 800]

for segment_size in segment_size_list:
    for disk_cache_num in disk_cache_num_list:
        (
            original_cold_nodes,
            total_cold_nodes,
            io_traffic,
            io_traffic_opt,
            disk_size,
            record_before,
            cache_ratio,
            access_ratio,
            avg_cache_num,
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
            round(io_traffic_opt / original_cold_nodes, 3),
            # round(original_cold_nodes / dataset.num_nodes, 3),
            # round(disk_size / dataset.num_nodes, 3),
            round(disk_size / original_cold_nodes, 3),
            original_cold_nodes,
            round(cache_ratio, 3),
            round(access_ratio, 3),
            f"{avg_cache_num:g}",
        ]
        writer.writerow(log_info)

f.close()
