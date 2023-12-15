import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from load_graph import *
from model import *
import psutil
import time
import json
from dataset import OffgsDataset
import csv

import offgs


def run(dataset, args):
    output_dir = f"{args.store_path}/{args.dataset}-{args.batchsize}-{args.fanout}"
    aux_dir = f"{output_dir}/cache-size-{args.feat_cache_size}"

    if not os.path.exists(aux_dir):
        os.mkdir(aux_dir)

    features = dataset.mmap_features

    sorted_idx = torch.load(f"{output_dir}/meta_node_popularity.pt").cpu()

    feat_load_time, nid_load_time, difference_time, save_time = 0, 0, 0, 0

    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")

    start = time.time()

    tic = time.time()
    num_batches = (
        dataset.split_idx["train"].numel() + args.batchsize - 1
    ) // args.batchsize
    table_size = 4 * dataset.num_nodes
    num_entries = min(
        (args.feat_cache_size - table_size) // (4 * features.shape[1]),
        dataset.num_nodes,
    )
    # Maximum 400GB cache size for int32 (aligned with Ginex)
    if num_entries > torch.iinfo(torch.int32).max:
        raise ValueError
    print(f"#Cached Entries: {num_entries} / {dataset.num_nodes}")
    cache_indices = sorted_idx[:num_entries]
    address_table = torch.full((dataset.num_nodes,), -1, dtype=torch.int32)
    address_table[cache_indices] = torch.arange(num_entries, dtype=torch.int32)
    torch.save(cache_indices, f"{aux_dir}/cached_nodes.pt")
    torch.save(address_table, f"{aux_dir}/address_table.pt")
    key, value = torch.ops.offgs._CAPI_BuildHashMap(cache_indices.to("cuda"))
    cache_init_time = time.time() - tic

    for i in trange(num_batches):
        # tic = time.time()
        # with open("/proc/sys/vm/drop_caches", "w") as stream:
        #     stream.write("1\n")
        # clear_cache_time += time.time() - tic

        tic = time.time()
        input_nodes: torch.Tensor = torch.load(f"{output_dir}/in-nid-{i}.pt").to("cuda")
        nid_load_time += time.time() - tic

        tic = time.time()
        (
            cold_nodes,
            rev_cold_idx,
            hot_nodes,
            rev_hot_idx,
        ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)

        # rev_hot_idx = torch.isin(input_nodes, cache_indices, assume_unique=True).nonzero(as_tuple=True)[0]
        # hot_nodes = input_nodes[rev_hot_idx]
        # rev_cold_idx = torch.isin(input_nodes, hot_nodes, assume_unique=True, invert=True).nonzero(as_tuple=True)[0]
        # cold_nodes = input_nodes[rev_cold_idx]
        torch.cuda.synchronize()
        difference_time += time.time() - tic

        tic = time.time()
        packed_feats = torch.ops.offgs._CAPI_GatherPReadDirect(
            dataset.features_path, cold_nodes.cpu(), dataset.num_features
        )
        # packed_feats: torch.Tensor = features[cold_nodes.cpu()]
        # packed_feats = torch.ops.offgs._CAPI_GatherMemMap(features, cold_nodes.cpu(), dataset.num_features)
        # packed_feats = torch.ops.offgs._CAPI_GatherPRead(dataset.features_path, cold_nodes.cpu(), dataset.num_features)
        feat_load_time += time.time() - tic

        tic = time.time()
        aux_data = torch.cat(
            [
                packed_feats.flatten(),
                cold_nodes.cpu(),
                hot_nodes.cpu(),
                rev_hot_idx.cpu(),
                rev_cold_idx.cpu(),
            ]
        )
        stored_data = np.memmap(
            f"{aux_dir}/train-aux-{i}.npy",
            mode="w+",
            shape=aux_data.numel() + 5,
            dtype=np.float32,
        )
        stored_data[:5] = [
            packed_feats.numel(),
            cold_nodes.numel(),
            hot_nodes.numel(),
            rev_hot_idx.numel(),
            rev_cold_idx.numel(),
        ]
        stored_data[5:] = aux_data
        stored_data.flush()
        # aux_data = [
        #     packed_feats,
        #     cold_nodes.cpu(),
        #     hot_nodes.cpu(),
        #     rev_hot_idx.cpu(),
        #     rev_cold_idx.cpu(),
        # ]
        # torch.save(aux_data, f"{aux_dir}/train-aux-{i}.pt")
        save_time += time.time() - tic

    total_time = time.time() - start
    with open("logs/pack_decompose.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.feat_cache_size,
            round(cache_init_time, 2),
            round(nid_load_time, 2),
            round(difference_time, 2),
            round(feat_load_time, 2),
            round(save_time, 2),
            round(total_time, 2),
        ]
        writer.writerow(log_info)

    print(
        f"Init Cache Indices Time: {cache_init_time:.3f}\t"
        f"NID Load Time: {nid_load_time:.3f}\t"
        f"Set Difference Time: {difference_time:.3f}\t"
        f"Feat Load Time: {feat_load_time:.3f}\t"
        f"Save Time: {save_time:.3f}\t"
        f"Total Time: {(total_time):.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="which dataset to load for training",
    )
    parser.add_argument(
        "--batchsize", type=int, default=1024, help="batch size for training"
    )
    parser.add_argument(
        "--fanout", type=str, default="10,10,10", help="sampling fanout"
    )
    parser.add_argument(
        "--store-path", default="/nvme1n1", help="path to store subgraph"
    )
    parser.add_argument(
        "--feat-cache-size", type=int, default=1000000000, help="cache size in bytes"
    )
    args = parser.parse_args()
    print(args)

    # --- load data --- #
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    dataset = OffgsDataset(dataset_path)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem, "GB")

    run(dataset, args)
