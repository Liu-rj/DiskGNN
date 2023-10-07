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
    output_dir = f"{args.store_path}/{args.dataset}-{args.fanout}"
    aux_dir = f"{output_dir}/cache-size-{args.feat_cache_size}"

    if not os.path.exists(aux_dir):
        os.mkdir(aux_dir)

    features = dataset.mmap_features

    sorted_idx = torch.load(f"{output_dir}/meta_node_popularity.pt").cpu()

    feat_load_time, nid_load_time, difference_time, save_time = 0, 0, 0, 0
    clear_cache_time = 0

    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")

    start = time.time()

    tic = time.time()
    num_batches = (dataset.split_idx["train"].numel() + args.batchsize - 1) // args.batchsize
    table_size = 4 * dataset.num_nodes
    num_entries = min((args.feat_cache_size - table_size) // (4 * features.shape[1]), dataset.num_nodes)
    if num_entries > torch.iinfo(torch.int32).max:
        raise ValueError
    print(f"#Cached Entries: {num_entries}")
    cache_indices = sorted_idx[:num_entries]
    address_table = torch.full((dataset.num_nodes,), -1, dtype=torch.int32)
    address_table[cache_indices] = torch.arange(num_entries, dtype=torch.int32)
    torch.save(cache_indices, f"{aux_dir}/cached_nodes.pt")
    torch.save(address_table, f"{aux_dir}/address_table.pt")
    cache_init_time = time.time() - tic

    cache_indices = cache_indices.to("cuda")
    for i in trange(num_batches):
        tic = time.time()
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")
        clear_cache_time += time.time() - tic

        tic = time.time()
        input_nodes: torch.Tensor = torch.load(f"{output_dir}/train-nid-{i}.pt").to("cuda")
        nid_load_time += time.time() - tic

        tic = time.time()
        # input_nodes = blocks[0].srcdata[dgl.NID].to("cuda")
        # cold_nodes, rev_cold_idx = torch.ops.offgs._CAPI_Difference(input_nodes, cache_indices)
        # hot_nodes, rev_hot_idx = torch.ops.offgs._CAPI_Difference(input_nodes, cold_nodes)

        rev_hot_idx = torch.isin(input_nodes, cache_indices, assume_unique=True).nonzero(as_tuple=True)[0]
        hot_nodes = input_nodes[rev_hot_idx]
        rev_cold_idx = torch.isin(input_nodes, hot_nodes, assume_unique=True, invert=True).nonzero(as_tuple=True)[0]
        cold_nodes = input_nodes[rev_cold_idx]
        torch.cuda.synchronize()
        difference_time += time.time() - tic

        tic = time.time()
        packed_feats: torch.Tensor = features[cold_nodes.cpu()]
        # packed_feats = torch.ops.offgs._CAPI_GatherMemMap(features, cold_nodes.cpu(), dataset.num_features)
        # packed_feats = torch.ops.offgs._CAPI_GatherPRead(dataset.features_path, cold_nodes.cpu(), dataset.num_features)
        feat_load_time += time.time() - tic

        tic = time.time()
        aux_data = torch.cat([packed_feats.flatten(), cold_nodes.cpu(), hot_nodes.cpu(), rev_hot_idx.cpu(), rev_cold_idx.cpu()])
        stored_data = np.memmap(f"{aux_dir}/train-aux-{i}.npy", mode='w+', shape=aux_data.numel() + 5, dtype=np.float32)
        stored_data[:5] = [packed_feats.numel(), cold_nodes.numel(), hot_nodes.numel(), rev_hot_idx.numel(), rev_cold_idx.numel()]
        stored_data[5:] = aux_data
        stored_data.flush()
        save_time += time.time() - tic

    total_time = time.time() - start
    # with open("logs/pack_decompose.csv", "a") as f:
    #     writer = csv.writer(f, lineterminator="\n")
    #     log_info = [
    #         round(cache_init_time, 3),
    #         round(blocks_load_time, 3),
    #         round(difference_time, 3),
    #         round(feat_load_time, 3),
    #         round(save_time, 3),
    #         round(total_time, 3),
    #     ]
    #     writer.writerow(log_info)

    print(
        f"Init Cache Indices Time: {cache_init_time:.3f}\t"
        f"Drop Cache Time: {clear_cache_time:.3f}\t"
        f"NID Load Time: {nid_load_time:.3f}\t"
        f"Set Difference Time: {difference_time:.3f}\t"
        f"Feat Load Time: {feat_load_time:.3f}\t"
        f"Save Time: {save_time:.3f}\t"
        f"Total Time: {(total_time - clear_cache_time):.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-products", help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=1000, help="batch size for training")
    parser.add_argument("--fanout", type=str, default="10,10,10", help="sampling fanout")
    parser.add_argument("--store-path", default="/nvme2n1", help="path to store subgraph")
    parser.add_argument("--feat-cache-size", type=int, default=1000000000, help="cache size in bytes")
    args = parser.parse_args()
    print(args)

    # --- load data --- #
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    dataset = OffgsDataset(dataset_path)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem, "GB")

    run(dataset, args)
