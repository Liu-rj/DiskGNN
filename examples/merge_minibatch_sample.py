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

import offgs


def run(args, dataset):
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    subg_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    subg_dir = os.path.join(args.store_path, subg_dir)
    output_dir = os.path.join(subg_dir, str(args.mega_batch_size))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    train_idx = (
        dataset.split_idx["train"]
        if args.ratio == 1.0
        else torch.load(f"{dataset_path}/train_idx_{args.ratio}.pt")
    )
    num_batches = (train_idx.numel() + args.batchsize - 1) // args.batchsize

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory consumption:", mem1 - mem, "GB")

    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")

    sample_time, save_block, save_rank = 0, 0, 0
    tic = time.time()

    num_megabatch = args.mega_batch_size // args.batchsize
    batch_in_nid = []
    unique_inv_idx = []

    for i in trange(num_batches, ncols=100):
        input_nodes = torch.load(f"{subg_dir}/in-nid-{i}.pt")
        sample_time += time.time() - tic
        batch_in_nid.append(input_nodes)
        if (i + 1) % num_megabatch == 0 or i == num_batches - 1:
            merged_input_nodes = torch.cat(batch_in_nid)
            # cal unique nodes
            unique_nodes, inverse_indices = torch.unique(
                merged_input_nodes, return_inverse=True
            )

            start_idx = 0
            for block_index, block_input_nodes in enumerate(batch_in_nid):
                end_idx = start_idx + block_input_nodes.size(0)
                block_unique_indices = inverse_indices[start_idx:end_idx]
                unique_inv_idx.append(block_unique_indices)
                start_idx = end_idx
                assert unique_nodes[unique_inv_idx[block_index]].equal(
                    block_input_nodes
                )

            torch.save(unique_nodes, f"{output_dir}/merge-in-nid-{i}.pt")
            ## store mapping
            torch.save(unique_inv_idx, f"{output_dir}/unique_inv_idx-{i}.pt")
            # clear
            batch_in_nid = []
            unique_inv_idx = []

        tic = time.time()

    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.ratio,
            round(sample_time, 2),
            round(save_block, 2),
            round(save_rank, 2),
            round(sample_time + save_block + save_rank, 2),
        ]
        writer.writerow(log_info)

    print(
        f"Sampling Time: {sample_time:.3f}\t"
        f"Save Block Time: {save_block:.3f}\t"
        f"Save Rank Time: {save_rank:.3f}\t"
        f"Total Time: {(sample_time + save_block + save_rank):.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--store-path", default="/nvme2n1")
    parser.add_argument("--ratio", type=float, default=1.0)
    ## add mega batch size eg 2048. 4096, 8192
    parser.add_argument("--mega_batch_size", type=int, default=4096)
    parser.add_argument(
        "--log", type=str, default="logs/merge_megabatch_sample_decompose.csv"
    )
    args = parser.parse_args()
    print(args)

    # --- load dataset --- #
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    dataset = OffgsDataset(dataset_path)

    run(args, dataset)
