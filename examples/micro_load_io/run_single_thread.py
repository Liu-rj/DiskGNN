import torch
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from offgs.utils import SAGE, GAT
from offgs.dataset import OffgsDataset
import threading
import queue
import psutil
import csv
import json
import os

import offgs


def train(args, dataset: OffgsDataset, subg_dir: str, aux_dir: str):
    train_num = dataset.split_idx["train"].numel()
    print(f"Label Ratio: {train_num / dataset.num_nodes}, Down Sample: {args.ratio}")
    pool_size = (int(train_num * args.ratio) + args.batchsize - 1) // args.batchsize

    batch_id = torch.randperm(pool_size).tolist()
    info_recorder = [0] * 2
    for i in tqdm(batch_id, ncols=100):
        input_nodes = torch.load(f"{subg_dir}/in-nid-{i}.pt")
        info_recorder[0] += input_nodes.numel()  # graph load

        (
            disk_cold,
            disk_rev_cold_idx,
            disk_loc,
            disk_rev_hot_idx,
            rev_cold_idx,
            mem_loc,
            rev_hot_idx,
        ) = torch.load(f"{aux_dir}/meta_data/train-aux-meta-{i}.pt")
        info_recorder[1] += disk_cold.numel() + disk_loc.numel()

    print(
        f"Block Input Node Num: {info_recorder[0]}\t"
        f"Cold Feats Num: {info_recorder[1]}\t"
    )

    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.batchsize,
            f"{args.cpu_cache_size:g}",
            f"{args.gpu_cache_size:g}",
            args.blowup,
        ]
        for epoch_info in info_recorder:
            log_info.append(epoch_info)
        writer.writerow(log_info)


def start(args):
    total_cache_size = args.cpu_cache_size + args.gpu_cache_size

    subg_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    subg_dir = os.path.join(args.dir, subg_dir)
    aux_dir = f"{subg_dir}/cache-size-{total_cache_size:g}/blowup-{args.blowup}"
    dataset_dir = f"{args.dir}/{args.dataset}-offgs"

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    dataset = OffgsDataset(dataset_dir)

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem1 - mem, "GB")

    train(args, dataset, subg_dir, aux_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--dataset", default="friendster")
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--dir", type=str, default="/nvme1n1/offgs_dataset")
    parser.add_argument("--cpu-cache-size", type=float, default=1e10)
    parser.add_argument("--gpu-cache-size", type=float, default=0)
    parser.add_argument("--blowup", type=float, default=-1)
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--log", type=str, default="io_load.csv")
    args = parser.parse_args()
    print(args)

    start(args)
