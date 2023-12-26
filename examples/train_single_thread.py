import torch
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
from ctypes import *
from ctypes.util import *
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
import pandas as pd
from load_graph import *
from model import *
from dataset import OffgsDataset
from queue import Queue
import threading
import psutil
import csv

import offgs


def train(
    args,
    dataset: OffgsDataset,
    address_table: torch.Tensor,
    cpu_cached_feats: torch.Tensor,
    gpu_cached_feats: torch.Tensor,
    subg_dir: str,
    aux_dir: str,
    cached_nodes: torch.Tensor,
):
    device = torch.device(f"cuda:{args.device}")
    fanout = [int(x) for x in args.fanout.split(",")]

    labels = dataset.labels.pin_memory()
    cpu_cached_feats = cpu_cached_feats.pin_memory()

    if args.model == "SAGE":
        model = SAGE(dataset.num_features, 256, dataset.num_classes, len(fanout)).to(
            device
        )
    elif args.model == "GAT":
        model = GAT(dataset.num_features, 256, dataset.num_classes, [8, 2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    size = (dataset.split_idx["train"].numel() + args.batchsize - 1) // args.batchsize

    epoch_info_recorder = [[] for i in range(9)]
    for epoch in range(args.num_epoch):
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")

        info_recorder = [0] * 8

        torch.cuda.synchronize()
        start = time.time()

        model.train()
        for i in trange(size, ncols=100):
            # tic = time.time()
            # with open("/proc/sys/vm/drop_caches", "w") as stream:
            #     stream.write("1\n")
            # clear_cache += time.time() - tic

            tic = time.time()
            blocks = torch.load(f"{subg_dir}/train-{i}.pt")
            output_nodes = torch.load(f"{subg_dir}/out-nid-{i}.pt")
            info_recorder[0] += time.time() - tic  # graph load
            tic = time.time()
            (
                cold_nodes,
                hot_nodes,
                rev_hot_idx,
                rev_cold_idx,
            ) = torch.load(f"{aux_dir}/train-aux-meta-{i}.pt")
            if cold_nodes.numel() > 0:
                cold_feats = torch.ops.offgs._CAPI_LoadFeats_Direct(
                    f"{aux_dir}/train-aux-{i}.npy",
                    cold_nodes.numel(),
                    dataset.num_features,
                )
            # cold_feats, cold_nodes, hot_nodes, rev_hot_idx, rev_cold_idx = torch.load(
            #     f"{aux_dir}/train-aux-{i}.pt"
            # )
            info_recorder[1] += time.time() - tic  # feature load
            info_recorder[6] += cold_nodes.numel()  # cold_feats_num

            tic = time.time()
            num_input = cold_nodes.numel() + hot_nodes.numel()
            x = torch.empty(
                (num_input, dataset.num_features),
                dtype=torch.float32,
                device=device,
            )
            if cold_nodes.numel() > 0:
                x[rev_cold_idx] = cold_feats
                # x[rev_hot_idx] = cached_feats[address_table[hot_nodes]]
            torch.ops.offgs._CAPI_GatherInGPU(
                x,
                rev_hot_idx,
                cpu_cached_feats,
                gpu_cached_feats,
                hot_nodes,
                address_table,
            )
            info_recorder[2] += time.time() - tic  # assemble

            tic = time.time()
            blocks = [block.to(device) for block in blocks]
            torch.cuda.synchronize()
            info_recorder[3] += time.time() - tic  # graph transfer
            tic = time.time()
            y = labels[output_nodes].to(device).long()
            torch.cuda.synchronize()
            info_recorder[4] += time.time() - tic  # feature transfer
            info_recorder[7] += x.shape[0]  # input node num

            tic = time.time()
            pred = model(blocks, x)
            loss = F.cross_entropy(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            info_recorder[5] += time.time() - tic

        print(
            f"Graph Load Time: {info_recorder[0]:.3f}\t"
            f"Feature Load Time: {info_recorder[1]:.3f}\t"
            f"Assemble Time: {info_recorder[2]:.3f}\t"
            f"Graph Transfer Time: {info_recorder[3]:.3f}\t"
            f"Feat Transfer Time: {info_recorder[4]:.3f}\t"
            f"Train Time: {info_recorder[5]:.3f}\t"
            f"Epoch Time: {np.sum(info_recorder[:6]):.3f}\t"
            f"Cold Feats Num: {info_recorder[6]}\t"
            f"Feature Transfer Num: {info_recorder[7]}\t"
        )
        for i, info in enumerate(info_recorder):
            epoch_info_recorder[i].append(info)
        epoch_info_recorder[-1].append(np.sum(info_recorder[:6]))

    with open("logs/train_single_thread_decompose.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.cpu_cache_size,
            args.gpu_cache_size,
            args.model,
            args.num_epoch,
        ]
        for epoch_info in epoch_info_recorder:
            log_info.append(round(np.mean(epoch_info[1:]), 2))
        writer.writerow(log_info)


def init_cache(args, dataset, cached_nodes):
    device = torch.device(f"cuda:{args.device}")
    table_size = 4 * dataset.num_nodes
    print(f"Adress Table Size: {table_size / (1024 * 1024 * 1024)} GB")
    gpu_num_entries = (args.gpu_cache_size - table_size) // (4 * dataset.num_features)
    gpu_cached_idx = cached_nodes[:gpu_num_entries]
    cpu_cache_idx = cached_nodes[gpu_num_entries:]
    gpu_cached_feats = dataset.mmap_features[gpu_cached_idx].to(device)
    cpu_cached_feats = torch.empty(
        (cpu_cache_idx.numel(), dataset.num_features), dtype=torch.float32
    )
    cpu_cached_feats[:] = dataset.mmap_features[cpu_cache_idx]
    address_table = torch.zeros((dataset.num_nodes,), dtype=torch.int32)
    address_table[cpu_cache_idx] = (
        torch.arange(cpu_cache_idx.numel(), dtype=torch.int32) + 1
    )
    address_table[gpu_cached_idx] = -(
        torch.arange(gpu_cached_idx.numel(), dtype=torch.int32) + 1
    )
    address_table = address_table.to(device)
    return cpu_cached_feats, gpu_cached_feats, address_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Training model device")
    parser.add_argument(
        "--batchsize", type=int, default=1024, help="batch size for training"
    )
    parser.add_argument("--dataset", default="ogbn-products", help="dataset")
    parser.add_argument(
        "--fanout", type=str, default="10,10,10", help="sampling fanout"
    )
    parser.add_argument("--model", type=str, default="SAGE", help="training model")
    parser.add_argument(
        "--dir",
        type=str,
        default="/nvme1n1/offgs_dataset",
        help="path to store subgraph",
    )
    parser.add_argument(
        "--cpu-cache-size", type=int, default=1000000000, help="cache size in bytes"
    )
    parser.add_argument(
        "--gpu-cache-size", type=int, default=1000000000, help="cache size in bytes"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=3, help="numbers of epoch in training"
    )
    args = parser.parse_args()
    print(args)

    total_cache_size = args.cpu_cache_size + args.gpu_cache_size

    subg_dir = f"{args.dir}/{args.dataset}-{args.batchsize}-{args.fanout}"
    aux_dir = f"{subg_dir}/cache-size-{total_cache_size}"
    dataset_dir = f"{args.dir}/{args.dataset}-offgs"

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    dataset = OffgsDataset(dataset_dir)
    cached_nodes = torch.load(f"{aux_dir}/cached_nodes.pt")

    (
        cpu_cached_feats,
        gpu_cached_feats,
        address_table,
    ) = init_cache(args, dataset, cached_nodes)

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem1 - mem, "GB")

    train(
        args,
        dataset,
        address_table,
        cpu_cached_feats,
        gpu_cached_feats,
        subg_dir,
        aux_dir,
        cached_nodes,
    )
