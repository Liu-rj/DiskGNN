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

import offgs


def train(args, dataset, address_table, cached_feats, subg_dir, aux_dir):
    device = torch.device(f"cuda:{args.device}")
    fanout = [int(x) for x in args.fanout.split(",")]

    labels = dataset.labels

    if args.model == "SAGE":
        model = SAGE(dataset.num_features, 256, dataset.num_classes, len(fanout)).to(device)
    elif args.model == "GAT":
        model = GAT(dataset.num_features, 256, dataset.num_classes, [8, 2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    size = (dataset.split_idx["train"].numel() + args.batchsize - 1) // args.batchsize

    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")

    epoch_time_list, clear_cache_list = [], []
    for epoch in range(args.num_epoch):
        clear_cache, transfer_time, train_time = 0, 0, 0
        graph_load_time, feats_load_time, assemble_time = 0, 0, 0
        input_feat_size, input_node_num = 0, 0
        cold_feats_num = 0

        torch.cuda.synchronize()
        start = time.time()

        model.train()
        for i in trange(size):
            torch.cuda.synchronize()
            tic = time.time()
            # Same effect of `sysctl -w vm.drop_caches=1`
            # Requires sudo
            with open("/proc/sys/vm/drop_caches", "w") as stream:
                stream.write("1\n")
            clear_cache += time.time() - tic

            tic = time.time()
            blocks = torch.load(f"{subg_dir}/train-{i}.pt")
            graph_load_time += time.time() - tic
            tic = time.time()
            cold_feats, cold_nodes, hot_nodes, rev_hot_idx, rev_cold_idx = torch.ops.offgs._CAPI_LoadFeats_ODirect(
                f"{aux_dir}/train-aux-{i}.npy", dataset.num_features, 8, 64
            )
            feats_load_time += time.time() - tic
            cold_feats_num += cold_feats.shape[0]

            tic = time.time()
            num_input = blocks[0].srcdata[dgl.NID].numel()
            x = torch.empty((num_input, dataset.num_features), dtype=torch.float32)
            x[rev_cold_idx] = cold_feats
            x[rev_hot_idx] = cached_feats[address_table[hot_nodes]]
            assemble_time = time.time() - tic

            tic = time.time()
            x = x.to(device)
            y = labels[blocks[-1].dstdata[dgl.NID]].to(device).long()
            blocks = [block.to(device) for block in blocks]
            torch.cuda.synchronize()
            transfer_time += time.time() - tic
            input_node_num += x.shape[0]
            input_feat_size += x.shape[0] * x.shape[1]

            tic = time.time()
            pred = model(blocks, x)
            loss = F.cross_entropy(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            train_time += time.time() - tic

        torch.cuda.synchronize()
        epoch_time = time.time() - start
        print(
            f"Graph Load Time: {graph_load_time:.3f}\t"
            f"Feature Load Time: {feats_load_time:.3f}\t"
            f"Assemble Time: {assemble_time:.3f}\t"
            f"Transfer Time: {transfer_time:.3f}\t"
            f"Train Time: {train_time:.3f}\t"
            f"Epoch Time: {(epoch_time - clear_cache):.3f}\t"
            f"Drop Cache Time: {clear_cache:.3f}"
        )
        print(f"Cold Feats Num: {cold_feats_num}\t" f"Feature Transfer size: {input_feat_size}\t" f"Input Node num: {input_node_num}")
        epoch_time_list.append(epoch_time)
        clear_cache_list.append(clear_cache)

    torch.cuda.synchronize()
    avg_clear_cache = np.mean(clear_cache_list[1:])
    print(f"Avg Clear Cache Time: {avg_clear_cache:.3f}\t" f"Avg Epoch Time: {(np.mean(epoch_time_list[1:]) - avg_clear_cache):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Training model device")
    parser.add_argument("--batchsize", type=int, default=1000, help="batch size for training")
    parser.add_argument("--dataset", default="ogbn-products", help="dataset")
    parser.add_argument("--fanout", type=str, default="10,10,10", help="sampling fanout")
    parser.add_argument("--model", type=str, default="SAGE", help="training model")
    parser.add_argument("--dir", type=str, default="/nvme2n1", help="path to store subgraph")
    parser.add_argument("--feat-cache-size", type=int, default=1000000000, help="cache size in bytes")
    parser.add_argument("--num-epoch", type=int, default=3, help="numbers of epoch in training")
    args = parser.parse_args()
    print(args)

    subg_dir = f"{args.dir}/{args.dataset}-{args.fanout}"
    aux_dir = f"{subg_dir}/cache-size-{args.feat_cache_size}"
    dataset_dir = f"{args.dir}/{args.dataset}-offgs"

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    # indices = torch.load(f"{subg_dir}/meta_node_popularity.pt")

    dataset = OffgsDataset(dataset_dir)
    cached_nodes = torch.load(f"{aux_dir}/cached_nodes.pt")
    cached_feats = torch.empty((cached_nodes.numel(), dataset.num_features), dtype=torch.float32)
    cached_feats[:] = dataset.mmap_features[cached_nodes]
    address_table = torch.load(f"{aux_dir}/address_table.pt")

    # address_table, cache = init_cache(indices, dataset.mmap_features, mmap_features.shape[0], mmap_features.shape[1], args.feat_cache_size)

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem1 - mem, "GB")

    train(args, dataset, address_table, cached_feats, subg_dir, aux_dir)
