import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from load_graph import *

import offgs


def testbench(dataset, args):
    device = "cuda:0"
    fanout = [int(x) for x in args.fanout.split(",")]
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats("csc")
    train_nid, _, _ = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    if args.dataset in ("igb_large", "igb_full"):
        perm_idx = torch.randperm(train_nid.numel())
        train_nid = train_nid[perm_idx[: train_nid.numel() // 6]]

    g = g.to("cpu") if args.use_uva else g.to(device)
    if args.use_uva:
        g.pin_memory_()
    print("training set percent:", train_nid.numel() / g.num_nodes())

    static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    print("memory allocated before training:", static_memory, "GB")

    dataloader = torch.utils.data.DataLoader(train_nid, batch_size=args.batchsize, shuffle=True, drop_last=False)

    hot_node_percent_list = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    cache_percent_list = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

    # dry-run 10 epochs to get nfeat popularity
    nfeat_counts = torch.zeros(g.num_nodes(), dtype=torch.int64)
    for epoch in range(10):
        for step, seeds in enumerate(tqdm(dataloader)):
            blocks = []
            seeds = seeds.to(device)
            for k in reversed(fanout):
                frontier = g.sample_neighbors(seeds, k, replace=False)
                block = dgl.to_block(frontier, seeds)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            input_nodes = blocks[0].srcdata[dgl.NID]
            nfeat_counts[input_nodes.to("cpu")] += 1
    ordered_counts, nfeat_idx = torch.sort(nfeat_counts, dim=0, descending=True)

    # online sampling
    frontier_num = 0
    nfeat_counts = torch.zeros(g.num_nodes(), dtype=torch.int64)
    for epoch in range(args.num_epoch):
        for step, seeds in enumerate(tqdm(dataloader)):
            blocks = []
            seeds = seeds.to(device)
            for k in reversed(fanout):
                frontier_num += seeds.numel()
                frontier = g.sample_neighbors(seeds, k, replace=False)
                block = dgl.to_block(frontier, seeds)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            input_nodes = blocks[0].srcdata[dgl.NID]
            nfeat_counts[input_nodes.to("cpu")] += 1
    print(f"[online] #frontier: {frontier_num // args.num_epoch}")
    total_counts = torch.sum(nfeat_counts).item()
    for cache_percent in cache_percent_list:
        cache_nfeat = nfeat_idx[: int(g.num_nodes() * cache_percent)]
        if cache_percent > 0:
            cache_counts = torch.sum(nfeat_counts[cache_nfeat]).item()
            res = (total_counts - cache_counts) // args.num_epoch
        else:
            res = total_counts // args.num_epoch
        print(
            f"[online] cache percent: {cache_percent},",
            f"#feature access per epoch: {res}",
        )

    # node-wise offline sampling
    # dry-run 10 epochs to get frontier popularity
    frontier_counts = torch.zeros(g.num_nodes(), dtype=torch.int64)
    for epoch in range(10):
        for step, seeds in enumerate(tqdm(dataloader)):
            seeds = seeds.to(device)
            for k in reversed(fanout):
                frontier_counts[seeds.cpu()] += 1
                subg = g.sample_neighbors(seeds, k, replace=False)
                block = dgl.to_block(subg, seeds)
                seeds = block.srcdata[dgl.NID]
    _, frontier_idx = torch.sort(frontier_counts, dim=0, descending=True)

    for sample_percent in hot_node_percent_list:
        hot_idx = frontier_idx[: int(g.num_nodes() * sample_percent)]
        nfeat_blowup_list = [0] * len(cache_percent_list)
        if sample_percent > 0:
            src = g.in_edges(hot_idx)[0]
            unique_idx, counts = torch.unique(src, return_counts=True)
            unique_idx = unique_idx.to(device)
            for i, cache_percent in enumerate(cache_percent_list):
                if cache_percent > 0:
                    hot_nfeat = nfeat_idx[: int(g.num_nodes() * cache_percent)].to(device)
                    _, idx = torch.ops.offgs._CAPI_Difference(unique_idx, hot_nfeat)
                    nfeat_blowup = torch.sum(counts[idx.cpu()]).item() / g.num_nodes()
                else:
                    nfeat_blowup = torch.sum(counts).item() / g.num_nodes()
                nfeat_blowup_list[i] = nfeat_blowup
        hot_idx = hot_idx.to(device)
        hit_num, miss_num = 0, 0
        cold_nfeat_counts = [0] * len(cache_percent_list)
        for epoch in range(args.num_epoch):
            for step, seeds in enumerate(tqdm(dataloader)):
                fetched_nfeat = torch.tensor([], dtype=torch.int64, device=device)
                blocks = []
                seeds = seeds.to(device)
                for k in reversed(fanout):
                    subg = g.sample_neighbors(seeds, k, replace=False)
                    if sample_percent > 0:
                        miss, _ = torch.ops.offgs._CAPI_Difference(seeds, hot_idx)
                        hit, _ = torch.ops.offgs._CAPI_Difference(seeds, miss)
                        hit_num += hit.numel()
                        miss_num += miss.numel()
                        new_nfeat = torch.cat([subg.in_edges(hit)[0], hit])
                        fetched_nfeat = torch.unique(torch.cat([fetched_nfeat, new_nfeat]))
                    else:
                        miss_num += seeds.numel()
                    block = dgl.to_block(subg, seeds)
                    seeds = block.srcdata[dgl.NID]
                    blocks.insert(0, block)
                input_nodes = blocks[0].srcdata[dgl.NID]
                for i, cache_percent in enumerate(cache_percent_list):
                    if cache_percent > 0:
                        hot_nfeat = nfeat_idx[: int(g.num_nodes() * cache_percent)].to(device)
                        ready_nfeat = torch.unique(torch.cat([fetched_nfeat, hot_nfeat]))
                    else:
                        ready_nfeat = fetched_nfeat
                    cold_access, _ = torch.ops.offgs._CAPI_Difference(input_nodes, ready_nfeat)
                    cold_nfeat_counts[i] += cold_access.numel()
        print(
            f"[node-wise offline] pre-sample node percent: {sample_percent},",
            f"#pre-sample hit per epoch: {hit_num // args.num_epoch},",
            f"#pre-sample miss per epoch: {miss_num // args.num_epoch}",
        )
        for i, cache_percent in enumerate(cache_percent_list):
            count = cold_nfeat_counts[i] // args.num_epoch
            print(
                f"[node-wise offline] pre-sample node percent: {sample_percent},",
                f"Nfeat cache percent: {cache_percent},",
                f"#cold feature access per epoch: {count},",
                f"Nfeat blowup: {nfeat_blowup_list[i]}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbn-products", help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=1024, help="batch size for training")
    parser.add_argument("--use-uva", type=bool, default=False, help="use uva for sampling or not")
    parser.add_argument("--num-epoch", type=int, default=3, help="numbers of epoch in training")
    parser.add_argument("--fanout", type=str, default="5,10,15", help="sampling fanout")
    parser.add_argument(
        "--path",
        type=str,
        default="/efs/rjliu/dataset/igb_tiny",
        help="path containing the datasets",
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium", "large", "full"],
        help="size of the datasets",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=19,
        choices=[19, 2983],
        help="number of classes",
    )
    parser.add_argument(
        "--in_memory",
        type=int,
        default=1,
        choices=[0, 1],
        help="0:read only mmap_mode=r, 1:load into memory",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        choices=[0, 1],
        help="0:nlp-node embeddings, 1:random",
    )
    args = parser.parse_args()
    print(args)
    if args.dataset.startswith("ogbn"):
        dataset = load_ogb(args.dataset, "/efs/rjliu/dataset")
    elif args.dataset == "friendster":
        dataset = load_dglgraph("/efs/rjliu/dataset/friendster/friendster.bin")
    elif args.dataset.startswith("igb"):
        dataset = load_igb(args)
    else:
        raise NotImplementedError
    print(dataset[0])
    testbench(dataset, args)
