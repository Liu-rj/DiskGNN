import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from load_graph import *


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

    g = g.to("cpu") if args.use_uva else g.to(device)
    if args.use_uva:
        g.pin_memory_()
    train_nid = train_nid.to(device)
    print("training set percent:", train_nid.numel() / g.num_nodes())

    static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    print("memory allocated before training:", static_memory, "GB")

    dataloader = torch.utils.data.DataLoader(
        train_nid, batch_size=args.batchsize, shuffle=True, drop_last=False
    )

    hot_node_percent_list = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    cache_percent_list = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

    # online sampling, currently only consider feature loading latency
    sampled_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device=device)
    for epoch in range(args.num_epoch):
        for step, seeds in enumerate(tqdm(dataloader)):
            blocks = []
            for k in reversed(fanout):
                frontier = g.sample_neighbors(seeds, k, replace=False)
                block = dgl.to_block(frontier, seeds)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            input_nodes = blocks[0].srcdata[dgl.NID]
            sampled_counts[input_nodes] += 1
    ordered_counts, idx = torch.sort(sampled_counts, dim=0, descending=True)
    for cache_percent in cache_percent_list:
        res = torch.sum(ordered_counts[int(g.num_nodes() * cache_percent) :])
        res = res.item() // args.num_epoch
        print(
            f"[online] cache percent: {cache_percent}, #feature access per epoch: {res}"
        )

    # node-wise offline sampling
    counts_layer = [
        torch.zeros(g.num_nodes(), dtype=torch.int64) for i in range(len(fanout))
    ]
    for epoch in range(args.num_epoch):
        for step, seeds in enumerate(tqdm(dataloader)):
            for layer, k in enumerate(reversed(fanout)):
                counts_layer[layer][seeds.cpu()] += 1
                subg = g.sample_neighbors(seeds, k, replace=False)
                block = dgl.to_block(subg, seeds)
                seeds = block.srcdata[dgl.NID]
    frontier_counts = torch.zeros(g.num_nodes(), dtype=torch.int64)
    for counts in counts_layer:
        frontier_counts += counts
    _, idx = torch.sort(frontier_counts, dim=0, descending=True)

    for percent in hot_node_percent_list:
        hot_idx = idx[: int(g.num_nodes() * percent)]
        cold_idx = idx[int(g.num_nodes() * percent) :]
        if percent > 0:
            hot_counts = torch.sum(frontier_counts[hot_idx]).item() // args.num_epoch
        else:
            hot_counts = 0
        cold_counts = 0
        for layer, counts in enumerate(counts_layer):
            k = fanout[len(fanout) - layer - 1]
            cold_counts += k * (torch.sum(counts[cold_idx]).item() // args.num_epoch)
        print(
            f"[node-wise offline] pre-sample node percent: {percent}, #pre-sample access per epoch: {hot_counts}, #cold feature access per epoch: {cold_counts}"
        )
        if percent > 0:
            src = g.in_edges(hot_idx)[0]
            unique_res, counts = torch.unique(src, return_counts=True)
            ordered_counts, _ = torch.sort(counts, dim=0, descending=True)
            for cache_percent in cache_percent_list:
                res = torch.sum(ordered_counts[int(g.num_nodes() * cache_percent) :])
                res = res.item() / g.num_nodes()
                print(
                    f"[node-wise offline] pre-sample node percent: {percent}, cache percent: {cache_percent}, feature percent: {res}"
                )
        else:
            print(
                f"[node-wise offline] pre-sample node percent: {percent}, feature percent: {0}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="ogbn-products", help="which dataset to load for training"
    )
    parser.add_argument(
        "--batchsize", type=int, default=1024, help="batch size for training"
    )
    parser.add_argument(
        "--use-uva", type=bool, default=False, help="use uva for sampling or not"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=10, help="numbers of epoch in training"
    )
    parser.add_argument("--fanout", type=str, default="5,10,15", help="sampling fanout")
    parser.add_argument(
        "--path",
        type=str,
        default="/efs/rjliu/dataset/igb_large",
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
