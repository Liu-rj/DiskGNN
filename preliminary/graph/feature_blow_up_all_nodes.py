import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from load_graph import *

import offgs


def testbench(dataset, args):
    device = "cuda:2"
    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, _, _ = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    # g = g.to("cpu") if args.use_uva else g.to(device)
    train_nid = train_nid.to(device)
    print("training set percent:", train_nid.numel() / g.num_nodes())

    sampled_counts = g.out_degrees()
    ordered_counts, idx = torch.sort(sampled_counts, dim=0, descending=True)
    cache_percent_list = [0.1, 0.2, 0.3, 0.4]
    results = []
    for cache_percent in cache_percent_list:
        res = torch.sum(ordered_counts[int(ordered_counts.shape[0] * cache_percent) :])
        res = res.item() / g.num_nodes()
        results.append(res)
        print(f"cache percent.: {cache_percent}, feature percent.: {res}")

    font_size = 20
    marker_size = 10
    plt.figure(figsize=(15, 8))
    x = cache_percent_list
    plt.plot(
        x,
        results,
        label="gather all neighbors",
        color="r",
        marker="o",
        markersize=marker_size,
    )
    plt.grid()
    plt.xticks(x, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel("packed node feature / #nodes", fontsize=font_size)
    plt.xlabel("cache percentage", fontsize=font_size)
    plt.title(
        f"Cache effect on node feature redundancy ({args.dataset})", fontsize=font_size
    )
    plt.legend(fontsize=font_size)
    plt.savefig(
        f"imgs/feature_redundancy_allsample_{args.dataset}.png", bbox_inches="tight"
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
