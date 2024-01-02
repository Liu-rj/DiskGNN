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


def run(args, dataset, label_offset):
    fanout = [int(x) for x in args.fanout.split(",")]
    output_dir = f"{args.store_path}/{args.dataset}-{args.batchsize}-{args.fanout}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats("csc")
    train_nid, _, _ = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    sampler = NeighborSampler(fanout)
    train_dataloader = DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=2,
    )

    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")

    node_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device="cuda")
    sample_time, save_block, save_rank = 0, 0, 0
    tic = time.time()
    for i, (input_nodes, output_nodes, blocks) in enumerate(tqdm(train_dataloader)):
        for it, block in enumerate(blocks):
            block.ndata.clear()
            block.edata.clear()
            block.srcdata.clear()
            block.dstdata.clear()
            blocks[it] = block
        sample_time += time.time() - tic

        tic = time.time()
        torch.save(blocks, f"{output_dir}/train-{i}.pt")
        torch.save(input_nodes, f"{output_dir}/in-nid-{i}.pt")
        torch.save(output_nodes, f"{output_dir}/out-nid-{i}.pt")
        node_counts[input_nodes.cuda()] += 1
        save_block += time.time() - tic

        tic = time.time()

    tic = time.time()
    sorted_idx = torch.argsort(node_counts, descending=True).cpu()
    ## save nodecounts
    torch.save(node_counts, f"{output_dir}/node_counts.pt")
    torch.save(sorted_idx, f"{output_dir}/meta_node_popularity.pt")
    save_rank += time.time() - tic

    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
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
        "--store-path", default="/nvme2n1", help="path to store subgraph"
    )
    parser.add_argument(
        "--log", type=str, default="logs/sample_decompose.csv", help="log file"
    )
    args = parser.parse_args()
    print(args)

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    label_offset = 0
    if args.dataset.startswith("ogbn"):
        dataset = load_ogb(args.dataset, "/efs/rjliu/dataset")
    elif args.dataset.startswith("igb"):
        dataset = load_igb(args)
    elif args.dataset == "mag240m":
        dataset = load_mag240m("/efs/rjliu/dataset/mag240m", only_graph=False)
        label_offset = dataset[-1]
        dataset = dataset[:-1]
    elif args.dataset == "friendster":
        dataset = load_dglgraph("/efs/rjliu/dataset/friendster/friendster.bin", 0, 0)
    else:
        raise NotImplementedError

    print(dataset[0])
    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Graph total memory:", mem1 - mem, "GB")

    run(args, dataset, label_offset)
