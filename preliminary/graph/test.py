import torch
import dgl
from dgl.dataloading import DataLoader, ShaDowKHopSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from load_graph import *
from model import *
import psutil
import os
import copy


def train_dgl(dataset, args):
    fanout = [int(x) for x in args.fanout.split(",")]
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats("csc")
    train_nid, val_nid, _ = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    print("training data percentage:", train_nid.numel() / g.num_nodes())

    sampler = ShaDowKHopSampler(fanout)
    dataloader = DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        device="cpu",
    )
    print("Start Pre-sample")
    blocks_pool = []
    save_graph_time = 0
    process = psutil.Process(os.getpid())
    before_mem = process.memory_info().rss / (1024 * 1024 * 1024)
    start = time.time()
    for it, (input_nodes, output_nodes, blocks) in tqdm(enumerate(dataloader), total=len(dataloader)):
        blocks_pool.append(blocks)
        if (it + 1) % 500 == 0 or it == (len(dataloader) - 1):
            tic = time.time()
            dgl.save_graphs(f"data/{args.dataset}_{it}.bin", blocks_pool)
            blocks_pool = []
            save_graph_time += time.time() - tic
    end = time.time()
    presample_mem = process.memory_info().rss / (1024 * 1024 * 1024) - before_mem
    print(f"Memory of pre-sampled sub-graphs: {presample_mem} GB")
    print(f"Time to sample one epoch (CPU, 0 worker): {end - start - save_graph_time} s")

    dataloader = DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        device="cpu",
    )
    print("Start Pre-sample")
    start = time.time()
    for it, (input_nodes, output_nodes, blocks) in tqdm(enumerate(dataloader), total=len(dataloader)):
        pass
    end = time.time()
    print(f"Time to sample one epoch (CPU, 2 worker): {end - start} s")

    train_nid = train_nid.to("cuda")
    dataloader = DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        device="cuda",
        use_uva=True,
    )
    torch.cuda.synchronize()
    start = time.time()
    for it, (input_nodes, output_nodes, blocks) in tqdm(enumerate(dataloader), total=len(dataloader)):
        pass
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time to sample one epoch (UVA): {end - start} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbn-products", help="dataset")
    parser.add_argument("--batchsize", type=int, default=1024, help="batch size")
    parser.add_argument("--fanout", type=str, default="10,10,10", help="sampling fanout")
    parser.add_argument("--path", type=str, default="/efs/rjliu/dataset/igb_large", help="path containing the datasets")
    parser.add_argument("--dataset_size", type=str, default="tiny", choices=["tiny", "small", "medium", "large", "full"], help="size of the datasets")
    parser.add_argument("--num_classes", type=int, default=19, choices=[19, 2983], help="number of classes")
    parser.add_argument("--in_memory", type=int, default=1, choices=[0, 1], help="0:read only mmap_mode=r, 1:load into memory")
    parser.add_argument("--synthetic", type=int, default=0, choices=[0, 1], help="0:nlp-node embeddings, 1:random")
    args = parser.parse_args()
    print(args)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    if args.dataset == "reddit":
        dataset = load_reddit()
    elif args.dataset.startswith("ogbn"):
        dataset = load_ogb(args.dataset, "/efs/rjliu/dataset")
    elif args.dataset.startswith("igb"):
        dataset = load_igb(args)
    elif args.dataset == "mag240m":
        dataset = load_mag240m("/efs/rjliu/dataset/mag240m_kddcup2021")
    else:
        raise NotImplementedError
    print(dataset[0])
    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Graph total memory:", mem1 - mem, "GB")
    train_dgl(dataset, args)
