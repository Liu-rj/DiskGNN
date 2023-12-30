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

import offgs

import numba
from numba.core import types
from numba.typed import Dict
import numpy as np
@numba.njit
def find_indices_in(a, b):
    d = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i, v in enumerate(b):
        d[v] = i
    ai = np.zeros_like(a)
    for i, v in enumerate(a):
        ai[i] = d.get(v, -1)
    return ai

def run(args, dataset, label_offset):
    fanout = [int(x) for x in args.fanout.split(",")]
    output_dir = f"{args.store_path}/{args.dataset}-{args.batchsize}-{args.fanout}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    g, features, labels, n_classes, splitted_idx = dataset
    g=g.remove_self_loop()
    g=g.add_self_loop()
    # g = g.formats("csc")
    train_nid, _, _ = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    sampler = NeighborSampler(fanout)
    train_dataloader = DataLoader(g, train_nid, sampler, batch_size=args.batchsize, shuffle=True, drop_last=False, num_workers=2)

    clear_cache_time = 0
    tic = time.time()
    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")
    clear_cache_time += time.time() - tic

    node_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device="cuda")
    start = time.time()
    for i, (input_nodes, output_nodes, blocks) in enumerate(tqdm(train_dataloader)):
        
        ##[Caution]: maybe poor efficiency.
        eids = torch.cat([block.edata[dgl.EID] for block in blocks])
        eids= torch.unique(eids)
        subgraph = dgl.edge_subgraph(g, eids.cpu(), relabel_nodes=True)
        rev_idx = find_indices_in(output_nodes.cpu().numpy(), subgraph.ndata[dgl.NID].numpy())
        rev_idx = torch.from_numpy(rev_idx).to('cuda')
        # subgraph.ndata.clear()
        subgraph.edata.clear()
        # subgraph.srcdata.clear()
        # subgraph.dstdata.clear()
        subgraph.train_idx=rev_idx
        for it, block in enumerate(blocks):
            block.ndata.clear()
            block.edata.clear()
            block.srcdata.clear()
            block.dstdata.clear()
            blocks[it] = block
        torch.save(subgraph, f"{output_dir}/subgraph_{i}.pt")
        torch.save(blocks, f"{output_dir}/train-{i}.pt")
        ## TODO use subgraph.ndata[dgl.NID] instead and test here to see if the result is the same
        torch.save(subgraph.ndata[dgl.NID], f"{output_dir}/in-nid-{i}.pt")
        torch.save(output_nodes, f"{output_dir}/out-nid-{i}.pt")
        node_counts[input_nodes.cuda()] += 1
    sampling_time = time.time() - start
    sorted_idx = torch.argsort(node_counts, descending=True).cpu()
    ## save nodecounts
    torch.save(node_counts, f"{output_dir}/node_counts.pt")
    torch.save(sorted_idx, f"{output_dir}/meta_node_popularity.pt")
    print(
        f"Drop Cache Time: {clear_cache_time:.3f}\t"
        f"Sampling Time: {sampling_time:.3f}\t"
        f"Total Time: {(time.time() - start - clear_cache_time):.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-products", help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=1000, help="batch size for training")
    parser.add_argument("--fanout", type=str, default="10,10,10", help="sampling fanout")
    parser.add_argument("--store-path", default="/nvme2n1", help="path to store subgraph")
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
        dataset = load_mag240m("/home/ubuntu/mag", only_graph=False)
        label_offset = dataset[-1]
        dataset = dataset[:-1]
    elif args.dataset == "friendster":
        dataset = load_dglgraph("/nvme1n1/dataset/friendster/friendster.bin", 0, 0)
    else:
        raise NotImplementedError

    print(dataset[0])
    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Graph total memory:", mem1 - mem, "GB")

    run(args, dataset, label_offset)