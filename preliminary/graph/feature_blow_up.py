import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from load_graph import *
from model import *

import offgs


def testbench(dataset, args):
    device = "cuda:0"
    # fanout = [int(x) for x in args.fanout.split(",")]
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats("csc")
    train_nid, _, _ = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    g = g.to("cpu") if args.use_uva else g.to(device)
    train_nid = train_nid.to(device)
    # sampler = NeighborSampler(fanout)
    # train_dataloader = DataLoader(
    #     g,
    #     train_nid,
    #     sampler,
    #     batch_size=args.batchsize,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=0,
    #     device=torch.device(device),
    #     use_uva=args.use_uva,
    # )
    # static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    # print("memory allocated before training:", static_memory, "GB")

    # for epoch in range(args.num_epoch):
    #     for step, (input_nodes, output_nodes, blocks) in enumerate(
    #         tqdm(train_dataloader)
    #     ):
    #         sampled_counts[input_nodes] += 1
    # idx = torch.argsort(sampled_counts, descending=True)
    # cache_feat_idx = idx[: idx.shape[0] // 5]

    degrees = g.in_degrees()
    idx = torch.argsort(degrees, descending=True)
    pre_sampled_idx = idx[: idx.shape[0] // 5]
    if args.use_uva:
        g.pin_memory_()
    dataloader = torch.utils.data.DataLoader(
        pre_sampled_idx, batch_size=51200, shuffle=False, drop_last=False
    )
    fanout_list = [10, 15, 30, 50]
    cache_percent_list = [0.1, 0.2, 0.3, 0.4]
    for fanout in fanout_list:
        sampled_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device=device)
        for j, seeds in tqdm(enumerate(dataloader)):
            subg = g.sample_neighbors(seeds.cuda(), fanout, replace=False)
            src, dst = subg.edges()

            unique_res, counts = torch.unique(src, return_counts=True)
            sampled_counts[unique_res] += counts

            # need_feat_idx = torch.ops.offgs._CAPI_Difference(src, cache_feat_idx)
            # all_counts += need_feat_idx.shape[0]
        ordered_counts, idx = torch.sort(sampled_counts, dim=0, descending=True)
        for cache_percent in cache_percent_list:
            print(
                f"Fanout: {fanout}, cache percent.: {cache_percent}, feature percent.:",
                torch.sum(
                    ordered_counts[int(ordered_counts.shape[0] * cache_percent) :]
                ).item()
                / g.num_nodes(),
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
    parser.add_argument("--fanout", type=int, default=10, help="sampling fanout")
    args = parser.parse_args()
    print(args)
    if args.dataset.startswith("ogbn"):
        dataset = load_ogb(args.dataset, "/home/ubuntu/dataset")
    elif args.dataset == "friendster":
        dataset = load_dglgraph("/home/ubuntu/dataset/friendster/friendster.bin")
    else:
        raise NotImplementedError
    print(dataset[0])
    testbench(dataset, args)
