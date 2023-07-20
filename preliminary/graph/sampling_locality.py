import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from load_graph import *


def testbench(dataset, args):
    device = "cuda:2"
    fanout = [int(x) for x in args.fanout.split(",")]
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats("csc")
    train_nid, _, _ = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    g = g.to("cpu") if args.use_uva else g.to(device)
    train_nid = train_nid.to(device)
    print("training set percent:", train_nid.numel() / g.num_nodes())

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
    static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    print("memory allocated before training:", static_memory, "GB")

    dataloader = torch.utils.data.DataLoader(
        train_nid, batch_size=args.batchsize, shuffle=True, drop_last=False
    )
    if args.use_uva:
        g.pin_memory_()
    frontier_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device=device)
    for epoch in range(args.num_epoch):
        for step, seeds in enumerate(tqdm(dataloader)):
            for k in reversed(fanout):
                frontier_counts[seeds] += 1
                subg = g.sample_neighbors(seeds, k, replace=False)
                src, dst = subg.edges()
                seeds = torch.unique(src)
            # subg = g.sample_neighbors(seeds, 10, replace=False)
            # src, dst = subg.edges()
            # unique_res, counts = torch.unique(src, return_counts=True)
            # sampled_counts[unique_res] += counts
    all_counts = torch.sum(frontier_counts)
    print("all counts:", all_counts)

    ordered_counts, idx = torch.sort(frontier_counts, dim=0, descending=True)
    b = (torch.cumsum(ordered_counts, dim=0) / all_counts).cpu().detach().numpy()

    font_size = 10
    plt.plot(np.arange(b.shape[0]) / g.num_nodes(), b)
    plt.xticks([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4], minor=True)
    plt.xlabel(
        "proportion of nodes in descending order of counts",
        fontsize=font_size,
        fontweight="bold",
    )
    plt.ylabel(
        "proportion of accumulated counts",
        fontsize=font_size,
        fontweight="bold",
    )
    plt.title(
        args.dataset + ": the number of times being treated as frontier",
        fontsize=font_size,
        fontweight="bold",
    )
    plt.grid(which="major", linestyle="-.")
    plt.grid(which="minor", linestyle="-.")
    plt.savefig(
        "imgs/"
        + args.dataset
        + f"-sampling-locality-{args.fanout.replace(',','-')}-being-frontier.png",
        bbox_inches="tight",
    )
    plt.clf()

    g.unpin_memory_()
    idx = idx.cpu()
    hot_node_percent_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    cache_percent_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    results = []
    for percent in hot_node_percent_list:
        pre_sampled_idx = idx[: int(g.num_nodes() * percent)]
        src = g.in_edges(pre_sampled_idx)[0]
        unique_res, counts = torch.unique(src, return_counts=True)
        ordered_counts, _ = torch.sort(counts, dim=0, descending=True)
        result = []
        for cache_percent in cache_percent_list:
            res = torch.sum(ordered_counts[int(g.num_nodes() * cache_percent) :])
            res = res.item() / g.num_nodes()
            result.append(res)
            print(
                f"pre-sample node percent: {percent}, cache percent: {cache_percent}, feature percent: {res}"
            )
        results.append(result)

    font_size = 20
    marker_size = 10
    x = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    markers = ["o", "v", "1", "s", "p", "|"]
    plt.figure(figsize=(15, 8))
    for it, result in enumerate(results):
        plt.plot(
            x,
            result,
            label=f"pre-sample {cache_percent_list[it] * 100}% hot nodes",
            marker=markers[it],
            markersize=marker_size,
        )
    plt.grid()
    plt.xticks(x, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylim(-0.1, 3)
    plt.ylabel("packed node feature / #nodes", fontsize=font_size)
    plt.xlabel("cache percentage", fontsize=font_size)
    plt.title(
        f"Cache effect on node feature redundancy ({args.dataset})", fontsize=font_size
    )
    plt.legend(fontsize=font_size)
    plt.savefig(
        f"imgs/feature_redundancy_hotnodes_allsample_{args.dataset}.png",
        bbox_inches="tight",
    )
    plt.show()

    # sampled_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device=device)
    # for epoch in range(args.num_epoch):
    #     for step, (input_nodes, output_nodes, blocks) in enumerate(
    #         tqdm(train_dataloader)
    #     ):
    #         # sampled_nodes = input_nodes[output_nodes.numel() :]
    #         # unique_res, counts = torch.unique(sampled_nodes, return_counts=True)
    #         sampled_counts[input_nodes] += 1
    # all_counts = torch.sum(sampled_counts)

    # a, idx = torch.sort(sampled_counts, dim=0, descending=True)
    # b = (torch.cumsum(a, dim=0) / all_counts).cpu().detach().numpy()
    # font_size = 10
    # plt.plot(np.arange(b.shape[0]) / g.num_nodes(), b)
    # plt.xlabel(
    #     "proportion of nodes in descending order of counts",
    #     fontsize=font_size,
    #     fontweight="bold",
    # )
    # plt.ylabel(
    #     "proportion of accumulated counts",
    #     fontsize=font_size,
    #     fontweight="bold",
    # )
    # plt.title(
    #     args.dataset + ": the number of times being sampling",
    #     fontsize=font_size,
    #     fontweight="bold",
    # )
    # plt.grid(linestyle="-.")
    # plt.savefig(
    #     args.dataset + f"-sampling-frequency-{args.fanout.replace(',','-')}.pdf",
    #     bbox_inches="tight",
    # )
    # plt.clf()

    # g = g.formats("csr")
    # a = g.out_degrees().to(device)[idx]
    # b = (torch.cumsum(a, dim=0) / g.num_edges()).cpu().detach().numpy()
    # font_size = 10
    # plt.plot(np.arange(b.shape[0]) / g.num_nodes(), b)
    # plt.xlabel(
    #     "proportion of nodes in descending order of counts",
    #     fontsize=font_size,
    #     fontweight="bold",
    # )
    # plt.ylabel(
    #     "proportion of accumulated out-degrees",
    #     fontsize=font_size,
    #     fontweight="bold",
    # )
    # plt.title(
    #     args.dataset + ": accumulated out-degress in sampling priority",
    #     fontsize=font_size,
    #     fontweight="bold",
    # )
    # plt.grid(linestyle="-.")
    # plt.savefig(
    #     args.dataset + f"-sampling-outdegrees-{args.fanout.replace(',','-')}.pdf",
    #     bbox_inches="tight",
    # )
    # plt.clf()


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
