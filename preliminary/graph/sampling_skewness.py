import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from load_graph import *
from model import *


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
    train_nid = train_nid.to(device)
    if args.dataset == "friendster":
        idx = torch.randperm(train_nid.shape[0], device=device)
        train_nid = train_nid[idx[: int(train_nid.shape[0] / 10)]]
    sampler = NeighborSampler(fanout)
    train_dataloader = DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        device=torch.device(device),
        use_uva=args.use_uva,
    )
    static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    print("memory allocated before training:", static_memory, "GB")

    sampled_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device=device)
    for epoch in range(args.num_epoch):
        for step, (input_nodes, output_nodes, blocks) in enumerate(
            tqdm(train_dataloader)
        ):
            sampled_nodes = input_nodes[output_nodes.numel() :]
            unique_res, counts = torch.unique(sampled_nodes, return_counts=True)
            sampled_counts[unique_res] += counts
    all_counts = torch.sum(sampled_counts)

    a, idx = torch.sort(sampled_counts, dim=0, descending=True)
    b = (torch.cumsum(a, dim=0) / all_counts).cpu().detach().numpy()
    font_size = 10
    plt.plot(np.arange(b.shape[0]) / g.num_nodes(), b)
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
        args.dataset + ": the number of times being sampling",
        fontsize=font_size,
        fontweight="bold",
    )
    plt.grid(linestyle="-.")
    plt.savefig(
        args.dataset + f"-sampling-frequency-{args.fanout.replace(',','-')}.pdf",
        bbox_inches="tight",
    )
    plt.clf()

    g = g.formats("csr")
    a = g.out_degrees().to(device)[idx]
    b = (torch.cumsum(a, dim=0) / g.num_edges()).cpu().detach().numpy()
    font_size = 10
    plt.plot(np.arange(b.shape[0]) / g.num_nodes(), b)
    plt.xlabel(
        "proportion of nodes in descending order of counts",
        fontsize=font_size,
        fontweight="bold",
    )
    plt.ylabel(
        "proportion of accumulated out-degrees",
        fontsize=font_size,
        fontweight="bold",
    )
    plt.title(
        args.dataset + ": accumulated out-degress in sampling priority",
        fontsize=font_size,
        fontweight="bold",
    )
    plt.grid(linestyle="-.")
    plt.savefig(
        args.dataset + f"-sampling-outdegrees-{args.fanout.replace(',','-')}.pdf",
        bbox_inches="tight",
    )
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="ogbn-products", help="which dataset to load for training"
    )
    parser.add_argument(
        "--batchsize", type=int, default=512, help="batch size for training"
    )
    parser.add_argument(
        "--use-uva", type=bool, default=False, help="use uva for sampling or not"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=10, help="numbers of epoch in training"
    )
    parser.add_argument(
        "--fanout", type=str, default="10,10,10", help="sampling fanout"
    )
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
