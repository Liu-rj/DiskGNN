import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from load_graph import *
import psutil
import time
import json
from offgs.dataset import OffgsDataset
import csv
import dgl.graphbolt as gb

import offgs


def create_dataloader(graph, itemset, batch_size, fanout, device, num_workers, job):
    """
    [HIGHLIGHT]
    Get a GraphBolt version of a dataloader for node classification tasks.
    This function demonstrates how to utilize functional forms of datapipes in
    GraphBolt. For a more detailed tutorial, please read the examples in
    `dgl/notebooks/graphbolt/walkthrough.ipynb`.
    Alternatively, you can create a datapipe using its class constructor.

    Parameters
    ----------
    job : one of ["train", "evaluate", "infer"]
        The stage where dataloader is created, with options "train", "evaluate"
        and "infer".
    Other parameters are explicated in the comments below.
    """

    ############################################################################
    # [Step-1]:
    # gb.ItemSampler()
    # [Input]:
    # 'itemset': The current dataset. (e.g. `train_set` or `valid_set`)
    # 'batch_size': Specify the number of samples to be processed together,
    # referred to as a 'mini-batch'. (The term 'mini-batch' is used here to
    # indicate a subset of the entire dataset that is processed together. This
    # is in contrast to processing the entire dataset, known as a 'full batch'.)
    # 'job': Determines whether data should be shuffled. (Shuffling is
    # generally used only in training to improve model generalization. It's
    # not used in validation and testing as the focus there is to evaluate
    # performance rather than to learn from the data.)
    # [Output]:
    # An ItemSampler object for handling mini-batch sampling.
    # [Role]:
    # Initialize the ItemSampler to sample mini-batche from the dataset.
    ############################################################################
    datapipe = gb.ItemSampler(itemset, batch_size=batch_size, shuffle=(job == "train"))

    ############################################################################
    # [Step-2]:
    # self.copy_to()
    # [Input]:
    # 'device': The device to copy the data to.
    # [Output]:
    # A CopyTo object to copy the data to the specified device. Copying here
    # ensures that the rest of the operations run on the GPU.
    ############################################################################
    datapipe = datapipe.copy_to(device=device)

    ############################################################################
    # [Step-3]:
    # self.sample_neighbor()
    # [Input]:
    # 'graph': The network topology for sampling.
    # '[-1] or fanout': Number of neighbors to sample per node. In
    # training or validation, the length of `fanout` should be equal to the
    # number of layers in the model. In inference, this parameter is set to
    # [-1], indicating that all neighbors of a node are sampled.
    # [Output]:
    # A NeighborSampler object to sample neighbors.
    # [Role]:
    # Initialize a neighbor sampler for sampling the neighborhoods of nodes.
    ############################################################################
    datapipe = datapipe.sample_neighbor(graph, fanouts=fanout)

    ############################################################################
    # [Step-4]:
    # self.fetch_feature()
    # [Input]:
    # 'features': The node features.
    # 'node_feature_keys': The keys of the node features to be fetched.
    # [Output]:
    # A FeatureFetcher object to fetch node features.
    # [Role]:
    # Initialize a feature fetcher for fetching features of the sampled
    # subgraphs.
    ############################################################################
    # datapipe = datapipe.fetch_feature(features, node_feature_keys=["feat"])

    ############################################################################
    # [Step-5]:
    # self.copy_to()
    # [Input]:
    # 'device': The device to copy the data to.
    # [Output]:
    # A CopyTo object to copy the data to the specified device.
    ############################################################################
    # datapipe = datapipe.copy_to(device=device)

    ############################################################################
    # [Step-6]:
    # gb.DataLoader()
    # [Input]:
    # 'datapipe': The datapipe object to be used for data loading.
    # 'num_workers': The number of processes to be used for data loading.
    # [Output]:
    # A DataLoader object to handle data loading.
    # [Role]:
    # Initialize a multi-process dataloader to load the data in parallel.
    ############################################################################
    dataloader = gb.DataLoader(datapipe, num_workers=num_workers)

    # Return the fully-initialized DataLoader object.
    return dataloader


def run(args, dataset: OffgsDataset):
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    fanout = [int(x) for x in args.fanout.split(",")]
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    output_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    output_dir = os.path.join(args.store_path, output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    g = dataset.graph
    g.pin_memory_()
    train_nid = (
        dataset.split_idx["train"]
        if args.ratio == 1.0
        else torch.load(f"{dataset_path}/train_idx_{args.ratio}.pt")
    )
    print(f"Train Node Ratio: {train_nid.numel() / g.num_nodes}")
    train_set = gb.ItemSet(train_nid, "seed_nodes")

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory consumption:", mem1 - mem, "GB")

    # train_nid = dataset.split_idx["train"]
    # int_part, dec_part2 = int(args.ratio), args.ratio - int(args.ratio)
    # dec_size = int(dec_part2 * train_nid.numel())
    # perm_idx = [torch.randperm(train_nid.numel()) for i in range(int_part)]
    # perm_idx.append(torch.randperm(train_nid.numel())[:dec_size])
    # perm_idx = torch.cat(perm_idx, dim=0)
    # print(
    #     f"Subsampled {perm_idx.numel()} nodes from {train_nid.numel()} nodes,",
    #     f"Down Sample Ratio: {perm_idx.numel() / train_nid.numel()}",
    #     f"Train Node Ratio: {perm_idx.numel() / g.num_nodes}",
    # )
    # train_nid = train_nid[perm_idx]  # subsample
    # torch.save(train_nid, f"{dataset_path}/train_idx_{args.ratio}.pt")

    dataloader = create_dataloader(
        graph=g,
        itemset=train_set,
        batch_size=args.batchsize,
        fanout=fanout,
        device=args.device,
        num_workers=args.num_workers,
        job="train",
    )

    # with open("/proc/sys/vm/drop_caches", "w") as stream:
    #     stream.write("1\n")

    len = (train_nid.numel() + args.batchsize - 1) // args.batchsize
    node_counts = torch.zeros(g.num_nodes, dtype=torch.int64, device=device)
    sample_time, save_block, save_rank = 0, 0, 0
    tic = time.time()
    for i, data in enumerate(tqdm(dataloader, total=len, ncols=100)):
        blocks = [block for block in data.blocks]

        for it, block in enumerate(blocks):
            block.ndata.clear()
            block.edata.clear()
            block.srcdata.clear()
            block.dstdata.clear()
            blocks[it] = block.cpu()
        sample_time += time.time() - tic

        input_nodes = data.input_nodes.long()
        output_nodes = data.seed_nodes.long()

        tic = time.time()
        torch.save(blocks, f"{output_dir}/train-{i}.pt")
        torch.save(input_nodes.cpu(), f"{output_dir}/in-nid-{i}.pt")
        torch.save(output_nodes.cpu(), f"{output_dir}/out-nid-{i}.pt")
        node_counts[input_nodes] += 1
        save_block += time.time() - tic

        tic = time.time()

    tic = time.time()
    sorted_idx = torch.argsort(node_counts, descending=True)
    ## save nodecounts
    torch.save(node_counts.cpu(), f"{output_dir}/node_counts.pt")
    torch.save(sorted_idx.cpu(), f"{output_dir}/meta_node_popularity.pt")
    save_rank += time.time() - tic

    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.ratio,
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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--store-path", default="/nvme2n1")
    parser.add_argument("--log", type=str, default="logs/sample_decompose.csv")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    print(args)

    # --- load dataset --- #
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    dataset = OffgsDataset(dataset_path)

    run(args, dataset)
