"""
This script is a GraphBolt counterpart of
``/examples/core/rgcn/hetero_rgcn.py``. It demonstrates how to use GraphBolt
to train a R-GCN model for node classification on the Open Graph Benchmark
(OGB) dataset "ogbn-mag" and "ogb-lsc-mag240m". For more details on "ogbn-mag",
please refer to the OGB website: (https://ogb.stanford.edu/docs/linkprop/). For
more details on "ogb-lsc-mag240m", please refer to the OGB website:
(https://ogb.stanford.edu/docs/lsc/mag240m/).

Paper [Modeling Relational Data with Graph Convolutional Networks]
(https://arxiv.org/abs/1703.06103).

This example highlights the user experience of GraphBolt while the model and
training/evaluation procedures are almost identical to the original DGL
implementation. Please refer to original DGL implementation for more details.

This flowchart describes the main functional sequence of the provided example.
main
│
├───> load_dataset
│     │
│     └───> Load dataset
│
├───> rel_graph_embed [HIGHLIGHT]
│     │
│     └───> Generate graph embeddings
│
├───> Instantiate RGCN model
│     │
│     ├───> RelGraphConvLayer (input to hidden)
│     │
│     └───> RelGraphConvLayer (hidden to output)
│
└───> run
      │
      │
      └───> Training loop
            │
            ├───> EntityClassify.forward (RGCN model forward pass)
            │
            └───> validate and test
                  │
                  └───> EntityClassify.evaluate
"""

import argparse
import itertools
import sys
import time

import dgl
import dgl.graphbolt as gb
import dgl.nn as dglnn

import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroEmbedding
from ogb.lsc import MAG240MEvaluator
from ogb.nodeproppred import Evaluator
from tqdm import tqdm

import os
import csv


def load_dataset(dataset_name):
    """Load the dataset and return the graph, features, train/valid/test sets
    and the number of classes.

    Here, we use `BuiltInDataset` to load the dataset which returns graph,
    features, train/valid/test sets and the number of classes.
    """
    dataset = gb.BuiltinDataset(
        dataset_name, root="/nvme2n1/graphbolt_dataset/datasets"
    ).load()
    print(f"Loaded dataset: {dataset.tasks[0].metadata['name']}")

    graph = dataset.graph
    features = dataset.feature
    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    test_set = dataset.tasks[0].test_set
    num_classes = dataset.tasks[0].metadata["num_classes"]

    return (
        graph,
        features,
        train_set,
        valid_set,
        test_set,
        num_classes,
    )


def create_dataloader(
    name,
    graph,
    features,
    item_set,
    device,
    batch_size,
    fanouts,
    shuffle,
    num_workers,
):
    """Create a GraphBolt dataloader for training, validation or testing."""

    ###########################################################################
    # Initialize the ItemSampler to sample mini-batches from the dataset.
    # `item_set`:
    #   The set of items to sample from. This is typically the
    #   training, validation or test set.
    # `batch_size`:
    #   The number of nodes to sample in each mini-batch.
    # `shuffle`:
    #   Whether to shuffle the items in the dataset before sampling.
    datapipe = gb.ItemSampler(item_set, batch_size=batch_size, shuffle=shuffle)

    # Move the mini-batch to the appropriate device.
    # `device`:
    #   The device to move the mini-batch to.
    datapipe = datapipe.copy_to(device)

    # Sample neighbors for each seed node in the mini-batch.
    # `graph`:
    #   The graph(FusedCSCSamplingGraph) from which to sample neighbors.
    # `fanouts`:
    #   The number of neighbors to sample for each node in each layer.
    datapipe = datapipe.sample_neighbor(graph, fanouts=fanouts)

    # Fetch the features for each node in the mini-batch.
    # `features`:
    #   The feature store from which to fetch the features.
    # `node_feature_keys`:
    #   The node features to fetch. This is a dictionary where the keys are
    #   node types and the values are lists of feature names.

    # node_feature_keys = {"paper": ["feat"]}
    # if name == "ogb-lsc-mag240m":
    #     node_feature_keys["author"] = ["feat"]
    #     node_feature_keys["institution"] = ["feat"]
    # datapipe = datapipe.fetch_feature(features, node_feature_keys)

    # Create a DataLoader from the datapipe.
    # `num_workers`:
    #   The number of worker processes to use for data loading.
    return gb.DataLoader(datapipe, num_workers=num_workers)


def main(args):
    device = torch.device(f"cuda:{args.device}")
    fanout = [int(x) for x in args.fanout.split(",")]
    output_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    output_dir = os.path.join(args.store_path, output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Load dataset.
    (
        g,
        features,
        train_set,
        valid_set,
        test_set,
        num_classes,
    ) = load_dataset(args.dataset)

    print(g)

    category = "paper"

    num_etype = len(g.num_edges)
    data_loader = create_dataloader(
        args.dataset,
        g,
        features,
        train_set,
        device,
        batch_size=args.batchsize,
        fanouts=[
            torch.full((num_etype,), fanout[0]),
            torch.full((num_etype,), fanout[1]),
        ],
        shuffle=True,
        num_workers=args.num_workers,
    )

    loader_len = (train_set[:][category][0].numel() + args.batchsize - 1) // args.batchsize

    start = time.time()
    node_counts = {
        type: torch.zeros(num, dtype=torch.int64, device=device)
        for type, num in g.num_nodes.items()
    }
    for i, data in enumerate(
        tqdm(data_loader, desc="Sampling", total=loader_len, ncols=100)
    ):
        # Convert MiniBatch to DGL Blocks and move them to the target
        # device.
        blocks = [block for block in data.blocks]

        for it, block in enumerate(blocks):
            block.ndata.clear()
            block.edata.clear()
            block.srcdata.clear()
            block.dstdata.clear()
            blocks[it] = block.cpu()

        # Fetch the input and output nodes in the batch.
        input_nodes = data.input_nodes
        # iterate over the input nodes as a dict
        for key, value in input_nodes.items():
            input_nodes[key] = value.cpu()
        output_nodes = data.seed_nodes
        for key, value in output_nodes.items():
            output_nodes[key] = value.cpu()

        torch.save(blocks, f"{output_dir}/train-{i}.pt")
        torch.save(input_nodes, f"{output_dir}/in-nid-{i}.pt")
        torch.save(output_nodes, f"{output_dir}/out-nid-{i}.pt")
        for key, value in input_nodes.items():
            node_counts[key][value.to(device)] += 1

    all_node_counts = [None] * len(list(g.num_nodes))
    for key, idx in g.node_type_to_id.items():
        all_node_counts[idx] = node_counts[key]
    all_node_counts = torch.cat(all_node_counts, dim=0)
    sorted_idx = torch.argsort(all_node_counts, descending=True)

    ## save nodecounts
    torch.save(g.node_type_to_id, f"{output_dir}/node_type_to_id.pt")
    torch.save(g.node_type_offset, f"{output_dir}/node_type_offset.pt")
    torch.save(all_node_counts.cpu(), f"{output_dir}/node_counts.pt")
    torch.save(sorted_idx.cpu(), f"{output_dir}/meta_node_popularity.pt")
    
    end = time.time()
    sample_time = end - start
    print(f"Sampling time: {sample_time:.2f}s")
    
    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.ratio,
            args.num_workers,
            round(sample_time, 2),
        ]
        writer.writerow(log_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphBolt RGCN")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-mag",
        choices=["ogbn-mag", "ogb-lsc-mag240m"],
        help="Dataset name. Possible values: ogbn-mag, ogb-lsc-mag240m",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--fanout", type=str, default="25,10")
    parser.add_argument("--store-path", default="/nvme2n1")
    parser.add_argument("--log", type=str, default="logs/sample_decompose.csv")
    parser.add_argument("--ratio", type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    main(args)
