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
import dgl.graphbolt as gb


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
    features = dataset.feature.read("node", None, "feat")
    labels = dataset.feature.read("node", None, "label").flatten()
    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    test_set = dataset.tasks[0].test_set
    num_classes = dataset.tasks[0].metadata["num_classes"]

    # labels = torch.full((graph.num_nodes,), -1, dtype=torch.int64)
    splitted_idx = {"train": {}, "test": {}, "valid": {}}
    for it, node_set in enumerate([train_set, valid_set, test_set]):
        nid, label = node_set[:]
        set_type = ["train", "valid", "test"][it]
        splitted_idx[set_type] = nid
        # labels[nid] = label

    return (
        graph,
        features,
        labels,
        num_classes,
        splitted_idx,
    )


def run(args, dataset, label_offset):
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"

    g, features, labels, n_classes, splitted_idx = dataset
    # print(f"features nan value: {torch.isnan(features).sum()}")
    print(f"features dtype: {features.dtype}")
    print(f"features shape: {features.shape}")
    print("training nodes ratio:", splitted_idx["train"].numel() / g.num_nodes)

    os.makedirs(dataset_path, exist_ok=True)
    graph_path = os.path.join(dataset_path, "graph.pth")
    features_path = os.path.join(dataset_path, "features.bin")
    labels_path = os.path.join(dataset_path, "labels.pth")
    conf_path = os.path.join(dataset_path, "conf.json")
    split_idx_path = os.path.join(dataset_path, "split_idx.pth")

    print("Saving graph...")
    # g = g.formats("coo")
    torch.save(g, graph_path)
    print("Done!")

    print("Saving features...")
    features.numpy().tofile(features_path)
    print("Done!")

    print("Saving labels...")
    # labels = labels.type(torch.float32)
    torch.save(labels, labels_path)
    print("Done!")

    print("Making conf file...")
    mmap_config = dict()
    mmap_config["total_num_nodes"] = g.total_num_nodes
    mmap_config["num_nodes"] = int(g.num_nodes)
    mmap_config["features_shape"] = tuple(features.shape)
    mmap_config["features_dtype"] = str(features.dtype)
    mmap_config["feat_itemsize"] = features.element_size()
    mmap_config["labels_shape"] = tuple(labels.shape)
    mmap_config["labels_dtype"] = str(labels.dtype)
    mmap_config["num_classes"] = int(n_classes)
    mmap_config["label_offset"] = int(label_offset)
    json.dump(mmap_config, open(conf_path, "w"))
    print("Done!")

    print("Saving split index...")
    torch.save(splitted_idx, split_idx_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="igb-tiny")
    parser.add_argument("--store-path", type=str, default="/nvme1n1/offgs_dataset")
    parser.add_argument("--path", type=str, default="/efs/rjliu/dataset/igb_dataset")
    parser.add_argument("--dataset_size", type=str, default="tiny")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--in_memory", type=int, default=0)
    parser.add_argument("--synthetic", type=int, default=0)
    args = parser.parse_args()
    print(args)

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    label_offset = 0
    # if args.dataset.startswith("ogbn"):
    #     dataset = load_ogb(args.dataset, "/efs/rjliu/dataset")
    # elif args.dataset.startswith("igb"):
    #     dataset = load_igb(args)
    # elif args.dataset == "mag240m":
    #     dataset = load_mag240m("/efs/rjliu/dataset/mag240m", only_graph=False)
    #     label_offset = dataset[-1]
    #     dataset = dataset[:-1]
    # elif args.dataset == "friendster":
    #     dataset = load_friendster("/efs/rjliu/dataset/friendster", 128, 20)
    # else:
    #     raise NotImplementedError
    dataset = load_dataset(args.dataset)

    print(dataset[0])
    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Graph total memory:", mem1 - mem, "GB")

    run(args, dataset, label_offset)
