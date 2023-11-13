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


def run(args, dataset, label_offset):
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"

    g, features, labels, n_classes, splitted_idx = dataset
    print("training nodes ratio:", splitted_idx["train"].numel() / g.num_nodes())

    os.makedirs(dataset_path, exist_ok=True)
    features_path = os.path.join(dataset_path, "features.npy")
    labels_path = os.path.join(dataset_path, "labels.npy")
    conf_path = os.path.join(dataset_path, "conf.json")
    split_idx_path = os.path.join(dataset_path, "split_idx.pth")

    print("Saving features...")
    features_mmap = np.memmap(features_path, mode="w+", shape=features.shape, dtype=np.float32)
    features_mmap[:] = features[:]
    features_mmap.flush()
    print("Done!")

    print("Saving labels...")
    labels = labels.type(torch.float32)
    np.save(labels_path, labels)
    print("Done!")

    print("Making conf file...")
    mmap_config = dict()
    mmap_config["num_nodes"] = int(g.num_nodes())
    mmap_config["features_shape"] = tuple(features_mmap.shape)
    mmap_config["features_dtype"] = str(features_mmap.dtype)
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
    parser.add_argument("--dataset", type=str, default="friendster", help="which dataset to load for training")
    parser.add_argument("--store-path", default="/nvme2n1", help="path to store subgraph")
    parser.add_argument("--path", type=str, default="/efs/rjliu/dataset/igb_full")
    parser.add_argument("--dataset_size", type=str, default="full", choices=["tiny", "small", "medium", "large", "full"])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--in_memory", type=int, default=0)
    parser.add_argument("--synthetic", type=int, default=0)
    args = parser.parse_args()
    print(args)

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    label_offset = 0
    if args.dataset.startswith("ogbn"):
        dataset = load_ogb(args.dataset, "/nvme1n1/dataset")
    elif args.dataset.startswith("igb"):
        dataset = load_igb(args)
    elif args.dataset == "mag240m":
        dataset = load_mag240m("/home/ubuntu/mag", only_graph=False)
        label_offset = dataset[-1]
        dataset = dataset[:-1]
    elif args.dataset == "friendster":
        dataset = load_dglgraph("/nvme1n1/dataset/friendster/friendster.bin", 128, 20)
    else:
        raise NotImplementedError

    print(dataset[0])
    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Graph total memory:", mem1 - mem, "GB")

    run(args, dataset, label_offset)
