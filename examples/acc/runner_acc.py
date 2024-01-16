import argparse
from offgs.dataset import OffgsDataset
import psutil
import os
import torch
import dgl
from train_offline import train as train_mini_batch


def get_predefined_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Training model device")
    parser.add_argument(
        "--batchsize", type=int, default=5000, help="batch size for training"
    )
    parser.add_argument("--dataset", default="ogbn-products", help="dataset")
    parser.add_argument(
        "--fanout", type=str, default="10,10,10", help="sampling fanout"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--model", type=str, default="SAGE", help="training model")
    parser.add_argument(
        "--dir", type=str, default="/nvme1n1/offgs_dataset", help="data path"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=10, help="numbers of epoch in training"
    )
    parser.add_argument("--log", type=str, default="../logs/train.csv", help="log file")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    print(args)
    return args


def process_args(args):
    if args.dataset == "igb-full":
        ratio = [0.1, 0.05]
    elif args.dataset == "mag240m":
        ratio = [1.0, 2.0]
    else:
        raise NotImplementedError

    # generate args
    for i in range(len(ratio)):
        yield {"ratio": ratio[i]}


def load_dataset(args):
    dataset_dir = f"{args.dir}/{args.dataset}-offgs"

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    dataset = OffgsDataset(dataset_dir)
    features = dataset.features.pin_memory()
    graph = dataset.graph

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem1 - mem, "GB")

    return dataset, graph, features


if __name__ == "__main__":
    args = get_predefined_args()
    dataset, graph, features = load_dataset(args)

    for inputs in process_args(args):
        for key, value in inputs.items():
            setattr(args, key, value)
        print(args)
        train_mini_batch(args, dataset, graph, features)
