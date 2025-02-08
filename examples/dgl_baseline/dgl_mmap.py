import torch
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from offgs.utils import SAGE, GAT
from offgs.dataset import OffgsDataset
import threading
import queue
import psutil
import csv
import json
import os

import offgs


def train(args, dataset: OffgsDataset, subg_dir: str):
    dataset_path = f"{args.dir}/{args.dataset}-offgs"
    device = torch.device(f"cuda:{args.device}")
    fanout = [int(x) for x in args.fanout.split(",")]
    torch.cuda.set_device(device)

    labels = dataset.labels.pin_memory()
    label_offset = dataset.conf["label_offset"]

    if args.model == "SAGE":
        model = SAGE(
            dataset.num_features,
            args.hidden,
            dataset.num_classes,
            len(fanout),
            args.dropout,
        ).to(device)
    elif args.model == "GAT":
        model = GAT(
            dataset.num_features,
            args.hidden,
            dataset.num_classes,
            4,
            len(fanout),
            args.dropout,
        ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_num = dataset.split_idx["train"].numel()
    print(f"Label Ratio: {train_num / dataset.num_nodes}, Down Sample: {args.ratio}")
    pool_size = (int(train_num * args.ratio) + args.batchsize - 1) // args.batchsize

    mmap_features = dataset.mmap_features

    epoch_info_recorder = [[] for i in range(10)]
    for epoch in range(args.num_epoch):
        # with open("/proc/sys/vm/drop_caches", "w") as stream:
        #     stream.write("1\n")

        batch_id = torch.randperm(pool_size).tolist()

        tot_loss = 0
        info_recorder = [0] * 9
        torch.cuda.synchronize()

        model.train()
        for i in tqdm(batch_id, ncols=100):
            tic = time.time()
            blocks = torch.load(f"{subg_dir}/train-{i}.pt")
            input_nodes = torch.load(f"{subg_dir}/in-nid-{i}.pt")
            output_nodes = torch.load(f"{subg_dir}/out-nid-{i}.pt")
            info_recorder[0] += time.time() - tic  # graph load

            tic = time.time()
            feats = mmap_features[input_nodes]
            info_recorder[1] += time.time() - tic  # feature load

            tic = time.time()
            x = feats.to(device)
            torch.cuda.synchronize()
            info_recorder[2] += time.time() - tic  # feat transfer
            info_recorder[8] += x.shape[0]  # input node num

            tic = time.time()
            blocks = [block.to(device) for block in blocks]
            y = labels[output_nodes - label_offset].to(device).long()
            torch.cuda.synchronize()
            info_recorder[3] += time.time() - tic  # graph & label transfer

            tic = time.time()
            pred = model(blocks, x)
            loss = F.cross_entropy(pred, y)
            tot_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            info_recorder[5] += time.time() - tic

        print(
            f"Graph Load Time: {info_recorder[0]:.3f}\t"
            f"Feature Load Time: {info_recorder[1]:.3f}\t"
            f"Feat Transfer Time: {info_recorder[2]:.3f}\t"
            f"Graph Transfer Time : {info_recorder[3]:.3f}\t"
            f"Feat Transfer Time: {info_recorder[4]:.3f}\t"
            f"Train Time: {info_recorder[5]:.3f}\t"
            f"Sample init time: {info_recorder[6]:.3f}\t"
            f"Epoch Time: {np.sum(info_recorder[:7]):.3f}\t"
            f"Cold Feats Num: {info_recorder[7]}\t"
            f"Block Input Node Num: {info_recorder[8]}\t"
        )
        for i, info in enumerate(info_recorder):
            epoch_info_recorder[i].append(info)
        epoch_info_recorder[-1].append(np.sum(info_recorder[:7]))

    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.model,
            args.num_epoch,
        ]
        for epoch_info in epoch_info_recorder:
            log_info.append(round(np.mean(epoch_info), 2))
        writer.writerow(log_info)


def start(args):
    subg_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    subg_dir = os.path.join(args.dir, subg_dir)
    dataset_dir = f"{args.dir}/{args.dataset}-offgs"

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    dataset = OffgsDataset(dataset_dir)

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem1 - mem, "GB")

    train(args, dataset, subg_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--dataset", default="friendster")
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--model", type=str, default="SAGE")
    parser.add_argument("--dir", type=str, default="/nvme1n1/offgs_dataset")
    parser.add_argument("--num-epoch", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--log", type=str, default="dgl_on_disk.csv")
    args = parser.parse_args()
    print(args)

    start(args)
