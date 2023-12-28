import torch
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
from ctypes import *
from ctypes.util import *
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
import pandas as pd
from load_graph import *
from model import *
from dataset import OffgsDataset
from queue import Queue
import threading
import psutil
import csv
from dgl.utils import gather_pinned_tensor_rows
import offgs


def train(
    args,
    dataset: OffgsDataset,
    address_table: torch.Tensor,
    cpu_cached_feats: torch.Tensor,
    gpu_cached_feats: torch.Tensor,
    subg_dir: str,
    aux_dir: str,
):
    device = torch.device(f"cuda:{args.device}")
    fanout = [int(x) for x in args.fanout.split(",")]

    labels = dataset.labels.pin_memory()
    cpu_cached_feats = cpu_cached_feats.pin_memory()

    if args.model == "SAGE":
        model = SAGE(dataset.num_features, 256, dataset.num_classes, len(fanout)).to(
            device
        )
    elif args.model == "GAT":
        model = GAT(dataset.num_features, 256, dataset.num_classes, [8, 2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    size = (dataset.split_idx["train"].numel() + args.batchsize - 1) // args.batchsize

    epoch_info_recorder = [[] for i in range(11)]
    for epoch in range(args.num_epoch):
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")

        info_recorder = [0] * 10
        subgraph_sampler_init_time = 0
        sampler = NeighborSampler([10, 10, 10])

        torch.cuda.synchronize()
        start = time.time()

        model.train()
        for i in trange(size, ncols=100):
            # tic = time.time()
            # with open("/proc/sys/vm/drop_caches", "w") as stream:
            #     stream.write("1\n")
            # clear_cache += time.time() - tic

            tic = time.time()
            if args.mega_batch == True:
                subgraph = torch.load(f"{subg_dir}/subgraph_{i}.pt")
            else:
                blocks = torch.load(f"{subg_dir}/train-{i}.pt")
            output_nodes = torch.load(f"{subg_dir}/out-nid-{i}.pt")
            input_nodes = torch.load(f"{subg_dir}/in-nid-{i}.pt")
            info_recorder[0] += time.time() - tic  # graph load
            tic = time.time()
            (
                cold_nodes,
                hot_nodes,
                rev_hot_idx,
                rev_cold_idx,
            ) = torch.load(f"{aux_dir}/train-aux-meta-{i}.pt")
            if cold_nodes.numel() > 0:
                cold_feats = torch.ops.offgs._CAPI_LoadFeats_Direct(
                    f"{aux_dir}/train-aux-{i}.npy",
                    cold_nodes.numel(),
                    dataset.num_features,
                )
            info_recorder[1] += time.time() - tic  # feature load
            info_recorder[6] += cold_nodes.numel()  # cold_feats_num

            if args.mega_batch == True:
                tic = time.time()
                global_nid = input_nodes.to(device)
                cold_idx_map = torch.tensor([])
                if cold_nodes.numel() > 0:
                    cold_idx_map = torch.full(
                        (subgraph.num_nodes(),),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    )
                    cold_idx_map[rev_cold_idx] = torch.arange(
                        cold_nodes.numel(),
                        dtype=torch.int32,
                        device=device,
                    )
                torch.cuda.synchronize()
                info_recorder[2] += time.time() - tic  # feat assemble

                tic = time.time()
                rev_idx = subgraph.train_idx.to(device)
                sub_train_dataloader = DataLoader(
                    subgraph,
                    rev_idx,
                    sampler,
                    device=torch.device("cuda"),
                    batch_size=1024,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0,
                    use_uva=True,
                )
                torch.cuda.synchronize()
                subgraph_sampler_init_time += time.time() - tic
                ## may need to cal sample time here modify code!
                sample_begin_time = time.time()

                for it, (input_nodes, output_nodes, blocks) in enumerate(
                    sub_train_dataloader
                ):
                    y = gather_pinned_tensor_rows(
                        labels,
                        global_nid[output_nodes],
                    ).long()
                    torch.cuda.synchronize()
                    # sample, graph and label transfer
                    info_recorder[3] += time.time() - sample_begin_time

                    tic = time.time()
                    x = torch.empty(
                        (input_nodes.numel(), dataset.num_features),
                        dtype=torch.float32,
                        device=device,
                    )
                    torch.ops.offgs._CAPI_GatherInGPU_MegaBatch(
                        x,
                        input_nodes,
                        global_nid,
                        cpu_cached_feats,
                        gpu_cached_feats,
                        cold_feats,
                        address_table,
                        cold_idx_map,
                    )
                    torch.cuda.synchronize()
                    info_recorder[2] += time.time() - tic  # feat assemble
                    info_recorder[7] += x.shape[0]  # input node num

                    tic = time.time()
                    y_hat = model(blocks, x)
                    ## cal the acc
                    loss = F.cross_entropy(y_hat, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    torch.cuda.synchronize()
                    info_recorder[5] += time.time() - tic

                    sample_begin_time = time.time()
            else:
                tic = time.time()
                num_input = cold_nodes.numel() + hot_nodes.numel()
                x = torch.empty(
                    (num_input, dataset.num_features),
                    dtype=torch.float32,
                    device=device,
                )
                if cold_nodes.numel() > 0:
                    x[rev_cold_idx] = cold_feats.to(device)
                torch.ops.offgs._CAPI_GatherInGPU(
                    x,
                    rev_hot_idx,
                    cpu_cached_feats,
                    gpu_cached_feats,
                    hot_nodes,
                    address_table,
                )
                torch.cuda.synchronize()
                info_recorder[2] += time.time() - tic  # feat assemble
                info_recorder[7] += x.shape[0]  # input node num

                tic = time.time()
                blocks = [block.to(device) for block in blocks]
                y = labels[output_nodes].to(device).long()
                torch.cuda.synchronize()
                info_recorder[3] += time.time() - tic  # graph & label transfer

                tic = time.time()
                pred = model(blocks, x)
                loss = F.cross_entropy(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                torch.cuda.synchronize()
                info_recorder[5] += time.time() - tic

        print(
            f"Graph Load Time: {info_recorder[0]:.3f}\t"
            f"Feature Load Time: {info_recorder[1]:.3f}\t"
            f"Feat Assemble Time: {info_recorder[2]:.3f}\t"
            f"Sample and Graph Transfer Time : {info_recorder[3]:.3f}\t"
            f"Feat Transfer Time: {info_recorder[4]:.3f}\t"
            f"Train Time: {info_recorder[5]:.3f}\t"
            f"Epoch Time: {np.sum(info_recorder[:6]):.3f}\t"
            f"Cold Feats Num: {info_recorder[6]}\t"
            f"Feature Transfer Num: {info_recorder[7]}\t"
        )
        for i, info in enumerate(info_recorder):
            epoch_info_recorder[i].append(info)
        epoch_info_recorder[-1].append(np.sum(info_recorder[:6]))

    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.cpu_cache_size,
            args.gpu_cache_size,
            round(args.cpu_cache_ratio, 2),
            round(args.gpu_cache_ratio, 2),
            args.model,
            args.num_epoch,
        ]
        for epoch_info in epoch_info_recorder:
            log_info.append(round(np.mean(epoch_info[1:]), 2))
        writer.writerow(log_info)


def init_cache(args, dataset, cached_nodes):
    device = torch.device(f"cuda:{args.device}")
    table_size = 4 * dataset.num_nodes
    print(f"Adress Table Size: {table_size / (1024 * 1024 * 1024)} GB")
    gpu_idx_cache_size = max(0, args.gpu_cache_size - table_size)
    gpu_num_entries = gpu_idx_cache_size // (4 * dataset.num_features)
    gpu_cached_idx = cached_nodes[:gpu_num_entries]
    cpu_cache_idx = cached_nodes[gpu_num_entries:]
    gpu_cached_feats = dataset.mmap_features[gpu_cached_idx].to(device)
    cpu_cached_feats = torch.empty(
        (cpu_cache_idx.numel(), dataset.num_features), dtype=torch.float32
    )
    cpu_cached_feats[:] = dataset.mmap_features[cpu_cache_idx]
    address_table = torch.zeros((dataset.num_nodes,), dtype=torch.int32)
    address_table[cpu_cache_idx] = (
        torch.arange(cpu_cache_idx.numel(), dtype=torch.int32) + 1
    )
    address_table[gpu_cached_idx] = -(
        torch.arange(gpu_cached_idx.numel(), dtype=torch.int32) + 1
    )
    address_table = address_table.to(device)
    return cpu_cached_feats, gpu_cached_feats, address_table


def start(args):
    total_cache_size = args.cpu_cache_size + args.gpu_cache_size

    subg_dir = f"{args.dir}/{args.dataset}-{args.batchsize}-{args.fanout}"
    aux_dir = f"{subg_dir}/cache-size-{total_cache_size}"
    dataset_dir = f"{args.dir}/{args.dataset}-offgs"

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    dataset = OffgsDataset(dataset_dir)
    cached_nodes = torch.load(f"{aux_dir}/cached_nodes.pt")

    (
        cpu_cached_feats,
        gpu_cached_feats,
        address_table,
    ) = init_cache(args, dataset, cached_nodes)

    args.cpu_cache_ratio = cpu_cached_feats.shape[0] / dataset.num_nodes
    args.gpu_cache_ratio = gpu_cached_feats.shape[0] / dataset.num_nodes

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem1 - mem, "GB")

    train(
        args,
        dataset,
        address_table,
        cpu_cached_feats,
        gpu_cached_feats,
        subg_dir,
        aux_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Training model device")
    parser.add_argument(
        "--batchsize", type=int, default=5000, help="batch size for training"
    )
    parser.add_argument("--dataset", default="ogbn-products", help="dataset")
    parser.add_argument(
        "--fanout", type=str, default="10,10,10", help="sampling fanout"
    )
    parser.add_argument("--model", type=str, default="SAGE", help="training model")
    parser.add_argument(
        "--dir",
        type=str,
        default="/nvme1n1/offgs_dataset",
        help="path to store subgraph",
    )
    parser.add_argument(
        "--cpu-cache-size", type=int, default=1000000000, help="cache size in bytes"
    )
    parser.add_argument(
        "--gpu-cache-size", type=int, default=1000000000, help="cache size in bytes"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=3, help="numbers of epoch in training"
    )
    ## argument whether use mega batch sampling
    parser.add_argument(
        "--mega_batch", action="store_true", help="whether use mega batch sampling"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="logs/train_single_thread_decompose.csv",
        help="log file",
    )
    args = parser.parse_args()
    print(args)

    start(args)
