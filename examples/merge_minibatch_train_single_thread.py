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
from offgs.utils import SAGE, GAT
from offgs.dataset import OffgsDataset
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
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    device = torch.device(f"cuda:{args.device}")
    fanout = [int(x) for x in args.fanout.split(",")]

    # labels = dataset.labels.pin_memory()
    labels = torch.randint(0, dataset.num_classes, (dataset.num_nodes,)).pin_memory()
    cpu_cached_feats = cpu_cached_feats.pin_memory()

    if args.model == "SAGE":
        model = SAGE(
            dataset.num_features,
            256,
            dataset.num_classes,
            len(fanout),
            args.dropout,
        ).to(device)
    elif args.model == "GAT":
        model = GAT(
            dataset.num_features,
            256,
            dataset.num_classes,
            [8, 2],
        ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    train_nid = (
        dataset.split_idx["train"]
        if args.ratio == 1.0
        else torch.load(f"{dataset_path}/train_idx_{args.ratio}.pt")
    )
    size = (train_nid.numel() + args.batchsize - 1) // args.batchsize

    epoch_info_recorder = [[] for i in range(10)]
    num_megabatch = args.mega_batch_size // args.batchsize
    last = size - size % num_megabatch

    for epoch in range(args.num_epoch):
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")

        info_recorder = [0] * 9
        torch.cuda.synchronize()
        model.train()
        for i in trange(size, ncols=100):
            tic = time.time()
            blocks = torch.load(f"{subg_dir}/train-{i}.pt")
            input_nodes = torch.load(f"{subg_dir}/in-nid-{i}.pt")
            output_nodes = torch.load(f"{subg_dir}/out-nid-{i}.pt")
            info_recorder[0] += time.time() - tic  # graph load

            if (i + num_megabatch) % num_megabatch == 0 or i == last:
                load_id = min(i + num_megabatch - 1, size - 1)
                unique_inv_idx = torch.load(f"{subg_dir}/unique_inv_idx-{load_id}.pt")

                tic = time.time()
                (
                    cold_nodes,
                    hot_nodes,
                    rev_hot_idx,
                    rev_cold_idx,
                ) = torch.load(f"{aux_dir}/train-aux-meta-{load_id}.pt")
                cold_feats = torch.tensor([], dtype=torch.float32)
                if cold_nodes.numel() > 0:
                    cold_feats = torch.ops.offgs._CAPI_LoadFeats_Direct(
                        f"{aux_dir}/train-aux-{load_id}.npy",
                        cold_nodes.numel(),
                        dataset.num_features,
                    )
                info_recorder[1] += time.time() - tic  # feature load
                info_recorder[7] += cold_nodes.numel()  # cold_feats_num

                tic = time.time()
                num_input = cold_nodes.numel() + hot_nodes.numel()
                cold_idx_map = torch.tensor([], dtype=torch.int32, device=device)
                if cold_nodes.numel() > 0:
                    cold_idx_map = torch.full(
                        (num_input,),
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
            minibatch_inv_idx = unique_inv_idx[i % num_megabatch].to(device)
            x = torch.empty(
                (input_nodes.numel(), dataset.num_features),
                dtype=torch.float32,
                device=device,
            )
            torch.ops.offgs._CAPI_GatherInGPU_MergeMiniBatch(
                x,
                input_nodes.to(device),
                minibatch_inv_idx,
                cpu_cached_feats,
                gpu_cached_feats,
                cold_feats,
                address_table,
                cold_idx_map,
            )
            torch.cuda.synchronize()
            info_recorder[2] += time.time() - tic  # feat assemble
            info_recorder[8] += x.shape[0]  # input node num

            if args.debug:
                input_nodes = input_nodes.to(device)
                cold_idx = cold_idx_map[
                    minibatch_inv_idx[address_table[input_nodes] == 0]
                ]
                minibatch_cold_feats = gather_pinned_tensor_rows(cold_feats, cold_idx)
                x[address_table[input_nodes] == 0] = minibatch_cold_feats
                cpu_hot_idx = (
                    address_table[input_nodes[address_table[input_nodes] > 0]] - 1
                )
                cpu_hot_feats = gather_pinned_tensor_rows(cpu_cached_feats, cpu_hot_idx)
                x[address_table[input_nodes] > 0] = cpu_hot_feats
                gpu_hot_idx = (
                    -address_table[input_nodes[address_table[input_nodes] < 0]] - 1
                )
                gpu_hot_feats = gpu_cached_feats[gpu_hot_idx]
                x[address_table[input_nodes] < 0] = gpu_hot_feats
                torch.cuda.synchronize()

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
            f"Assemble Time: {info_recorder[2]:.3f}\t"
            f"Graph & Label Transfer Time : {info_recorder[3]:.3f}\t"
            f"Feat Transfer Time: {info_recorder[4]:.3f}\t"
            f"Train Time: {info_recorder[5]:.3f}\t"
            f"Sample init time: {info_recorder[6]:.3f}\t"
            f"Epoch Time: {np.sum(info_recorder[:7]):.3f}\t"
            f"Cold Feats Num: {info_recorder[7]}\t"
            f"Feature Transfer Num: {info_recorder[8]}\t"
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
            args.ratio,
            args.mega_batch_size,
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

    subg_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    subg_dir = os.path.join(args.dir, subg_dir)
    mega_batch_dir = os.path.join(subg_dir, str(args.mega_batch_size))
    aux_dir = os.path.join(mega_batch_dir, f"cache-size-{total_cache_size}")
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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--mega_batch_size", type=int, default=4096)
    parser.add_argument("--dataset", default="ogbn-products")
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="SAGE")
    parser.add_argument("--dir", type=str, default="/nvme1n1/offgs_dataset")
    parser.add_argument("--ratio", type=float, default="1.0")
    parser.add_argument("--cpu-cache-size", type=int, default=1000000000)
    parser.add_argument("--gpu-cache-size", type=int, default=1000000000)
    parser.add_argument("--num-epoch", type=int, default=3)
    parser.add_argument(
        "--log",
        type=str,
        default="logs/merge_minibatch_train_single_thread_decompose.csv",
    )
    # debug flag
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    print(args)

    start(args)
