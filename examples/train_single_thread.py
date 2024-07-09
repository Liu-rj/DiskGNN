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


def train(
    args,
    dataset: OffgsDataset,
    cpu_cached_feats: torch.Tensor,
    gpu_cached_feats: torch.Tensor,
    subg_dir: str,
    aux_dir: str,
):
    dataset_path = f"{args.dir}/{args.dataset}-offgs"
    device = torch.device(f"cuda:{args.device}")
    fanout = [int(x) for x in args.fanout.split(",")]
    acc_logfile = f"acc/logs/offline_{args.dataset}_{args.fanout}_{args.model}_{args.hidden}_{args.dropout}_{args.ratio}.csv"
    torch.cuda.set_device(device)

    if args.debug:
        graph = dataset.graph
        features = dataset.features.pin_memory()
    labels = dataset.labels.pin_memory()
    cpu_cached_feats = cpu_cached_feats.pin_memory()
    label_offset = dataset.conf["label_offset"]

    sampler = NeighborSampler(fanout)
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
    size = (train_num + args.batchsize - 1) // args.batchsize
    print(f"Label Ratio: {train_num / dataset.num_nodes}, Down Sample: {args.ratio}")
    pool_size = (int(train_num * args.ratio) + args.batchsize - 1) // args.batchsize

    torch.ops.offgs._CAPI_Init_iouring()

    best_val_acc, best_test_acc, best_epoch = 0, 0, 0
    epoch_info_recorder = [[] for i in range(10)]
    feat_load_decompose = [[] for i in range(13)]
    for epoch in range(args.num_epoch):
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")

        # batch_id = []
        # while len(batch_id) < size:
        #     permutation = torch.randperm(pool_size).tolist()
        #     batch_id += permutation
        # batch_id = batch_id[:size]
        batch_id = torch.randperm(pool_size).tolist()

        tot_loss = 0
        info_recorder = [0] * 9
        torch.cuda.synchronize()

        meta_load, feat_pin, feat_free = 0, 0, 0
        feat_create, cold_load, cache_load = 0, 0, 0
        total_disk_cold, total_disk_cache = 0, 0
        (
            disk_page_unique,
            page_alloc,
            disk_cache_load,
            disk_cache_copy,
            disk_page_free,
        ) = (0, 0, 0, 0, 0)

        model.train()
        for i in tqdm(batch_id, ncols=100):
            tic = time.time()
            blocks = torch.load(f"{subg_dir}/train-{i}.pt")
            output_nodes = torch.load(f"{subg_dir}/out-nid-{i}.pt")
            info_recorder[0] += time.time() - tic  # graph load

            tic = time.time()
            (
                disk_cold,
                disk_rev_cold_idx,
                disk_loc,
                disk_rev_hot_idx,
                rev_cold_idx,
                mem_loc,
                rev_hot_idx,
            ) = torch.load(f"{aux_dir}/meta_data/train-aux-meta-{i}.pt")
            meta_load += time.time() - tic

            tic = time.time()
            num_disk = disk_rev_cold_idx.numel() + disk_rev_hot_idx.numel()
            disk_feats = torch.empty(
                (num_disk, dataset.num_features),
                dtype=torch.float32,
                pin_memory=True,
            )
            feat_create += time.time() - tic

            tic = time.time()
            if disk_cold.numel() > 0:
                cold_feats = torch.ops.offgs._CAPI_LoadFeats_Direct(
                    f"{aux_dir}/feat/train-aux-{i}.bin",
                    disk_cold.numel(),
                    dataset.num_features,
                )
                disk_feats[disk_rev_cold_idx] = cold_feats
            cold_load += time.time() - tic

            tic = time.time()
            torch.ops.offgs._CAPI_LoadDiskCache_Direct_OMP_iouring(
                f"{aux_dir}/disk_cache/disk-cache-{i // args.segment_size}.bin",
                disk_feats,
                disk_loc,
                disk_rev_hot_idx,
                dataset.num_features,
            )
            cache_load += time.time() - tic
            io_traffic = disk_cold.numel() + torch.unique(disk_loc // 8).numel() * 8
            info_recorder[7] += io_traffic  # cold_feats_num
            total_disk_cold += disk_cold.numel()
            total_disk_cache += torch.unique(disk_loc // 8).numel() * 8

            tic = time.time()
            num_input = num_disk + mem_loc.numel()
            x = torch.empty(
                (num_input, dataset.num_features),
                dtype=torch.float32,
                device=device,
            )
            if num_disk > 0:
                x[rev_cold_idx] = disk_feats.to(device)
            torch.ops.offgs._CAPI_GatherInGPU(
                x,
                rev_hot_idx.to(device),
                cpu_cached_feats,
                gpu_cached_feats,
                mem_loc.to(device),
            )
            torch.cuda.synchronize()
            info_recorder[2] += time.time() - tic  # feat assemble
            info_recorder[8] += x.shape[0]  # input node num
            if args.debug:
                input_nodes = torch.load(f"{subg_dir}/in-nid-{i}.pt")
                input_feats = features[input_nodes].to(device)
                assert torch.equal(x, input_feats)

            tic = time.time()
            blocks = [block.to(device) for block in blocks]
            y = labels[output_nodes - label_offset].to(device).long()
            # y = labels[output_nodes].to(device).long()
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

            # tic = time.time()
            # if disk_cold.numel() > 0:
            #     torch.ops.offgs._CAPI_FreeTensor(cold_feats)
            # feat_free += time.time() - tic

        info_recorder[1] += (
            meta_load + feat_create + cold_load + cache_load + feat_pin + feat_free
        )  # feature load

        print(
            f"Meta Load Time: {meta_load:.3f}\t"
            f"Feat Create Time: {feat_create:.3f}\t"
            f"Cold Load Time: {cold_load:.3f}\t"
            f"Cache Load Time: {cache_load:.3f}\t"
            f"Feat Pin Time: {feat_pin:.3f}\t"
            f"Feat Free Time: {feat_free:.3f}\t"
            f"Total Disk Cold: {total_disk_cold}\t"
            f"Total Disk Cache: {total_disk_cache}\t"
            f"Disk Page Unique: {disk_page_unique:.3f}\t"
            f"Disk Page Alloc: {page_alloc:.3f}\t"
            f"Disk Cache Load: {disk_cache_load:.3f}\t"
            f"Disk Cache Copy: {disk_cache_copy:.3f}\t"
            f"Disk Page Free: {disk_page_free:.3f}\t"
        )

        print(
            f"Graph Load Time: {info_recorder[0]:.3f}\t"
            f"Feature Load Time: {info_recorder[1]:.3f}\t"
            f"Feat Assemble Time: {info_recorder[2]:.3f}\t"
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
        for i, info in enumerate(
            [
                meta_load,
                feat_create,
                cold_load,
                cache_load,
                feat_pin,
                feat_free,
                total_disk_cold,
                total_disk_cache,
                disk_page_unique,
                page_alloc,
                disk_cache_load,
                disk_cache_copy,
                disk_page_free,
            ]
        ):
            feat_load_decompose[i].append(info)

    torch.ops.offgs._CAPI_Exit_iouring()

    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            f"{args.cpu_cache_size:g}",
            f"{args.gpu_cache_size:g}",
            round(args.cpu_cache_ratio, 2),
            round(args.gpu_cache_ratio, 2),
            args.blowup,
            f"{args.disk_cache_num:g}",
            args.segment_size,
            args.model,
            args.num_epoch,
            best_epoch,
            round(best_val_acc * 100, 2),
            round(best_test_acc * 100, 2),
        ]
        for epoch_info in epoch_info_recorder:
            log_info.append(round(np.mean(epoch_info[1:]), 2))
        for epoch_info in feat_load_decompose:
            log_info.append(round(np.mean(epoch_info[1:]), 2))
        writer.writerow(log_info)


def init_cache(args, dataset, cached_nodes):
    device = torch.device(f"cuda:{args.device}")
    # table_size = 4 * dataset.num_nodes
    # print(f"Adress Table Size: {table_size / (1024 * 1024 * 1024)} GB")
    # gpu_idx_cache_size = max(0, args.gpu_cache_size - table_size)
    gpu_num_entries = args.gpu_cache_size // (4 * dataset.num_features)
    gpu_cached_idx = cached_nodes[:gpu_num_entries]
    cpu_cache_idx = cached_nodes[gpu_num_entries:]
    gpu_cached_feats = dataset.mmap_features[gpu_cached_idx].to(device)
    cpu_cached_feats = torch.empty(
        (cpu_cache_idx.numel(), dataset.num_features), dtype=torch.float32
    )
    cpu_cached_feats[:] = dataset.mmap_features[cpu_cache_idx]
    # address_table = torch.zeros((dataset.num_nodes,), dtype=torch.int32)
    # address_table[cpu_cache_idx] = (
    #     torch.arange(cpu_cache_idx.numel(), dtype=torch.int32) + 1
    # )
    # address_table[gpu_cached_idx] = -(
    #     torch.arange(gpu_cached_idx.numel(), dtype=torch.int32) + 1
    # )
    # address_table = address_table.to(device)
    return cpu_cached_feats, gpu_cached_feats


def start(args):
    total_cache_size = args.cpu_cache_size + args.gpu_cache_size

    subg_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    subg_dir = os.path.join(args.dir, subg_dir)
    aux_dir = f"{subg_dir}/cache-size-{total_cache_size:g}/blowup-{args.blowup}"
    dataset_dir = f"{args.dir}/{args.dataset}-offgs"

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    dataset = OffgsDataset(dataset_dir)
    cached_nodes = torch.load(f"{aux_dir}/cached_nodes.pt")

    (
        cpu_cached_feats,
        gpu_cached_feats,
    ) = init_cache(args, dataset, cached_nodes)

    args.cpu_cache_ratio = cpu_cached_feats.shape[0] / dataset.num_nodes
    args.gpu_cache_ratio = gpu_cached_feats.shape[0] / dataset.num_nodes

    dc_config = json.load(open(f"{aux_dir}/dc_config.json", "r"))
    args.disk_cache_num = dc_config["disk_cache_num"]
    args.segment_size = dc_config["segment_size"]

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem1 - mem, "GB")

    train(
        args,
        dataset,
        cpu_cached_feats,
        gpu_cached_feats,
        subg_dir,
        aux_dir,
    )


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
    parser.add_argument("--cpu-cache-size", type=float, default=1e10)
    parser.add_argument("--gpu-cache-size", type=float, default=0)
    # parser.add_argument("--disk-cache-num", type=float, default=1e6)
    # parser.add_argument("--segment-size", type=int, default=100)
    parser.add_argument("--blowup", type=float, default=-1)
    parser.add_argument("--num-epoch", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--log", type=str, default="logs/train_single_thread.csv")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()
    print(args)
    args.cpu_cache_size = int(args.cpu_cache_size)
    args.gpu_cache_size = int(args.gpu_cache_size)

    start(args)
