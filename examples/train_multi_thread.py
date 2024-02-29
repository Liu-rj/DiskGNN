import torch
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from load_graph import *
from offgs.utils import SAGE, GAT
from offgs.dataset import OffgsDataset
import threading
import queue
import psutil
import csv
import atexit
import utils

import offgs


def load_graph(queue, path, batch_id, labels):
    for i in batch_id:
        blocks = torch.load(f"{path}/train-{i}.pt")
        output_nodes = torch.load(f"{path}/out-nid-{i}.pt")
        y = labels[output_nodes].long()
        queue.put((blocks, y))


def load_meta(feat_load_queue, transfer_queue, aux_dir, batch_id):
    for i in batch_id:
        (
            disk_cold,
            disk_rev_cold_idx,
            disk_loc,
            disk_rev_hot_idx,
            rev_cold_idx,
            mem_loc,
            rev_hot_idx,
        ) = torch.load(f"{aux_dir}/meta_data/train-aux-meta-{i}.pt")
        # pack_queue.put((disk_cold, disk_rev_cold_idx, disk_feats))
        # dc_queue.put((disk_loc, disk_rev_hot_idx, unique_page, inv_idx, disk_feats))
        # transfer_queue.put((num_disk, mem_loc, rev_cold_idx, rev_hot_idx, disk_feats))

        feat_load_queue.put((disk_cold, disk_rev_cold_idx, disk_loc, disk_rev_hot_idx))
        transfer_queue.put((rev_cold_idx, mem_loc, rev_hot_idx))


def load_packed_feats(in_queue, out_queue, aux_dir, batch_id, dataset):
    for i in batch_id:
        disk_cold, disk_rev_cold_idx, disk_feats = in_queue.get()
        if disk_cold.numel() > 0:
            cold_feats = torch.ops.offgs._CAPI_LoadFeats_Direct(
                f"{aux_dir}/feat/train-aux-{i}.bin",
                disk_cold.numel(),
                dataset.num_features,
            )
            disk_feats[disk_rev_cold_idx] = cold_feats
            torch.ops.offgs._CAPI_FreeTensor(cold_feats)
        out_queue.put(i)


def load_disk_cache(in_queue, out_queue, aux_dir, batch_id, dataset, segment_size):
    torch.ops.offgs._CAPI_Init_iouring()
    for i in batch_id:
        disk_loc, disk_rev_hot_idx, unique_page, inv_idx, disk_feats = in_queue.get()
        torch.ops.offgs._CAPI_LoadDiskCache_Direct_OMP_iouring(
            f"{aux_dir}/disk_cache/disk-cache-{i // segment_size}.bin",
            disk_feats,
            disk_loc,
            disk_rev_hot_idx,
            unique_page,
            inv_idx,
            dataset.num_features,
        )
        out_queue.put(i)
    torch.ops.offgs._CAPI_Exit_iouring()


def load_feats(in_queue, out_queue, aux_dir, batch_id, dataset, segment_size):
    torch.ops.offgs._CAPI_Init_iouring()
    for i in batch_id:
        disk_cold, disk_rev_cold_idx, disk_loc, disk_rev_hot_idx = in_queue.get()
        num_disk = disk_rev_cold_idx.numel() + disk_rev_hot_idx.numel()
        disk_feats = torch.empty(
            (num_disk, dataset.num_features),
            dtype=torch.float32,
            pin_memory=True,
        )

        if disk_cold.numel() > 0:
            cold_feats = torch.ops.offgs._CAPI_LoadFeats_Direct(
                f"{aux_dir}/feat/train-aux-{i}.bin",
                disk_cold.numel(),
                dataset.num_features,
            )
            disk_feats[disk_rev_cold_idx] = cold_feats
            torch.ops.offgs._CAPI_FreeTensor(cold_feats)

        torch.ops.offgs._CAPI_LoadDiskCache_Direct_OMP_iouring(
            f"{aux_dir}/disk_cache/disk-cache-{i // segment_size}.bin",
            disk_feats,
            disk_loc,
            disk_rev_hot_idx,
            dataset.num_features,
        )

        out_queue.put(disk_feats)
    torch.ops.offgs._CAPI_Exit_iouring()


def feat_transfer(
    in_queue,
    disk_feats_queue,
    out_queue,
    cpu_cached_feats,
    gpu_cached_feats,
    batch_id,
    dataset,
    device,
):
    for i in batch_id:
        # num_disk, mem_loc, rev_cold_idx, rev_hot_idx, disk_feats = in_queue.get()
        # pack_task = pack_task_q.get()
        # disk_cache_task = dc_task_q.get()
        # if pack_task != i or disk_cache_task != i:
        #     print("Error")
        #     exit()

        rev_cold_idx, mem_loc, rev_hot_idx = in_queue.get()
        disk_feats = disk_feats_queue.get()
        num_disk = rev_cold_idx.numel()

        num_input = num_disk + mem_loc.numel()
        x = torch.empty(
            (num_input, dataset.num_features),
            dtype=torch.float32,
            device=device,
        )
        if num_disk > 0:
            x[rev_cold_idx] = disk_feats.to(device)
        s = torch.cuda.Stream(device)
        with torch.cuda.stream(s):
            torch.ops.offgs._CAPI_GatherInGPU(
                x,
                rev_hot_idx.to(device),
                cpu_cached_feats,
                gpu_cached_feats,
                mem_loc.to(device),
            )
        s.synchronize()
        out_queue.put(x)


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

    if args.debug:
        graph = dataset.graph
        features = dataset.features.pin_memory()
    labels = dataset.labels.pin_memory()
    cpu_cached_feats = cpu_cached_feats.pin_memory()
    label_offset = dataset.conf["label_offset"]

    sampler = NeighborSampler([10, 10, 10])
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

    train_num = dataset.split_idx["train"].numel()
    size = (train_num + args.batchsize - 1) // args.batchsize
    print(f"Label Ratio: {train_num / dataset.num_nodes}, Down Sample: {args.ratio}")
    pool_size = (int(train_num * args.ratio) + args.batchsize - 1) // args.batchsize

    if args.debug:
        val_dataloader = DataLoader(
            graph,
            dataset.split_idx["valid"],
            sampler,
            batch_size=args.batchsize,
            shuffle=True,
            drop_last=False,
            device=torch.device(device),
            use_uva=True,
        )

        if args.dataset != "mag240m":
            test_dataloader = DataLoader(
                graph,
                dataset.split_idx["test"],
                sampler,
                batch_size=args.batchsize,
                shuffle=True,
                drop_last=False,
                device=torch.device(device),
                use_uva=True,
            )

    best_val_acc, best_test_acc, best_epoch = 0, 0, 0
    epoch_info_recorder = [[] for i in range(6)]
    for epoch in range(args.num_epoch):
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")

        batch_id = []
        while len(batch_id) < size:
            permutation = torch.randperm(pool_size).tolist()
            batch_id += permutation
        batch_id = batch_id[:size]

        threads = []
        (
            graph_queue,
            # pack_queue,
            # dc_queue,
            feat_load_queue,
            disk_feat_queue,
            transfer_queue,
            feature_queue,
        ) = [queue.Queue(maxsize=2) for i in range(5)]
        # pack_task_q, dc_task_q = queue.Queue(maxsize=2), queue.Queue(maxsize=2)

        threads.append(
            threading.Thread(
                target=load_graph,
                args=(graph_queue, subg_dir, batch_id, labels),
                daemon=True,
            )
        )

        threads.append(
            threading.Thread(
                target=load_meta,
                args=(feat_load_queue, transfer_queue, aux_dir, batch_id),
                daemon=True,
            )
        )

        # threads.append(
        #     threading.Thread(
        #         target=load_packed_feats,
        #         args=(pack_queue, pack_task_q, aux_dir, batch_id, dataset),
        #         daemon=True,
        #     )
        # )

        # threads.append(
        #     threading.Thread(
        #         target=load_disk_cache,
        #         args=(
        #             dc_queue,
        #             dc_task_q,
        #             aux_dir,
        #             batch_id,
        #             dataset,
        #             args.segment_size,
        #         ),
        #         daemon=True,
        #     )
        # )

        threads.append(
            threading.Thread(
                target=load_feats,
                args=(
                    feat_load_queue,
                    disk_feat_queue,
                    aux_dir,
                    batch_id,
                    dataset,
                    args.segment_size,
                ),
                daemon=True,
            )
        )

        threads.append(
            threading.Thread(
                target=feat_transfer,
                args=(
                    transfer_queue,
                    disk_feat_queue,
                    feature_queue,
                    cpu_cached_feats,
                    gpu_cached_feats,
                    batch_id,
                    dataset,
                    device,
                ),
                daemon=True,
            )
        )

        for t in threads:
            t.start()

        tot_loss = 0
        info_recorder = [0] * 5
        default_stream = torch.cuda.current_stream(device)
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        for i in tqdm(batch_id, ncols=100):
            # tic = time.time()
            blocks, y = graph_queue.get()
            blocks = [block.to(device, non_blocking=True) for block in blocks]
            y = y.to(device, non_blocking=True)
            # info_recorder[0] += time.time() - tic  # graph load

            x = feature_queue.get()

            if args.debug:
                input_nodes = torch.load(f"{subg_dir}/in-nid-{i}.pt")
                input_feats = features[input_nodes].to(device)
                assert torch.equal(x, input_feats)

            # tic = time.time()
            pred = model(blocks, x)
            loss = F.cross_entropy(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # default_stream.synchronize()
            # info_recorder[2] += time.time() - tic  # train

        for t in threads:
            t.join()
        torch.cuda.synchronize()
        epoch_time = time.time() - start

        if args.debug:
            # --- valid ---#
            model.eval()
            valid_correct, valid_tot, val_acc = 0, 0, 0
            for it, (input_nodes, output_nodes, blocks) in enumerate(
                tqdm(val_dataloader, ncols=100)
            ):
                blocks = [block.to(device) for block in blocks]
                x = gather_pinned_tensor_rows(features, input_nodes.to(device))
                # x = features[input_nodes.cpu()].to(device)
                y = gather_pinned_tensor_rows(
                    labels, output_nodes - label_offset
                ).long()
                pred = model(blocks, x)
                correct = (pred.argmax(dim=1) == y).sum().item()
                total = y.shape[0]
                valid_correct += correct
                valid_tot += total
            val_acc = valid_correct / valid_tot

            # --- test --- #
            model.eval()
            test_correct, test_tot, test_acc = 0, 0, 0
            if args.dataset != "mag240m":
                for it, (input_nodes, output_nodes, blocks) in enumerate(
                    tqdm(test_dataloader, ncols=100)
                ):
                    blocks = [block.to(device) for block in blocks]
                    x = gather_pinned_tensor_rows(features, input_nodes.to(device))
                    # x = features[input_nodes.cpu()].to(device)
                    y = gather_pinned_tensor_rows(
                        labels, output_nodes - label_offset
                    ).long()
                    pred = model(blocks, x)
                    correct = (pred.argmax(dim=1) == y).sum().item()
                    total = y.shape[0]
                    test_correct += correct
                    test_tot += total
                test_acc = test_correct / test_tot
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = epoch

            print(
                f"Loss: {tot_loss:.10f}\t"
                f"Valid acc: {val_acc * 100:.2f}%\t"
                f"Test acc: {test_acc * 100:.2f}%\t"
            )

        print(
            f"Graph Load Time: {info_recorder[0]:.3f}\t"
            f"Feature Load Time: {info_recorder[1]:.3f}\t"
            f"Train Time: {info_recorder[2]:.3f}\t"
            f"Epoch Time: {epoch_time:.3f}\t"
            f"Cold Feats num: {info_recorder[3]}\t"
            f"Feature Transfer num: {info_recorder[4]}\t"
        )
        for i, info in enumerate(info_recorder):
            epoch_info_recorder[i].append(info)
        epoch_info_recorder[-1].append(epoch_time)

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
            args.disk_cache_num,
            args.segment_size,
            args.model,
            args.num_epoch,
            best_epoch,
            round(best_val_acc * 100, 2),
            round(best_test_acc * 100, 2),
        ]
        for epoch_info in epoch_info_recorder:
            log_info.append(round(np.mean(epoch_info[1:]), 2))
        writer.writerow(log_info)


def init_cache(args, dataset, cached_nodes):
    device = torch.device(f"cuda:{args.device}")
    gpu_num_entries = args.gpu_cache_size // (4 * dataset.num_features)
    gpu_cached_idx = cached_nodes[:gpu_num_entries]
    cpu_cache_idx = cached_nodes[gpu_num_entries:]
    gpu_cached_feats = dataset.mmap_features[gpu_cached_idx].to(device)
    cpu_cached_feats = torch.empty(
        (cpu_cache_idx.numel(), dataset.num_features), dtype=torch.float32
    )
    cpu_cached_feats[:] = dataset.mmap_features[cpu_cache_idx]
    return cpu_cached_feats, gpu_cached_feats


def start(args):
    total_cache_size = args.cpu_cache_size + args.gpu_cache_size

    subg_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    subg_dir = os.path.join(args.dir, subg_dir)
    aux_dir = f"{subg_dir}/cache-size-{total_cache_size:g}/{args.segment_size}-{args.disk_cache_num:g}"
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
    parser.add_argument("--dataset", default="ogbn-products")
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="SAGE")
    parser.add_argument("--dir", type=str, default="/nvme1n1/offgs_dataset")
    parser.add_argument("--cpu-cache-size", type=float, default=1e10)
    parser.add_argument("--gpu-cache-size", type=float, default=1e10)
    parser.add_argument("--disk-cache-num", type=float, default=1e6)
    parser.add_argument("--segment-size", type=int, default=100)
    parser.add_argument("--num-epoch", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--log", type=str, default="logs/train_multi_thread.csv")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(args)
    args.cpu_cache_size = int(args.cpu_cache_size)
    args.gpu_cache_size = int(args.gpu_cache_size)

    start(args)
