import torch
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from offgs.utils import SAGE, GAT, RGCN
from offgs.dataset import OffgsDataset
import threading
import queue
import psutil
import csv
import json
import os

import offgs


def load_graph(queue, path, batch_id, labels, label_offset):
    category = "paper"
    for i in batch_id:
        blocks = torch.load(f"{path}/train-{i}.pt")
        output_nodes = torch.load(f"{path}/out-nid-{i}.pt")
        y = labels[category][output_nodes[category] - label_offset].long()
        queue.put((blocks, y))


def load_meta(feat_load_queue, transfer_queue, aux_dir, batch_id, dataset):
    for i in batch_id:
        all_feat_load_info, all_transfer_info = {}, {}
        for node_type in dataset.conf["num_nodes"].keys():
            (
                disk_cold,
                disk_rev_cold_idx,
                disk_loc,
                disk_rev_hot_idx,
                rev_cold_idx,
                mem_loc,
                rev_hot_idx,
            ) = torch.load(f"{aux_dir}/meta_data/train-aux-meta-{i}-{node_type}.pt")
            all_feat_load_info[node_type] = (
                disk_cold,
                disk_rev_cold_idx,
                disk_loc,
                disk_rev_hot_idx,
            )
            all_transfer_info[node_type] = (rev_cold_idx, mem_loc, rev_hot_idx)
        feat_load_queue.put(all_feat_load_info)
        transfer_queue.put(all_transfer_info)


def load_feats(
    in_queue,
    out_queue,
    aux_dir,
    batch_id,
    dataset,
    segment_size,
    # mmap_features,
):
    torch.ops.offgs._CAPI_Init_iouring()
    for i in batch_id:
        all_feat_load_info = in_queue.get()
        all_disk_feats = {}
        for node_type, info in all_feat_load_info.items():
            disk_cold, disk_rev_cold_idx, disk_loc, disk_rev_hot_idx = info
            num_disk = disk_rev_cold_idx.numel() + disk_rev_hot_idx.numel()
            disk_feats = torch.empty(
                (num_disk, dataset.num_features),
                dtype=eval(dataset.conf["features_dtype"]),
                pin_memory=True,
            )

            if disk_cold.numel() > 0:
                cold_feats = torch.ops.offgs._CAPI_LoadFeats_Direct(
                    f"{aux_dir}/feat/train-aux-{i}-{node_type}.bin",
                    disk_cold.numel(),
                    dataset.num_features,
                    dataset.conf["feat_itemsize"],
                )

                # mmap_cold_feats = mmap_features[disk_cold]

                disk_feats[disk_rev_cold_idx] = cold_feats

            torch.ops.offgs._CAPI_LoadDiskCache_Direct_OMP_iouring(
                f"{aux_dir}/disk_cache/disk-cache-{i // segment_size}-{node_type}.bin",
                disk_feats,
                disk_loc,
                disk_rev_hot_idx,
                dataset.num_features,
                dataset.conf["feat_itemsize"],
            )

            all_disk_feats[node_type] = disk_feats

        out_queue.put(all_disk_feats)
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
    # subg_dir,
    # mmap_features,
):
    for i in batch_id:
        all_transfer_info = in_queue.get()
        all_disk_feats = disk_feats_queue.get()
        all_x = {}
        for node_type, info in all_transfer_info.items():
            rev_cold_idx, mem_loc, rev_hot_idx = info
            disk_feats = all_disk_feats[node_type]
            num_disk = rev_cold_idx.numel()

            num_input = num_disk + mem_loc.numel()
            x = torch.empty(
                (num_input, dataset.num_features),
                dtype=eval(dataset.conf["features_dtype"]),
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
            # s.synchronize()
            torch.cuda.synchronize()

            # input_nodes = torch.load(f"{subg_dir}/in-nid-{i}.pt")
            # idx = input_nodes[node_type][rev_hot_idx]
            # offset = dataset.conf["node_offsets"][node_type]
            # x[rev_hot_idx] = mmap_features[idx + offset].to(device)

            all_x[node_type] = x.float()
        out_queue.put(all_x)


def train(
    args,
    dataset: OffgsDataset,
    cpu_cached_feats: torch.Tensor,
    gpu_cached_feats: torch.Tensor,
    subg_dir: str,
    aux_dir: str,
):
    category = "paper"
    dataset_path = f"{args.dir}/{args.dataset}-offgs"
    device = torch.device(f"cuda:{args.device}")
    fanout = [int(x) for x in args.fanout.split(",")]
    acc_logfile = f"acc/logs/offline_{args.dataset}_{args.fanout}_{args.model}_{args.hidden}_{args.dropout}_{args.ratio}.csv"
    torch.cuda.set_device(device)

    mmap_features = dataset.mmap_features
    if args.debug:
        graph = dataset.graph
        # features = dataset.features.pin_memory()
        mmap_features = dataset.mmap_features
    labels = dataset.labels
    cpu_cached_feats = cpu_cached_feats.pin_memory()
    label_offset = dataset.conf["label_offset"]

    model = RGCN(
        dataset.conf["etypes"],
        dataset.conf["ntypes"],
        dataset.num_features,
        dataset.num_classes,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_num = dataset.split_idx["train"][category].numel()
    size = (train_num + args.batchsize - 1) // args.batchsize
    total_num_nodes = dataset.conf["total_num_nodes"]
    print(f"Label Ratio: {train_num / total_num_nodes}, Down Sample: {args.ratio}")
    pool_size = (int(train_num * args.ratio) + args.batchsize - 1) // args.batchsize

    if args.debug:
        sampler = NeighborSampler(fanout)
        val_dataloader = DataLoader(
            graph,
            dataset.split_idx["valid"],
            sampler,
            batch_size=args.batchsize,
            shuffle=True,
            drop_last=False,
            device=device,
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
                device=device,
                use_uva=True,
            )

    best_val_acc, best_test_acc, best_epoch = 0, 0, 0
    epoch_info_recorder = [[] for i in range(6)]
    for epoch in range(args.num_epoch):
        # with open("/proc/sys/vm/drop_caches", "w") as stream:
        #     stream.write("1\n")

        batch_id = torch.randperm(pool_size).tolist()

        threads = []
        (
            graph_queue,
            feat_load_queue,
            disk_feat_queue,
            transfer_queue,
            feature_queue,
        ) = [queue.Queue(maxsize=2) for i in range(5)]

        threads.append(
            threading.Thread(
                target=load_graph,
                args=(graph_queue, subg_dir, batch_id, labels, label_offset),
                daemon=True,
            )
        )

        threads.append(
            threading.Thread(
                target=load_meta,
                args=(feat_load_queue, transfer_queue, aux_dir, batch_id, dataset),
                daemon=True,
            )
        )

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
                    # mmap_features,
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
                    # subg_dir,
                    # mmap_features,
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
        train_correct, train_tot, train_acc = 0, 0, 0
        for i in tqdm(batch_id, ncols=100):
            tic = time.time()
            blocks, y = graph_queue.get()
            blocks = [block.to(device, non_blocking=True) for block in blocks]
            y = y.to(device, non_blocking=True)

            x = feature_queue.get()
            info_recorder[0] += time.time() - tic  # data load
            for node_type, feat in x.items():
                info_recorder[4] += feat.numel()

            if args.debug:
                input_nodes = torch.load(f"{subg_dir}/in-nid-{i}.pt")
                for node_type, nid in input_nodes.items():
                    offset = dataset.conf["node_offsets"][node_type]
                    input_feats = mmap_features[nid + offset].to(device).float()
                    assert torch.equal(x[node_type], input_feats)

            # for key, value in x.items():
            #     print(key, torch.isnan(value).sum().item(), value.shape)

            # tic = time.time()
            logits = model(blocks, x)[category]

            y_hat = logits.log_softmax(dim=-1)
            loss = F.nll_loss(y_hat, y)

            if (
                torch.isnan(logits).sum().item() != 0
                or torch.isnan(y_hat).sum().item() != 0
                or torch.isnan(y).sum().item() != 0
                or torch.isnan(loss).sum().item() != 0
            ):
                print(
                    torch.isnan(logits).sum().item(),
                    torch.isnan(y_hat).sum().item(),
                    torch.isnan(y).sum().item(),
                    torch.isnan(loss).sum().item(),
                    logits.shape,
                    i,
                )

                input_nodes = torch.load(f"{subg_dir}/in-nid-{i}.pt")
                for node_type, nid in input_nodes.items():
                    offset = dataset.conf["node_offsets"][node_type]
                    input_feats = dataset.mmap_features[nid + offset].to(device).float()
                    print(
                        node_type,
                        offset,
                        torch.isnan(x[node_type]).sum().item(),
                        torch.isnan(input_feats).sum().item(),
                        torch.equal(x[node_type], input_feats),
                    )
                exit()

            tot_loss += loss.item() * blocks[-1].num_dst_nodes(category)
            opt.zero_grad()
            loss.backward()
            opt.step()

            correct = (y_hat.argmax(dim=1) == y).sum().item()
            total = y.shape[0]
            train_correct += correct
            train_tot += total
            # default_stream.synchronize()
            # info_recorder[2] += time.time() - tic  # train

        train_acc = train_correct / train_tot
        for t in threads:
            t.join()
        torch.cuda.synchronize()
        epoch_time = time.time() - start

        if args.debug and (epoch % args.log_every == 0 or epoch == args.num_epoch - 1):
            # --- valid ---#
            model.eval()
            valid_correct, valid_tot, val_acc = 0, 0, 0
            for it, (input_nodes, output_nodes, blocks) in enumerate(
                tqdm(val_dataloader, ncols=100)
            ):
                blocks = [block.to(device) for block in blocks]
                # x = gather_pinned_tensor_rows(features, input_nodes.to(device))
                x = mmap_features[input_nodes.cpu()].to(device)
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
                    # x = gather_pinned_tensor_rows(features, input_nodes.to(device))
                    x = mmap_features[input_nodes.cpu()].to(device)
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
                f"Epoch: {epoch}\t"
                f"Loss: {tot_loss:.4f}\t"
                f"Valid acc: {val_acc * 100:.2f}%\t"
                f"Test acc: {test_acc * 100:.2f}%\t"
            )

            with open(acc_logfile, "a") as f:
                writer = csv.writer(f, lineterminator="\n")
                log_info = [
                    epoch,
                    round(tot_loss, 3),
                    round(train_acc * 100, 2),
                    round(val_acc * 100, 2),
                    round(test_acc * 100, 2),
                    round(best_val_acc * 100, 2),
                    round(best_test_acc * 100, 2),
                    best_epoch,
                ]
                writer.writerow(log_info)

        print(
            f"Epoch: {epoch}\t"
            f"Graph Load Time: {info_recorder[0]:.3f}\t"
            f"Feature Load Time: {info_recorder[1]:.3f}\t"
            f"Train Time: {info_recorder[2]:.3f}\t"
            f"Epoch Time: {epoch_time:.3f}\t"
            f"Cold Feats Num: {info_recorder[3]}\t"
            f"Input Feature Num: {info_recorder[4]}\t"
            f"Loss: {tot_loss:.10f}\t"
            f"Train acc: {train_acc * 100:.2f}%"
        )
        for i, info in enumerate(info_recorder):
            epoch_info_recorder[i].append(info)
        epoch_info_recorder[-1].append(epoch_time)

    print(f"Avg Epoch Time: {np.mean(epoch_info_recorder[-1][1:]):.3f}")

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
            args.ratio,
            args.blowup,
            f"{args.disk_cache_num:g}",
            args.segment_size,
            args.model,
            args.hidden,
            args.dropout,
            args.num_epoch,
            best_epoch,
            round(best_val_acc * 100, 2),
            round(best_test_acc * 100, 2),
        ]
        for epoch_info in epoch_info_recorder:
            log_info.append(round(np.mean(epoch_info[1:]), 2))
        writer.writerow(log_info)


def init_cache(args, dataset, cached_feats):
    device = torch.device(f"cuda:{args.device}")
    gpu_num_entries = args.gpu_cache_size // (
        dataset.conf["feat_itemsize"] * dataset.num_features
    )
    gpu_cached_feats = cached_feats[:gpu_num_entries].to(device)
    cpu_cached_feats = cached_feats[gpu_num_entries:]
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
    # cached_nodes = torch.load(f"{aux_dir}/cached_nodes.pt")
    cached_feats = torch.load(f"{aux_dir}/cached_feats.pt")

    (
        cpu_cached_feats,
        gpu_cached_feats,
    ) = init_cache(args, dataset, cached_feats)
    print("CPU Cache Shape:", cpu_cached_feats.shape, "dtype:", cpu_cached_feats.dtype)
    print("GPU Cache Shape:", gpu_cached_feats.shape, "dtype:", gpu_cached_feats.dtype)

    args.cpu_cache_ratio = cpu_cached_feats.shape[0] / dataset.conf["total_num_nodes"]
    args.gpu_cache_ratio = gpu_cached_feats.shape[0] / dataset.conf["total_num_nodes"]

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
    parser.add_argument("--dataset", default="ogbn-products")
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--model", type=str, default="SAGE")
    parser.add_argument("--dir", type=str, default="/nvme1n1/offgs_dataset")
    parser.add_argument("--cpu-cache-size", type=float, default=1e10)
    parser.add_argument("--gpu-cache-size", type=float, default=1e10)
    # parser.add_argument("--disk-cache-num", type=float, default=1e6)
    # parser.add_argument("--segment-size", type=int, default=100)
    parser.add_argument("--blowup", type=float, default=-1)
    parser.add_argument("--num-epoch", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--log", type=str, default="logs/train_multi_thread.csv")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()
    print(args)
    args.cpu_cache_size = int(args.cpu_cache_size)
    args.gpu_cache_size = int(args.gpu_cache_size)

    start(args)
