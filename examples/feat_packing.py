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
import csv

import offgs


def run(dataset: OffgsDataset, args):
    device = torch.device(f"cuda:{args.device}")

    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    output_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    output_dir = os.path.join(args.store_path, output_dir)
    aux_dir = f"{output_dir}/cache-size-{args.feat_cache_size:g}/{args.segment_size}-{args.disk_cache_num:g}"

    if not os.path.exists(aux_dir):
        os.makedirs(aux_dir)
    if not os.path.exists(f"{aux_dir}/feat"):
        os.mkdir(f"{aux_dir}/feat")
    if not os.path.exists(f"{aux_dir}/meta_data"):
        os.mkdir(f"{aux_dir}/meta_data")
    if not os.path.exists(f"{aux_dir}/disk_cache"):
        os.mkdir(f"{aux_dir}/disk_cache")

    features = dataset.mmap_features

    node_counts = torch.load(f"{output_dir}/node_counts.pt").cpu()
    sorted_idx = torch.load(f"{output_dir}/meta_node_popularity.pt").cpu()

    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")

    train_idx = (
        dataset.split_idx["train"]
        if args.ratio == 1.0
        else torch.load(f"{dataset_path}/train_idx_{args.ratio}.pt")
    )
    num_batches = (train_idx.numel() + args.batchsize - 1) // args.batchsize
    segment_size = num_batches if args.segment_size == -1 else args.segment_size

    tic = time.time()
    num_entries = min(
        args.feat_cache_size // (4 * features.shape[1]),
        dataset.num_nodes,
    )
    # Maximum 400GB cache size for int32 (aligned with Ginex)
    if num_entries > torch.iinfo(torch.int32).max:
        raise ValueError
    cache_indices = sorted_idx[:num_entries]
    address_table = torch.full((dataset.num_nodes,), -1, dtype=torch.int32)
    address_table[cache_indices] = torch.arange(num_entries, dtype=torch.int32)
    torch.save(cache_indices, f"{aux_dir}/cached_nodes.pt")
    # torch.save(address_table, f"{aux_dir}/address_table.pt")
    key, value = torch.ops.offgs._CAPI_BuildHashMap(cache_indices.to(device))
    cache_init_time = time.time() - tic
    print(
        f"#Cached Entries: {num_entries} / {dataset.num_nodes}",
        f"Ratio: {num_entries / dataset.num_nodes}",
        f"Access Cache Ratio: {node_counts[cache_indices].sum().item() / node_counts.sum().item()}",
    )

    assert args.disk_cache_num + num_entries < dataset.num_nodes
    num_segments = (num_batches + segment_size - 1) // segment_size
    print(
        f"Num Segments: {num_segments}, Segment Size: {segment_size} ({args.segment_size})"
    )

    time_record = [0] * 11
    total_packed_nodes = 0
    for segid in trange(num_segments, ncols=100):
        startid = segid * segment_size
        endid = min((segid + 1) * segment_size, num_batches)

        # calculate popularity (node counts) for packed features in each segment
        tic = time.time()
        popularity = torch.zeros(dataset.num_nodes, dtype=torch.int32, device=device)
        for bid in range(startid, endid):
            input_nodes = torch.load(f"{output_dir}/in-nid-{bid}.pt").to(device)
            cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
            popularity[cold_nodes] += 1
        time_record[0] += time.time() - tic

        # build disk cache
        tic = time.time()
        seg_sorted_idx = torch.argsort(popularity, descending=True)
        cache_num = min(args.disk_cache_num, (popularity > 0).sum().item())
        disk_cache = seg_sorted_idx[:cache_num].cpu()
        disk_table = torch.full((dataset.num_nodes,), -1, dtype=torch.int64)
        disk_table[disk_cache] = torch.arange(cache_num, dtype=torch.int64)
        disk_key, disk_value = torch.ops.offgs._CAPI_BuildHashMap(disk_cache.to(device))
        time_record[1] += time.time() - tic
        print(
            f"\nDisk Cache Entries: {cache_num} / {dataset.num_nodes}",
            f"Ratio: {cache_num / dataset.num_nodes:.3f}",
            f"Ratio In Seg: {cache_num / (popularity > 0).sum().item():.3f}",
            f"Access Cache Ratio: {popularity[disk_cache].sum().item() / popularity.sum().item():.3f}",
        )

        # build bipartite graph
        tic = time.time()
        src, dst = [], []
        for bid in range(startid, endid):
            input_nodes = torch.load(f"{output_dir}/in-nid-{bid}.pt").to(device)
            cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
            disk_hot = torch.ops.offgs._CAPI_QueryHashMap(
                cold_nodes, disk_key, disk_value
            )[2]

            disk_loc = disk_table[disk_hot.cpu()]
            assert (disk_loc != -1).all()
            src += disk_loc.tolist()
            dst += [bid] * disk_loc.numel()
        time_record[2] += time.time() - tic

        # calculate minhash
        tic = time.time()
        tensor_src = torch.tensor(src, dtype=torch.int64).pin_memory()
        tensor_dst = torch.tensor(dst, dtype=torch.int64).pin_memory()
        num_src = cache_num
        num_dst = num_batches
        h1 = torch.randperm(num_batches, device=device)
        h1_res, degree = torch.ops.offgs._CAPI_SegmentedMinHash(
            tensor_src, tensor_dst, h1, num_src, num_dst, False
        )
        h2 = torch.randperm(num_batches, device=device)
        h2_res, _ = torch.ops.offgs._CAPI_SegmentedMinHash(
            tensor_src, tensor_dst, h2, num_src, num_dst, False
        )
        h_res = h1_res * num_batches + h2_res
        time_record[3] += time.time() - tic

        # reorder disk cache
        tic = time.time()
        # h_res_cpu, degree_cpu = h_res.cpu().tolist(), degree.cpu().tolist()
        # indices = sorted(
        #     range(h_res.numel()), key=lambda k: (h_res_cpu[k], -degree_cpu[k])
        # )
        indices = torch.argsort(h_res).cpu()
        reordered_disk_cache = disk_cache[indices]
        disk_table[reordered_disk_cache] = torch.arange(cache_num, dtype=torch.int64)
        time_record[4] += time.time() - tic

        # load disk cache features
        tic = time.time()
        # disk_cache_feats: torch.Tensor = features[reordered_disk_cache]
        disk_cache_feats = torch.ops.offgs._CAPI_GatherPReadDirect(
            dataset.features_path, reordered_disk_cache.cpu(), dataset.num_features
        )
        time_record[5] += time.time() - tic

        # save disk cache features
        tic = time.time()
        torch.ops.offgs._CAPI_SaveFeats(
            f"{aux_dir}/disk_cache/disk-cache-{segid}.bin", disk_cache_feats
        )
        # if cache_num > 0:
        #     mmap_dc_feats = np.memmap(
        #         f"{aux_dir}/disk_cache/disk-cache-{segid}.npy",
        #         mode="w+",
        #         shape=disk_cache_feats.numel(),
        #         dtype=np.float32,
        #     )
        #     mmap_dc_feats[:] = disk_cache_feats.flatten()
        #     mmap_dc_feats.flush()
        time_record[6] += time.time() - tic

        for bid in range(startid, endid):
            # calculate cold nodes
            tic = time.time()
            input_nodes = torch.load(f"{output_dir}/in-nid-{bid}.pt").to(device)

            (
                cold_nodes,
                rev_cold_idx,
                hot_nodes,
                rev_hot_idx,
            ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)
            mem_loc = address_table[hot_nodes.cpu()].to(torch.int64)

            (
                disk_cold,
                disk_rev_cold_idx,
                disk_hot,
                disk_rev_hot_idx,
            ) = torch.ops.offgs._CAPI_QueryHashMap(cold_nodes, disk_key, disk_value)
            disk_loc = disk_table[disk_hot.cpu()]
            total_packed_nodes += disk_cold.numel()
            time_record[7] += time.time() - tic

            # load packed features
            tic = time.time()
            # packed_feats: torch.Tensor = features[disk_cold.cpu()]
            # packed_feats = torch.ops.offgs._CAPI_GatherMemMap(features, cold_nodes.cpu(), dataset.num_features)
            packed_feats = torch.ops.offgs._CAPI_GatherPReadDirect(
                dataset.features_path, disk_cold.cpu(), dataset.num_features
            )
            time_record[8] += time.time() - tic

            # save meta data
            tic = time.time()
            aux_meta_data = [
                disk_cold.cpu(),
                disk_rev_cold_idx.cpu(),
                disk_loc.cpu(),
                disk_rev_hot_idx.cpu(),
                rev_cold_idx.cpu(),
                mem_loc.cpu(),
                rev_hot_idx.cpu(),
            ]
            torch.save(aux_meta_data, f"{aux_dir}/meta_data/train-aux-meta-{bid}.pt")
            time_record[9] += time.time() - tic

            # save packed features
            tic = time.time()
            torch.ops.offgs._CAPI_SaveFeats(
                f"{aux_dir}/feat/train-aux-{bid}.bin", packed_feats
            )
            # if disk_cold.numel() > 0:
            #     mmap_packed_feats = np.memmap(
            #         f"{aux_dir}/feat/train-aux-{bid}.npy",
            #         mode="w+",
            #         shape=packed_feats.numel(),
            #         dtype=np.float32,
            #     )
            #     mmap_packed_feats[:] = packed_feats.flatten()
            #     mmap_packed_feats.flush()
            time_record[10] += time.time() - tic

    total_time = cache_init_time + np.sum(time_record)
    print(
        f"Init Cache Time: {cache_init_time:.3f}\t"
        f"Cal Counts Time: {time_record[0]:.3f}\t"
        f"Build Disk Cache Time: {time_record[1]:.3f}\t"
        f"Build Bipartite Graph Time: {time_record[2]:.3f}\t"
        f"Cal MinHash Time: {time_record[3]:.3f}\t"
        f"Reorder Time: {time_record[4]:.3f}\t"
        f"Load Disk Cache Time: {time_record[5]:.3f}\t"
        f"Save Disk Cache Time: {time_record[6]:.3f}\t"
        f"Cal Cold Nodes Time: {time_record[7]:.3f}\t"
        f"Load Packed Feats Time: {time_record[8]:.3f}\t"
        f"Save Meta Data Time: {time_record[9]:.3f}\t"
        f"Save Packed Feats Time: {time_record[10]:.3f}\t"
        f"Total Time: {total_time:.3f}"
    )

    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.ratio,
            f"{args.feat_cache_size:g}",
            args.disk_cache_num,
            args.segment_size,
            total_packed_nodes,
            round(cache_init_time, 2),
        ]
        for i in range(len(time_record)):
            log_info.append(round(time_record[i], 2))
        log_info.append(round(total_time, 2))
        writer.writerow(log_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--store-path", default="/nvme2n1")
    parser.add_argument("--feat-cache-size", type=float, default=1e10)
    parser.add_argument("--disk-cache-num", type=float, default=1e6)
    parser.add_argument("--segment-size", type=int, default=100)
    parser.add_argument("--log", type=str, default="logs/pack_decompose.csv")
    parser.add_argument("--ratio", type=float, default=1.0)
    args = parser.parse_args()
    print(args)
    args.feat_cache_size = int(args.feat_cache_size)
    args.disk_cache_num = int(args.disk_cache_num)

    # --- load data --- #
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    dataset = OffgsDataset(dataset_path)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem, "GB")

    run(dataset, args)
