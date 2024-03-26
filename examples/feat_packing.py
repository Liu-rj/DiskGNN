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


def binary_search(popularity, target_size):
    left, right = 0, (popularity > 1).sum().item()
    while left < right:
        mid = left + (right - left) // 2
        est_disk_nodes = mid + popularity[mid:].sum().item()
        if est_disk_nodes <= target_size:
            right = mid
        else:
            left = mid + 1
    return left


def search_disk_cache_brutal_force(
    dataset, subg_dir, cache_map, num_batches, target_size, device
):
    page_feats = 4096 / (dataset.num_features * 4)
    key, value = cache_map
    best_segment_size, best_disk_cache_num, best_io_amp = 0, 0, 8
    for segment_size in trange(2, num_batches + 1, 100, ncols=100):
        # calculate popularity (node counts) for packed features in each segment
        popularity = torch.zeros(dataset.num_nodes, dtype=torch.int32, device=device)
        for it, bid in enumerate(range(segment_size)):
            input_nodes = torch.load(f"{subg_dir}/in-nid-{bid}.pt").to(device)
            cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
            popularity[cold_nodes] += 1
        max_cache_num = (popularity > 0).sum().item()
        est_disk_nodes = max_cache_num * num_batches / segment_size
        if est_disk_nodes > target_size:
            continue
        sorted_popularity, seg_sorted_idx = torch.sort(popularity, descending=True)
        cache_num = binary_search(
            sorted_popularity, target_size * segment_size / num_batches
        )

        # build disk cache
        assert cache_num <= (popularity > 0).sum().item()
        disk_cache = seg_sorted_idx[:cache_num].cpu()
        disk_table = torch.full((dataset.num_nodes,), -1, dtype=torch.int64)
        disk_table[disk_cache] = torch.arange(cache_num, dtype=torch.int64)
        disk_key, disk_value = torch.ops.offgs._CAPI_BuildHashMap(disk_cache.to(device))

        # build bipartite graph
        src, dst = [], []
        for bid in range(segment_size):
            input_nodes = torch.load(f"{subg_dir}/in-nid-{bid}.pt").to(device)
            cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
            disk_hot = torch.ops.offgs._CAPI_QueryHashMap(
                cold_nodes, disk_key, disk_value
            )[2]

            disk_loc = disk_table[disk_hot.cpu()]
            assert (disk_loc != -1).all()
            src += disk_loc.tolist()
            dst += [bid] * disk_loc.numel()

        # calculate minhash
        tensor_src = torch.tensor(src, dtype=torch.int64).pin_memory()
        tensor_dst = torch.tensor(dst, dtype=torch.int64).pin_memory()
        num_src = cache_num
        num_dst = num_batches
        h1 = torch.randperm(num_batches, device=device)
        h1_res, _ = torch.ops.offgs._CAPI_SegmentedMinHash(
            tensor_src, tensor_dst, h1, num_src, num_dst, False
        )
        h2 = torch.randperm(num_batches, device=device)
        h2_res, _ = torch.ops.offgs._CAPI_SegmentedMinHash(
            tensor_src, tensor_dst, h2, num_src, num_dst, False
        )
        h_res = h1_res * num_batches + h2_res

        # reorder disk cache
        indices = torch.argsort(h_res).cpu()
        reordered_disk_cache = disk_cache[indices]
        disk_table[reordered_disk_cache] = torch.arange(cache_num, dtype=torch.int64)

        ori_io, opt_io, total_disk_nodes = 0, 0, cache_num
        for bid in range(segment_size):
            input_nodes = torch.load(f"{subg_dir}/in-nid-{bid}.pt").to(device)
            cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
            (
                disk_cold,
                _,
                disk_hot,
                _,
            ) = torch.ops.offgs._CAPI_QueryHashMap(cold_nodes, disk_key, disk_value)

            disk_loc = disk_table[disk_hot.cpu()]
            assert (disk_loc != -1).all()
            ori_io += cold_nodes.numel()
            num_pages = torch.unique(disk_loc // page_feats)
            opt_io += disk_cold.numel() + num_pages.numel() * page_feats
            total_disk_nodes += disk_cold.numel()

        tqdm.write(
            f"Segment Size: {segment_size}\t"
            f"Disk Cache Num: {cache_num} / {(popularity > 0).sum().item()}\t"
            f"Disk Blowup: {(total_disk_nodes * num_batches / segment_size) / dataset.num_nodes}\t"
            f"IO Traffic Amplification: {opt_io / ori_io}"
        )
        if opt_io / ori_io < best_io_amp:
            best_segment_size = segment_size
            best_disk_cache_num = cache_num
            best_io_amp = opt_io / ori_io
    print(
        f"Best Segment Size: {best_segment_size}\t"
        f"Best Disk Cache Num: {best_disk_cache_num}\t"
        f"IO Traffic Amplification: {best_io_amp}"
    )
    return best_segment_size, best_disk_cache_num


def search_disk_cache_approximate(
    dataset, subg_dir, cache_map, num_batches, target_size, device
):
    key, value = cache_map
    popularity = torch.zeros(dataset.num_nodes, dtype=torch.int32, device=device)
    for it, bid in enumerate(trange(num_batches, ncols=100)):
        input_nodes = torch.load(f"{subg_dir}/in-nid-{bid}.pt").to(device)
        cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
        popularity[cold_nodes] += 1
        disk_cache_num = (popularity > 1).sum().item()
        disk_cold_num = popularity[popularity == 1].sum().item()
        est_disk_nodes = (disk_cache_num + disk_cold_num) * num_batches / (it + 1)
        if est_disk_nodes < target_size or it == num_batches - 1:
            segment_size = it + 1
            break
    return segment_size, disk_cache_num


def run(dataset: OffgsDataset, args):
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    page_feats = 4096 / (dataset.num_features * 4)

    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    subg_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    subg_dir = os.path.join(args.store_path, subg_dir)
    aux_dir = f"{subg_dir}/cache-size-{args.feat_cache_size:g}/blowup-{args.blowup}"

    if not os.path.exists(aux_dir):
        os.makedirs(aux_dir)
    if not os.path.exists(f"{aux_dir}/feat"):
        os.mkdir(f"{aux_dir}/feat")
    if not os.path.exists(f"{aux_dir}/meta_data"):
        os.mkdir(f"{aux_dir}/meta_data")
    if not os.path.exists(f"{aux_dir}/disk_cache"):
        os.mkdir(f"{aux_dir}/disk_cache")

    features = dataset.mmap_features

    node_counts = torch.load(f"{subg_dir}/node_counts.pt").cpu()
    sorted_idx = torch.load(f"{subg_dir}/meta_node_popularity.pt").cpu()

    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")

    train_idx = (
        dataset.split_idx["train"]
        if args.ratio == 1.0
        else torch.load(f"{dataset_path}/train_idx_{args.ratio}.pt")
    )
    print(f"Train Node Ratio: {train_idx.numel() / dataset.num_nodes}")
    num_batches = (train_idx.numel() + args.batchsize - 1) // args.batchsize
    # segment_size = num_batches if args.segment_size == -1 else args.segment_size

    # build in-mem cache
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

    # search disk cache
    tic = time.time()
    if args.blowup == -1:
        segment_size, disk_cache_num = num_batches, 0
    else:
        assert args.blowup > 0
        target_size = dataset.num_nodes * args.blowup
        segment_size, disk_cache_num = search_disk_cache_approximate(
            dataset, subg_dir, (key, value), num_batches, target_size, device
        )
        # segment_size, disk_cache_num = search_disk_cache_brutal_force(
        #     dataset, subg_dir, (key, value), num_batches, target_size, device
        # )
        segment_size = num_batches if disk_cache_num == 0 else segment_size
    dc_search_time = time.time() - tic
    print(f"Search Time: {dc_search_time:.3f} s")
    print(f"Segment Size: {segment_size}, Disk Cache Num: {disk_cache_num}")
    dc_config = {"segment_size": segment_size, "disk_cache_num": disk_cache_num}
    json.dump(dc_config, open(os.path.join(aux_dir, "dc_config.json"), "w"))

    assert disk_cache_num + num_entries < dataset.num_nodes
    num_segments = (num_batches + segment_size - 1) // segment_size
    print(f"Num Segments: {num_segments}, Segment Size: {segment_size}")

    time_record = [0] * 11
    ori_io, opt_io, total_disk_nodes = 0, 0, 0
    for segid in trange(num_segments, ncols=100):
        startid = segid * segment_size
        endid = min((segid + 1) * segment_size, num_batches)

        # calculate popularity (node counts) for packed features in each segment
        tic = time.time()
        popularity = torch.zeros(dataset.num_nodes, dtype=torch.int32, device=device)
        for bid in range(startid, endid):
            input_nodes = torch.load(f"{subg_dir}/in-nid-{bid}.pt").to(device)
            cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
            popularity[cold_nodes] += 1
        time_record[0] += time.time() - tic

        # build disk cache
        tic = time.time()
        seg_sorted_idx = torch.argsort(popularity, descending=True)
        # assert disk_cache_num <= (popularity > 0).sum().item()
        cache_num = min(disk_cache_num, (popularity > 1).sum().item())
        disk_cache = seg_sorted_idx[:cache_num].cpu()
        disk_table = torch.full((dataset.num_nodes,), -1, dtype=torch.int64)
        disk_table[disk_cache] = torch.arange(cache_num, dtype=torch.int64)
        disk_key, disk_value = torch.ops.offgs._CAPI_BuildHashMap(disk_cache.to(device))
        time_record[1] += time.time() - tic
        tqdm.write(
            f"\nDisk Cache Entries: {cache_num} / {dataset.num_nodes}\t"
            f"Ratio: {cache_num / dataset.num_nodes:.3f}\t"
            f"Ratio In Seg: {cache_num / (popularity > 0).sum().item():.3f}\t"
            f"Access Cache Ratio: {popularity[disk_cache].sum().item() / popularity.sum().item():.3f}"
        )
        total_disk_nodes += cache_num

        # build bipartite graph
        tic = time.time()
        src, dst = [], []
        for bid in range(startid, endid):
            input_nodes = torch.load(f"{subg_dir}/in-nid-{bid}.pt").to(device)
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
        h1_res, _ = torch.ops.offgs._CAPI_SegmentedMinHash(
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
        indices = torch.argsort(h_res).cpu()
        reordered_disk_cache = disk_cache[indices]
        disk_table[reordered_disk_cache] = torch.arange(cache_num, dtype=torch.int64)
        time_record[4] += time.time() - tic

        # load disk cache features
        tic = time.time()
        disk_cache_feats = torch.ops.offgs._CAPI_GatherPReadDirect(
            dataset.features_path, reordered_disk_cache.cpu(), dataset.num_features
        )
        time_record[5] += time.time() - tic

        # save disk cache features
        tic = time.time()
        torch.ops.offgs._CAPI_SaveFeats(
            f"{aux_dir}/disk_cache/disk-cache-{segid}.bin", disk_cache_feats
        )
        time_record[6] += time.time() - tic

        for bid in range(startid, endid):
            # calculate cold nodes
            tic = time.time()
            input_nodes = torch.load(f"{subg_dir}/in-nid-{bid}.pt").to(device)

            (
                cold_nodes,
                rev_cold_idx,
                hot_nodes,
                rev_hot_idx,
            ) = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)
            ## hotnodes local id in CPU cache
            mem_loc = address_table[hot_nodes.cpu()].to(torch.int64)

            (
                disk_cold,
                disk_rev_cold_idx,
                disk_hot,
                disk_rev_hot_idx,
            ) = torch.ops.offgs._CAPI_QueryHashMap(cold_nodes, disk_key, disk_value)
            disk_loc = disk_table[disk_hot.cpu()]
            time_record[7] += time.time() - tic
            total_disk_nodes += disk_cold.numel()
            ori_io += cold_nodes.numel()
            num_pages = torch.unique(disk_loc // page_feats).numel()
            opt_io += disk_cold.numel() + num_pages * page_feats

            # load packed features
            tic = time.time()
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
            time_record[10] += time.time() - tic

    total_time = dc_search_time + cache_init_time + np.sum(time_record)
    print(
        f"Disk Cache Search Time: {dc_search_time:.3f}\t"
        f"Original Blowup: {ori_io / dataset.num_nodes:.3f}\t"
        f"Opt Blowup: {total_disk_nodes / dataset.num_nodes:.3f}\t"
        f"IO Traffic Amplification: {opt_io / ori_io:.3f}"
    )
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
            f"{disk_cache_num:g}",
            segment_size,
            total_disk_nodes,
            round(cache_init_time, 2),
            round(dc_search_time, 2),
            round(opt_io / ori_io, 2),
        ]
        for i in range(len(time_record)):
            log_info.append(round(time_record[i], 2))
        log_info.append(round(total_time, 2))
        writer.writerow(log_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="friendster")
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--store-path", default="/nvme1n1/offgs_dataset")
    parser.add_argument("--feat-cache-size", type=float, default=1e10)
    # parser.add_argument("--disk-cache-num", type=float, default=0)
    # parser.add_argument("--segment-size", type=int, default=200)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--blowup", type=float, default=-1)
    parser.add_argument("--log", type=str, default="logs/pack_decompose.csv")
    args = parser.parse_args()
    print(args)
    args.feat_cache_size = int(args.feat_cache_size)
    # args.disk_cache_num = int(args.disk_cache_num)

    # --- load data --- #
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    dataset = OffgsDataset(dataset_path)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem, "GB")

    run(dataset, args)
