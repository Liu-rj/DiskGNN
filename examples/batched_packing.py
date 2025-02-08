import torch
import argparse
import numpy as np
from tqdm import tqdm, trange
import psutil
import time
import json
from offgs.dataset import OffgsDataset
import csv
import os
import shutil

import offgs

time_record = [0] * 11


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


def reorder_disk_cache(
    dataset,
    startid,
    endid,
    key,
    value,
    device,
    num_batches,
    subg_dir,
    disk_cache_num,
    input_nodes_list,
    on_demand_load,
):
    # calculate popularity (node counts) for packed features in each segment
    tic = time.time()
    popularity = torch.zeros(dataset.num_nodes, dtype=torch.int32, device=device)
    for it, bid in enumerate(range(startid, endid)):
        if on_demand_load:
            input_nodes = torch.load(f"{subg_dir}/in-nid-{bid}.pt").to(device)
        else:
            input_nodes = input_nodes_list[it]
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
    if (popularity > 0).sum().item() > 0:
        tqdm.write(
            f"\nDisk Cache Entries: {cache_num} / {dataset.num_nodes}\t"
            f"Ratio: {cache_num / dataset.num_nodes:.3f}\t"
            f"Ratio In Seg: {cache_num / (popularity > 0).sum().item():.3f}\t"
            f"Access Cache Ratio: {popularity[disk_cache].sum().item() / popularity.sum().item():.3f}"
        )

    # build bipartite graph
    tic = time.time()
    src, dst = [], []
    for it, bid in enumerate(range(startid, endid)):
        if on_demand_load:
            input_nodes = torch.load(f"{subg_dir}/in-nid-{bid}.pt").to(device)
        else:
            input_nodes = input_nodes_list[it]
        cold_nodes = torch.ops.offgs._CAPI_QueryHashMap(input_nodes, key, value)[0]
        disk_hot = torch.ops.offgs._CAPI_QueryHashMap(cold_nodes, disk_key, disk_value)[
            2
        ]

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

    return reordered_disk_cache, disk_table, disk_key, disk_value


def save_disk_cache(dataset, segid, reordered_disk_cache, aux_dir):
    # load disk cache features
    tic = time.time()
    disk_cache_feats = torch.ops.offgs._CAPI_GatherPReadDirect(
        dataset.features_path,
        reordered_disk_cache.cpu(),
        dataset.num_features,
        dataset.conf["feat_itemsize"],
    )
    time_record[5] += time.time() - tic

    # save disk cache features
    tic = time.time()
    torch.ops.offgs._CAPI_SaveFeats(
        f"{aux_dir}/disk_cache/disk-cache-{segid}.bin", disk_cache_feats
    )
    time_record[6] += time.time() - tic


def run(dataset: OffgsDataset, args):
    start = time.time()
    device = torch.device(f"cuda:{args.device}")
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    subg_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
    subg_dir = os.path.join(args.store_path, subg_dir)
    aux_dir = f"{subg_dir}/cache-size-{args.feat_cache_size:g}/blowup-{args.blowup}"

    if os.path.exists(aux_dir):
        shutil.rmtree(aux_dir)
    os.makedirs(aux_dir)
    os.mkdir(f"{aux_dir}/feat")
    os.mkdir(f"{aux_dir}/meta_data")
    os.mkdir(f"{aux_dir}/disk_cache")

    node_counts = torch.load(f"{subg_dir}/node_counts.pt").cpu()
    sorted_idx = torch.load(f"{subg_dir}/meta_node_popularity.pt").cpu()

    # with open("/proc/sys/vm/drop_caches", "w") as stream:
    #     stream.write("1\n")

    train_idx = (
        dataset.split_idx["train"]
        if args.ratio == 1.0
        else torch.load(f"{dataset_path}/train_idx_{args.ratio}.pt")
    )
    num_batches = (train_idx.numel() + args.batchsize - 1) // args.batchsize

    # build in-mem cache
    tic = time.time()
    num_entries = min(
        args.feat_cache_size // (dataset.conf["feat_itemsize"] * dataset.num_features),
        dataset.num_nodes,
    )
    # Maximum 400GB cache size for int32 (aligned with Ginex)
    if num_entries > torch.iinfo(torch.int32).max:
        raise ValueError
    cache_indices = sorted_idx[:num_entries]
    cache_local_idx = torch.arange(num_entries, dtype=torch.int32)
    address_table = torch.full((dataset.num_nodes,), -1, dtype=torch.int32)
    address_table[cache_indices] = cache_local_idx
    torch.save(cache_indices, f"{aux_dir}/cached_nodes.pt")
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
    dc_search_time = time.time() - tic
    print(f"Search Time: {dc_search_time:.3f} s")
    print(f"Segment Size: {segment_size}, Disk Cache Num: {disk_cache_num}")
    dc_config = {"segment_size": segment_size, "disk_cache_num": disk_cache_num}
    json.dump(dc_config, open(os.path.join(aux_dir, "dc_config.json"), "w"))

    assert disk_cache_num + num_entries < dataset.num_nodes
    num_segments = (num_batches + segment_size - 1) // segment_size
    print(f"Num Segments: {num_segments}, Segment Size: {segment_size}")

    total_packed_nodes = 0
    on_demand_load = False
    num_seg_dataset = 8
    all_indices = dataset.num_nodes
    step_indices = dataset.num_nodes // num_seg_dataset + 1

    # iterate over segments of graph samples to generate indices
    all_disk_cold = []
    all_disk_rev_cold_idx = []
    for segid in trange(num_segments, ncols=100):
        startid = segid * segment_size
        endid = min((segid + 1) * segment_size, num_batches)

        # load all input node id and store it into a list of tensor
        tic = time.time()
        input_nodes_list = []
        if not on_demand_load:
            for bid in range(startid, endid):
                input_nodes_list.append(torch.load(f"{subg_dir}/in-nid-{bid}.pt"))
        time_record[0] += time.time() - tic

        # generate disk cache
        if disk_cache_num > 0:
            reordered_disk_cache, disk_table, disk_key, disk_value = reorder_disk_cache(
                dataset,
                startid,
                endid,
                key,
                value,
                device,
                num_batches,
                subg_dir,
                disk_cache_num,
                input_nodes_list,
                on_demand_load,
            )
            save_disk_cache(dataset, segid, reordered_disk_cache, aux_dir)
        else:
            disk_table = torch.full((dataset.num_nodes,), -1, dtype=torch.int64)
            disk_key, disk_value = torch.ops.offgs._CAPI_BuildHashMap(
                torch.tensor([], device=device, dtype=torch.int64)
            )

        # pack cold features
        for it, bid in enumerate(range(startid, endid)):
            tic = time.time()
            if on_demand_load:
                input_nodes = torch.load(f"{subg_dir}/in-nid-{bid}.pt").to(device)
            else:
                input_nodes = input_nodes_list[it].to(device)

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
            all_disk_cold.append(disk_cold.cpu())
            all_disk_rev_cold_idx.append(disk_rev_cold_idx.cpu())
            time_record[7] += time.time() - tic

            # save meta data
            tic = time.time()
            aux_meta_data = [
                # disk_cold.cpu(),
                # disk_rev_cold_idx.cpu(),
                disk_loc.cpu(),
                disk_rev_hot_idx.cpu(),
                rev_cold_idx.cpu(),
                mem_loc.cpu(),
                rev_hot_idx.cpu(),
            ]
            torch.save(aux_meta_data, f"{aux_dir}/meta_data/train-aux-meta-{bid}.pt")
            time_record[9] += time.time() - tic

    # iterate over segments of features to generate pack features
    sliced_feature_num = 0
    all_cache_rev_ind = []
    all_disk_rev_cold_idx_new_order = [[] for _ in range(num_batches)]
    all_disk_cold_new_order = [[] for _ in range(num_batches)]
    for partid in range(num_seg_dataset):
        ## init a table with a size of dataset.num_nodes (torch.tensor)
        table = torch.full((dataset.num_nodes,), 0, dtype=bool)

        current_indices = partid * step_indices
        if partid == num_seg_dataset - 1:
            step_indices = all_indices - current_indices

        # load feature partition
        tic = time.time()
        feature_part = torch.ops.offgs._CAPI_LoadFeats_Direct_lseek(
            dataset.features_path,
            current_indices,
            step_indices,
            dataset.num_features,
        )
        table[current_indices : current_indices + step_indices] = True
        cache_init_time += time.time() - tic

        # save cached feature partition
        tic = time.time()
        part_cache_mask = table[cache_indices]
        part_cache_id = cache_indices[part_cache_mask]
        part_cache_rev_ind = cache_local_idx[part_cache_mask]
        part_cache_feats = feature_part[part_cache_id - current_indices]
        all_cache_rev_ind.append(part_cache_rev_ind)
        torch.ops.offgs._CAPI_SaveFeatsAppend(
            f"{aux_dir}/cached_feats.bin", part_cache_feats
        )
        cache_init_time += time.time() - tic

        # iterate over segments of graph samples to pack features
        for segid in trange(num_segments, ncols=100):
            startid = segid * segment_size
            endid = min((segid + 1) * segment_size, num_batches)

            # pack cold features
            for bid in range(startid, endid):
                disk_cold = all_disk_cold[bid]
                disk_rev_cold_idx = all_disk_rev_cold_idx[bid]

                ## generate correct node order to load from disk (not in CPU cache)
                tic = time.time()
                # Boolean mask to identify the true indices within input_node
                mask = table[disk_cold]
                # Find the indices of the true values within input_node
                all_disk_cold_new_order[bid].append(disk_cold[mask])
                all_disk_rev_cold_idx_new_order[bid].append(disk_rev_cold_idx[mask])
                ## slice the required features in current partition
                features_slice = feature_part[disk_cold[mask] - current_indices]
                time_record[8] += time.time() - tic
                sliced_feature_num += features_slice.shape[0]

                # save packed features
                tic = time.time()
                torch.ops.offgs._CAPI_SaveFeatsAppend(
                    f"{aux_dir}/feat/train-aux-{bid}.bin", features_slice
                )
                time_record[10] += time.time() - tic

        print(f"partition {partid} done, sliced feature num: {sliced_feature_num}")

    tic = time.time()
    for bid in range(num_batches):
        torch.save(
            [
                torch.cat(all_disk_cold_new_order[bid]),
                torch.cat(all_disk_rev_cold_idx_new_order[bid]),
            ],
            f"{aux_dir}/meta_data/disk-cold-idx-{bid}.pt",
        )
    time_record[9] += time.time() - tic

    tic = time.time()
    torch.save(torch.cat(all_cache_rev_ind), f"{aux_dir}/cache_rev_idx.pt")
    cache_init_time += time.time() - tic

    total_time = time.time() - start

    print(
        "end of feature packing"
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
            disk_cache_num,
            segment_size,
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
    parser.add_argument("--dataset", type=str, default="friendster")
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--store-path", default="/nvme1n1/offgs_dataset")
    parser.add_argument("--feat-cache-size", type=float, default=3e9)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--blowup", type=float, default=-1)
    parser.add_argument("--log", type=str, default="logs/pack_decompose_mapreduce.csv")
    args = parser.parse_args()
    print(args)
    args.feat_cache_size = int(args.feat_cache_size)
    assert args.blowup == -1, "blowup only support -1 for batched packing now"

    # --- load data --- #
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    dataset = OffgsDataset(dataset_path)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem, "GB")

    run(dataset, args)
