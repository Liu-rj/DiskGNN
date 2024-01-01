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


def train(args, dataset: OffgsDataset, address_table, cached_feats, subg_dir, aux_dir):
    device = torch.device(f"cuda:{args.device}")
    fanout = [int(x) for x in args.fanout.split(",")]

    labels = dataset.labels.pin_memory()

    if args.model == "SAGE":
        model = SAGE(dataset.num_features, 256, dataset.num_classes, len(fanout)).to(
            device
        )
    elif args.model == "GAT":
        model = GAT(dataset.num_features, 256, dataset.num_classes, [8, 2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    size = (dataset.split_idx["train"].numel() + args.batchsize - 1) // args.batchsize

    epoch_info_recorder = [[] for i in range(9)]
    number_of_batch_merge_into_one = args.mega_batch_size // args.batchsize

    for epoch in range(args.num_epoch):
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")

        info_recorder = [0] * 9
        torch.cuda.synchronize()
        first_greater_multiple_index=((size + number_of_batch_merge_into_one - 1) // number_of_batch_merge_into_one) * number_of_batch_merge_into_one
        model.train()
        for i in trange(size):
            # tic = time.time()
            # with open("/proc/sys/vm/drop_caches", "w") as stream:
            #     stream.write("1\n")
            # clear_cache += time.time() - tic

            tic = time.time()
            blocks = torch.load(f"{subg_dir}/train-{i}.pt")
            output_nodes = torch.load(f"{subg_dir}/out-nid-{i}.pt")
            if i % number_of_batch_merge_into_one == 0 and (i+number_of_batch_merge_into_one<first_greater_multiple_index):
                unique_nodes_mapping=torch.load(f"{subg_dir}/unique_nodes_mapping-{i+number_of_batch_merge_into_one-1}.pt")
                train_aux_id_to_load=i+number_of_batch_merge_into_one-1

            
            info_recorder[0] += time.time() - tic  # graph load
            if (i+number_of_batch_merge_into_one>=first_greater_multiple_index):
                train_aux_id_to_load=i
            if i % number_of_batch_merge_into_one == 0 or (i+number_of_batch_merge_into_one>=first_greater_multiple_index):
                tic = time.time()
                
                (
                    cold_feats,
                    cold_nodes,
                    hot_nodes,
                    rev_hot_idx,
                    rev_cold_idx,
                ) = torch.ops.offgs._CAPI_LoadFeats_Direct(
                    f"{aux_dir}/train-aux-{train_aux_id_to_load}.npy",
                    dataset.num_features)
                # cold_feats, cold_nodes, hot_nodes, rev_hot_idx, rev_cold_idx = torch.load(
                #     f"{aux_dir}/train-aux-{i}.pt"
                # )
                info_recorder[1] += time.time() - tic  # feature load
                info_recorder[7] += cold_feats.shape[0]  # cold_feats_num

                tic = time.time()
                num_input = cold_nodes.numel() + hot_nodes.numel()
                ## large overhead
                x = torch.empty(
                    (num_input, dataset.num_features),
                    dtype=torch.float32,
                    pin_memory=True,
                )
                
                
                x[rev_cold_idx] = cold_feats
                # x[rev_hot_idx] = cached_feats[address_table[hot_nodes]]
                torch.ops.offgs._CAPI_GatherInMem(
                    x, rev_hot_idx, cached_feats, hot_nodes, address_table
                )
                info_recorder[2] += time.time() - tic  # assemble
            

            
            tic = time.time()
            if (i+number_of_batch_merge_into_one>=first_greater_multiple_index):
                x_in_mini_batch=x
                # x_in_mini_batch=x_in_mini_batch.pin_memory()
            else:
                x_in_mini_batch=torch.empty(
                    (unique_nodes_mapping[i%number_of_batch_merge_into_one].shape[0], dataset.num_features),
                    dtype=torch.float32,
                    pin_memory=True,
                )
                torch.index_select(x,0,unique_nodes_mapping[i%number_of_batch_merge_into_one],out=x_in_mini_batch)
                
            blocks = [block.to(device) for block in blocks]
            torch.cuda.synchronize()
            info_recorder[3] += time.time() - tic  # graph transfer
            tic = time.time()
            x_in_mini_batch = x_in_mini_batch.to(device)
            y = labels[output_nodes].to(device).long()
            torch.cuda.synchronize()
            info_recorder[4] += time.time() - tic  # feature transfer
            info_recorder[8] += x_in_mini_batch.shape[0]  # input node num
            tic = time.time()
            pred = model(blocks, x_in_mini_batch)
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
            f"Graph Transfer Time : {info_recorder[3]:.3f}\t"
            f"Feat Transfer Time: {info_recorder[4]:.3f}\t"
            f"Train Time: {info_recorder[5]:.3f}\t"
            f'Sample init time: {info_recorder[6]:.3f}\t'
            f"Epoch Time: {np.sum(info_recorder[:7]):.3f}\t"
            f"Cold Feats Num: {info_recorder[7]}\t"
            f"Feature Transfer Num: {info_recorder[8]}\t"
        )
        for i, info in enumerate(info_recorder):
            epoch_info_recorder[i].append(info)
        epoch_info_recorder[-1].append(np.sum(info_recorder[:7]))
    ## [TODO] fix report
    with open("/home/ubuntu/OfflineSampling/examples/logs/merge_mini_train_decompose.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.mega_batch_size,
            args.feat_cache_size,
            args.model,
            args.num_epoch,
        ]
        for epoch_info in epoch_info_recorder:
            log_info.append(round(np.mean(epoch_info[1:]), 2))
        writer.writerow(log_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Training model device")
    parser.add_argument(
        "--batchsize", type=int, default=1024, help="batch size for training"
    )
    parser.add_argument("--mega_batch_size", type=int, default=4096, help="mega batch size for training")

    parser.add_argument("--dataset", default="friendster", help="dataset")
    parser.add_argument(
        "--fanout", type=str, default="10,10,10", help="sampling fanout"
    )
    parser.add_argument("--model", type=str, default="SAGE", help="training model")
    parser.add_argument(
        "--dir", type=str, default="/nvme2n1", help="path to store subgraph"
    )
    parser.add_argument(
        "--feat-cache-size", type=int, default=6400000000, help="cache size in bytes"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=3, help="numbers of epoch in training"
    )
    ## argument whether use mega batch sampling
    parser.add_argument('--mega_batch', action='store_true',help='whether use mega batch sampling')
    args = parser.parse_args()
    print(args)

    subg_dir = f"{args.dir}/{args.dataset}-{args.mega_batch_size}-{args.fanout}"
    aux_dir = f"{subg_dir}/cache-size-{args.feat_cache_size}"
    dataset_dir = f"{args.dir}/{args.dataset}-offgs"

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    # indices = torch.load(f"{subg_dir}/meta_node_popularity.pt")

    dataset = OffgsDataset(dataset_dir)
    cached_nodes = torch.load(f"{aux_dir}/cached_nodes.pt")
    cached_feats = torch.empty(
        (cached_nodes.numel(), dataset.num_features), dtype=torch.float32
    )
    cached_feats[:] = dataset.mmap_features[cached_nodes]
    address_table = torch.load(f"{aux_dir}/address_table.pt")

    # address_table, cache = init_cache(indices, dataset.mmap_features, mmap_features.shape[0], mmap_features.shape[1], args.feat_cache_size)

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem1 - mem, "GB")

    train(args, dataset, address_table, cached_feats, subg_dir, aux_dir)