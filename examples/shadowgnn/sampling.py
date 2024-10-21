import torch
from dgl.dataloading import DataLoader, ShaDowKHopSampler
import argparse
import os
import psutil
import time
from offgs.dataset import OffgsDataset
import csv
from tqdm import tqdm

import offgs


def run(args, dataset: OffgsDataset):
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    fanout = [int(x) for x in args.fanout.split(",")]
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    output_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}-Shadow"
    output_dir = os.path.join(args.store_path, output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    g = dataset.graph
    print(g)
    train_nid = (
        dataset.split_idx["train"]
        if args.ratio == 1.0
        else torch.load(f"{dataset_path}/train_idx_{args.ratio}.pt")
    )
    print(f"Train Node Ratio: {train_nid.numel() / g.num_nodes()}")

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory consumption:", mem1 - mem, "GB")

    sampler = ShaDowKHopSampler(fanout)
    train_dataloader = DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        use_uva=True,
        num_workers=0,
        device=device,
    )

    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")

    node_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device=device)
    sample_time, save_block, save_rank = 0, 0, 0
    tic = time.time()
    for i, (input_nodes, output_nodes, subg) in enumerate(
        tqdm(train_dataloader, ncols=100)
    ):
        subg.ndata.clear()
        subg.edata.clear()
        subg.srcdata.clear()
        subg.dstdata.clear()
        sample_time += time.time() - tic

        tic = time.time()
        torch.save(subg, f"{output_dir}/train-{i}.pt")
        torch.save(input_nodes.cpu(), f"{output_dir}/in-nid-{i}.pt")
        torch.save(output_nodes.cpu(), f"{output_dir}/out-nid-{i}.pt")
        node_counts[input_nodes] += 1
        save_block += time.time() - tic

        tic = time.time()

    tic = time.time()
    sorted_idx = torch.argsort(node_counts, descending=True)
    ## save nodecounts
    torch.save(node_counts.cpu(), f"{output_dir}/node_counts.pt")
    torch.save(sorted_idx.cpu(), f"{output_dir}/meta_node_popularity.pt")
    save_rank += time.time() - tic

    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.ratio,
            round(sample_time, 2),
            round(save_block, 2),
            round(save_rank, 2),
            round(sample_time + save_block + save_rank, 2),
        ]
        writer.writerow(log_info)

    print(
        f"Sampling Time: {sample_time:.3f}\t"
        f"Save Block Time: {save_block:.3f}\t"
        f"Save Rank Time: {save_rank:.3f}\t"
        f"Total Time: {(sample_time + save_block + save_rank):.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--store-path", default="/nvme2n1")
    parser.add_argument("--log", type=str, default="logs/sample_decompose.csv")
    parser.add_argument("--ratio", type=float, default=1.0)
    args = parser.parse_args()
    print(args)

    # --- load dataset --- #
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    dataset = OffgsDataset(dataset_path)

    run(args, dataset)
