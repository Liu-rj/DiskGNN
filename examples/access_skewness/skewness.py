import torch
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="friendster")
parser.add_argument("--batchsize", type=int, default=1024)
parser.add_argument("--fanout", type=str, default="10,10,10")
parser.add_argument("--store-path", default="/nvme1n1/offgs_dataset")
parser.add_argument("--ratio", type=float, default=1.0)
args = parser.parse_args()
print(args)

dataset_path = f"{args.store_path}/{args.dataset}-offgs"
subg_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
subg_dir = os.path.join(args.store_path, subg_dir)

record_interval = torch.tensor([0, 0.01, 0.05, 0.1, 0.2, 0.5, 1])


node_counts = torch.load(f"{subg_dir}/node_counts.pt").cpu()
# sorted_idx = torch.load(f"{subg_dir}/meta_node_popularity.pt").cpu()
num_nodes = node_counts.numel()
all_counts = torch.sum(node_counts).item()
sort_sum_freq, perm_idx = torch.sort(node_counts, descending=True)
record_idx = (record_interval * num_nodes).to(torch.int64).tolist()
access_ratio = []
for i in range(len(record_idx) - 1):
    ratio = (
        torch.sum(sort_sum_freq[record_idx[i] : record_idx[i + 1]]).item() / all_counts
    )
    access_ratio.append(f"{round(ratio * 100, 1)}%")
print(access_ratio)
for i in access_ratio:
    print(i)
