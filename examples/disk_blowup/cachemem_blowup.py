import torch
import argparse
from tqdm import tqdm
import time
from offgs.dataset import OffgsDataset
import csv
import os


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--dataset", type=str, default="friendster")
parser.add_argument("--batchsize", type=int, default=1024)
parser.add_argument("--fanout", type=str, default="10,10,10")
parser.add_argument("--store-path", default="/nvme1n1/offgs_dataset")
parser.add_argument("--ratio", type=float, default=1.0)
args = parser.parse_args()
print(args)

if args.dataset == "friendster":
    # feat_cache_sizes = [4e9, 7e9, 10e9, 15e9, 20e9]
    feat_cache_sizes = [5e9]
elif args.dataset == "igb-full":
    # feat_cache_sizes = [15e9, 30e9, 45e9, 75e9, 105e9]
    feat_cache_sizes = [22.5e9]
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")
# feat_cache_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# --- load data --- #
dataset_path = f"{args.store_path}/{args.dataset}-offgs"
dataset = OffgsDataset(dataset_path)
device = torch.device(f"cuda:{args.device}")

dataset_path = f"{args.store_path}/{args.dataset}-offgs"
output_dir = f"{args.dataset}-{args.batchsize}-{args.fanout}-{args.ratio}"
output_dir = os.path.join(args.store_path, output_dir)

features = dataset.mmap_features

node_counts = torch.load(f"{output_dir}/node_counts.pt").cpu()
sorted_idx = torch.load(f"{output_dir}/meta_node_popularity.pt").cpu()
sorted_counts = node_counts[sorted_idx]

feat_load_time, nid_load_time, difference_time, save_time = 0, 0, 0, 0

start = time.time()

train_idx = (
    dataset.split_idx["train"]
    if args.ratio == 1.0
    else torch.load(f"{dataset_path}/train_idx_{args.ratio}.pt")
)
print(f"Train Node Ratio: {train_idx.numel() / dataset.num_nodes}")
num_batches = (train_idx.numel() + args.batchsize - 1) // args.batchsize

blowup_ratio_list = []
for cache_size in tqdm(feat_cache_sizes, ncols=100):
    num_entries = min(
        int(cache_size) // (4 * features.shape[1]),
        dataset.num_nodes,
    )
    pack_num = sorted_counts[num_entries:].sum().item()
    blowup_ratio = pack_num / dataset.num_nodes
    blowup_ratio_list.append(blowup_ratio)
    tqdm.write(
        f"#Cached Entries: {num_entries} / {dataset.num_nodes}"
        f"Ratio: {num_entries / dataset.num_nodes}"
        f"Access Cache Ratio: {sorted_counts[:num_entries].sum().item() / sorted_counts.sum().item()}"
        f"Blowup Ratio: {pack_num / dataset.num_nodes}"
    )

f = open("cachemem_blowup_fs_ig.csv", "a")
writer = csv.writer(f, lineterminator="\n")
log_info = [args.dataset, args.fanout, args.batchsize]
log_info += [f"{x:.3f}" for x in blowup_ratio_list]
writer.writerow(log_info)
f.close()
