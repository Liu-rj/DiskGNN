import torch
import argparse
from tqdm import tqdm
from offgs.dataset import OffgsDataset
from dgl.dataloading import DataLoader, NeighborSampler
import csv


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--dataset", type=str, default="friendster")
parser.add_argument("--batchsize", type=int, default=1024)
parser.add_argument("--feat-cache-size", type=float, default=1e10)
parser.add_argument("--store-path", default="/nvme1n1/offgs_dataset")
parser.add_argument("--ratio", type=float, default=1.0)
args = parser.parse_args()
print(args)

fanout_list = ["10,15", "10,10,10", "10,15,20"]

# --- load data --- #
dataset_path = f"{args.store_path}/{args.dataset}-offgs"
dataset = OffgsDataset(dataset_path)

device = torch.device(f"cuda:{args.device}")
dataset_path = f"{args.store_path}/{args.dataset}-offgs"

features = dataset.mmap_features
g = dataset.graph
print(g)
train_nid = (
    dataset.split_idx["train"]
    if args.ratio == 1.0
    else torch.load(f"{dataset_path}/train_idx_{args.ratio}.pt")
)
print(f"Train Node Ratio: {train_nid.numel() / g.num_nodes()}")

blowup_ratio_list = []
for fanout in fanout_list:
    fanout = [int(x) for x in fanout.split(",")]
    sampler = NeighborSampler(fanout)
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

    node_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device=device)
    for i, (input_nodes, output_nodes, blocks) in enumerate(
        tqdm(train_dataloader, ncols=100)
    ):
        node_counts[input_nodes] += 1

    sorted_counts = torch.sort(node_counts, descending=True)[0]

    pack_nodes_num = []
    num_entries = min(
        int(args.feat_cache_size) // (4 * features.shape[1]),
        dataset.num_nodes,
    )
    pack_num = sorted_counts[num_entries:].sum().item()
    blowup_ratio = pack_num / dataset.num_nodes
    blowup_ratio_list.append(blowup_ratio)
    tqdm.write(
        f"#Cached Entries: {num_entries} / {dataset.num_nodes}"
        f"Ratio: {num_entries / dataset.num_nodes}"
        f"Access Cache Ratio: {sorted_counts[:num_entries].sum().item() / sorted_counts.sum().item()}"
        f"Blowup Ratio: {blowup_ratio}"
    )

f = open("fanout_blowup_fs_ig.csv", "a")
writer = csv.writer(f, lineterminator="\n")
log_info = [args.dataset, args.batchsize]
log_info += [f"{x:.3f}" for x in blowup_ratio_list]
writer.writerow(log_info)
f.close()
