import argparse
from igb.dataloader import IGB260M
from ogb.lsc import MAG240MDataset
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import torch


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="ogbn-products",
    help="which dataset to load for training",
)
parser.add_argument(
    "--path",
    type=str,
    default="/efs/rjliu/dataset/igb_dataset",
    help="path containing the datasets",
)
parser.add_argument(
    "--dataset_size",
    type=str,
    default="tiny",
    choices=["tiny", "small", "medium", "large", "full"],
    help="size of the datasets",
)
parser.add_argument(
    "--num_classes", type=int, default=19, choices=[19, 2983], help="number of classes"
)
parser.add_argument(
    "--in_memory",
    type=int,
    default=0,
    choices=[0, 1],
    help="0:read only mmap_mode=r, 1:load into memory",
)
parser.add_argument(
    "--synthetic",
    type=int,
    default=0,
    choices=[0, 1],
    help="0:nlp-node embeddings, 1:random",
)
args = parser.parse_args()

print("Loading Dataset")
if args.dataset.startswith("igb"):
    dataset = IGB260M(
        root=args.path,
        size=args.dataset_size,
        in_memory=args.in_memory,
        classes=args.num_classes,
        synthetic=args.synthetic,
    )
    node_features = dataset.paper_feat
elif args.dataset == "mag240m":
    root = "/efs/rjliu/dataset/mag240m"
    dataset = MAG240MDataset(root=root)
    paper_offset = dataset.num_authors + dataset.num_institutions
    num_nodes = paper_offset + dataset.num_papers
    num_features = dataset.num_paper_features

    node_features = np.memmap(
        os.path.join(root, "full.npy"),
        mode="r",
        dtype="float16",
        shape=(num_nodes, num_features),
    )
else:
    raise NotImplementedError
print(node_features.shape)
num_nodes = node_features.shape[0]
print("Dataset Loaded")

batch_size = 1000000
incre_pca = IncrementalPCA(n_components=128)
num_batch = (num_nodes + batch_size - 1) // batch_size
for i in tqdm(range(num_batch)):
    start = i * batch_size
    end = num_nodes if i == num_batch - 1 else (i + 1) * batch_size
    partial_feats = node_features[start:end]
    incre_pca.partial_fit(partial_feats.astype(np.float32))

# Computed mean per feature
mean = incre_pca.mean_
# and stddev
stddev = np.sqrt(incre_pca.var_)

Xtransformed = None
for i in tqdm(range(num_batch)):
    start = i * batch_size
    end = num_nodes if i == num_batch - 1 else (i + 1) * batch_size
    partial_feats = node_features[start:end]
    Xchunk = incre_pca.transform(partial_feats.astype(np.float32))
    if Xtransformed is None:
        Xtransformed = Xchunk
    else:
        Xtransformed = np.vstack((Xtransformed, Xchunk))

Xtransformed = Xtransformed.astype(np.float32)
print(Xtransformed.shape, Xtransformed.dtype)
print(np.isnan(Xtransformed).sum())

if args.dataset.startswith("igb"):
    if args.dataset_size == "large" or args.dataset_size == "full":
        path = osp.join(args.path, "full", "processed", "paper", f"node_feat_{128}.npy")
    else:
        path = osp.join(
            args.path, args.dataset_size, "processed", "paper", f"node_feat_{128}.npy"
        )
elif args.dataset == "mag240m":
    path = os.path.join(root, "full_128.npy")

local_path = f"/nvme1n1/offgs_dataset/{args.dataset}-offgs/features.npy"

print("Saving features...")
features_mmap = np.memmap(path, mode="w+", shape=Xtransformed.shape, dtype=np.float32)
features_mmap[:] = Xtransformed[:]
features_mmap.flush()
print("Done!")

print("Saving features to local...")
features_mmap_local = np.memmap(
    local_path, mode="w+", shape=Xtransformed.shape, dtype=np.float32
)
features_mmap_local[:] = Xtransformed[:]
features_mmap_local.flush()
print("Done!")
