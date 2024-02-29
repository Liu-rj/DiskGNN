import argparse
import numpy as np
import torch
import os.path as osp
import os
from offgs.dataset import OffgsDataset

import dgl
from dgl.data import DGLDataset
import warnings

warnings.filterwarnings("ignore")


class IGB260M(object):
    def __init__(
        self, root: str, size: str, in_memory: int, classes: int, synthetic: int
    ):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes

    def num_nodes(self):
        if self.size == "experimental":
            return 100000
        elif self.size == "small":
            return 1000000
        elif self.size == "medium":
            return 10000000
        elif self.size == "large":
            return 100000000
        elif self.size == "full":
            return 269346174

    @property
    def paper_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        # TODO: temp for bafs. large and full special case
        if self.size == "large" or self.size == "full":
            path = osp.join(self.dir, "full", "processed", "paper", "node_feat_128.npy")
            emb = np.memmap(path, dtype="float32", mode="r", shape=(num_nodes, 128))
        else:
            path = osp.join(self.dir, self.size, "processed", "paper", "node_feat.npy")
            if self.synthetic:
                emb = np.random.rand(num_nodes, 1024).astype("f")
            else:
                if self.in_memory:
                    emb = np.load(path)
                else:
                    emb = np.load(path, mmap_mode="r")

        return emb

    @property
    def paper_label(self) -> np.ndarray:
        if self.size == "large" or self.size == "full":
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
                path = osp.join(
                    self.dir, "full", "processed", "paper", "node_label_19.npy"
                )
                node_labels = np.memmap(
                    path, dtype="float32", mode="r", shape=(num_nodes)
                )
                # Actual number 227130858
            else:
                path = osp.join(
                    self.dir, "full", "processed", "paper", "node_label_2K.npy"
                )
                node_labels = np.memmap(
                    path, dtype="float32", mode="r", shape=(num_nodes)
                )
                # Actual number 157675969

        else:
            if self.num_classes == 19:
                path = osp.join(
                    self.dir, self.size, "processed", "paper", "node_label_19.npy"
                )
            else:
                path = osp.join(
                    self.dir, self.size, "processed", "paper", "node_label_2K.npy"
                )
            if self.in_memory:
                node_labels = np.load(path)
            else:
                node_labels = np.load(path, mmap_mode="r")
        return node_labels

    @property
    def paper_edge(self) -> np.ndarray:
        path = osp.join(
            self.dir, self.size, "processed", "paper__cites__paper", "edge_index.npy"
        )
        # if self.size == 'full':
        #     path = '/mnt/nvme15/IGB260M_part_2/full/processed/paper__cites__paper/edge_index.npy'
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode="r")


class IGB260MDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name="IGB260MDGLDataset")

    def process(self):
        dataset = IGB260M(
            root=self.dir,
            size=self.args.dataset_size,
            in_memory=self.args.in_memory,
            classes=self.args.num_classes,
            synthetic=self.args.synthetic,
        )

        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        self.graph = dgl.graph(
            (node_edges[:, 0], node_edges[:, 1]), num_nodes=node_features.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels

        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)

        if self.args.dataset_size == "full":
            # TODO: Put this is a meta.pt file
            if self.args.num_classes == 19:
                n_labeled_idx = 227130858
            else:
                n_labeled_idx = 157675969

            n_nodes = node_features.shape[0]
            n_train = int(n_labeled_idx * 0.6)
            n_val = int(n_labeled_idx * 0.2)

            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)

            train_mask[:n_train] = True
            val_mask[n_train : n_train + n_val] = True
            test_mask[n_train + n_val : n_labeled_idx] = True

            self.graph.ndata["train_mask"] = train_mask
            self.graph.ndata["val_mask"] = val_mask
            self.graph.ndata["test_mask"] = test_mask
        else:
            n_nodes = node_features.shape[0]
            n_train = int(n_nodes * 0.6)
            n_val = int(n_nodes * 0.2)

            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)

            train_mask[:n_train] = True
            val_mask[n_train : n_train + n_val] = True
            test_mask[n_train + n_val :] = True

            self.graph.ndata["train_mask"] = train_mask
            self.graph.ndata["val_mask"] = val_mask
            self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)


def gen_train(ratio=0.01, name="train_001.pth"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="which dataset to load for training",
    )
    parser.add_argument(
        "--store-path", default="/nvme1n1/offgs_dataset", help="path to store subgraph"
    )
    args = parser.parse_args()
    print(args)

    # --- load dataset --- #
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    dataset = OffgsDataset(dataset_path)

    num_nodes = dataset.num_nodes
    train_001 = torch.randperm(num_nodes)[: int(num_nodes * ratio)]
    print(f"Ratio: {train_001.numel() / num_nodes}")
    torch.save(train_001, os.path.join(dataset_path, name))


def kill_proc(p):
    try:
        p.terminate()
    except Exception:
        pass
