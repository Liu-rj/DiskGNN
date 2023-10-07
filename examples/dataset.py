import os
import json
import numpy as np
import torch


class OffgsDataset:
    def __init__(self, path):
        self.features_path = os.path.join(path, "features.npy")
        self.labels_path = os.path.join(path, "labels.npy")
        self.split_idx_path = os.path.join(path, "split_idx.pth")
        conf_path = os.path.join(path, "conf.json")
        self.conf = json.load(open(conf_path, "r"))

        self.num_nodes = self.conf["num_nodes"]
        self.num_features = self.conf["features_shape"][1]
        self.num_classes = self.conf["num_classes"]

    @property
    def labels(self):
        return torch.from_numpy(np.load(self.labels_path))

    @property
    def mmap_features(self):
        features_shape = self.conf["features_shape"]
        features_shape[1] = self.num_features
        features = np.memmap(self.features_path, mode="r", shape=tuple(features_shape), dtype=self.conf["features_dtype"])
        features = torch.from_numpy(features)
        return features

    @property
    def split_idx(self):
        return torch.load(self.split_idx_path)
