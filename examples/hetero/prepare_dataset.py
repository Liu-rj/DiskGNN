from collections import defaultdict
import torch
import dgl.graphbolt as gb
import numpy as np
import json
import argparse
import os


def load_dataset(dataset_name):
    """Load the dataset and return the graph, features, train/valid/test sets
    and the number of classes.

    Here, we use `BuiltInDataset` to load the dataset which returns graph,
    features, train/valid/test sets and the number of classes.
    """
    dataset = gb.BuiltinDataset(
        dataset_name, root="/nvme2n1/graphbolt_dataset/datasets"
    ).load()
    print(f"Loaded dataset: {dataset.tasks[0].metadata['name']}")

    graph = dataset.graph
    features = dataset.feature
    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    test_set = dataset.tasks[0].test_set
    num_classes = dataset.tasks[0].metadata["num_classes"]

    labels = {}
    splitted_idx = {"train": {}, "test": {}, "valid": {}}
    for it, node_set in enumerate([train_set, valid_set, test_set]):
        for type, item_set in node_set[:].items():
            nid, label = item_set
            set_type = ["train", "valid", "test"][it]
            splitted_idx[set_type][type] = nid
            if type not in labels:
                labels[type] = torch.full(
                    (graph.num_nodes[type],), -1, dtype=torch.int64
                )
            labels[type][nid] = label.long()

    # note: not all node types have features
    feature_dim = features.size("node", "paper", "feat")[0]
    feature_dtype = features.read("node", "paper", "feat", torch.tensor([0])).dtype
    all_node_feats = [torch.tensor([])] * len(list(graph.num_nodes))
    node_counts = [0] * len(list(graph.num_nodes))
    for node_type, num_nodes in graph.num_nodes.items():
        idx = graph.node_type_to_id[node_type]
        node_counts[idx] = graph.num_nodes[node_type]
        if ("node", node_type, "feat") in features.keys():
            print("node", node_type, "feat")
            all_node_feats[idx] = features.read("node", node_type, "feat")
        else:
            # generate random features for node types that do not have features
            all_node_feats[idx] = torch.randn(
                num_nodes, feature_dim, dtype=feature_dtype
            )
    all_node_feats = torch.cat(all_node_feats, dim=0)

    # generate feature offsets for each type
    all_offsets = torch.tensor([0] + np.cumsum(node_counts).tolist())
    offset_dict = {
        type_name: all_offsets[graph.node_type_to_id[type_name]].item()
        for type_name in graph.num_nodes.keys()
    }

    return (
        graph,
        all_node_feats,
        labels,
        num_classes,
        splitted_idx,
        offset_dict,
    )


def run(args, dataset, label_offset):
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    category = "paper"

    g, features, labels, n_classes, splitted_idx, offset_dict = dataset
    print(g)
    print(labels)
    print(splitted_idx)
    print(f"features dtype: {features.dtype}")
    print(f"features shape: {features.shape}")
    print(
        "training nodes ratio:",
        splitted_idx["train"][category].numel() / g.total_num_nodes,
    )

    os.makedirs(dataset_path, exist_ok=True)
    graph_path = os.path.join(dataset_path, "graph.pth")
    features_path = os.path.join(dataset_path, "features.bin")
    labels_path = os.path.join(dataset_path, "labels.pth")
    conf_path = os.path.join(dataset_path, "conf.json")
    split_idx_path = os.path.join(dataset_path, "split_idx.pth")

    # print("Saving graph...")
    # g = g.formats("coo")
    # torch.save(g, graph_path)
    # print("Done!")

    print("Saving features...")
    features.numpy().tofile(features_path)
    print("Done!")

    print("Saving labels...")
    torch.save(labels, labels_path)
    print("Done!")

    print("Making conf file...")
    mmap_config = dict()
    mmap_config["total_num_nodes"] = g.total_num_nodes
    mmap_config["num_nodes"] = g.num_nodes
    mmap_config["features_shape"] = tuple(features.shape)
    mmap_config["features_dtype"] = str(features.dtype)
    mmap_config["feat_itemsize"] = features.element_size()
    mmap_config["labels_shape"] = tuple(labels[category].shape)
    mmap_config["labels_dtype"] = str(labels[category].dtype)
    mmap_config["num_classes"] = int(n_classes)
    mmap_config["label_offset"] = int(label_offset)
    mmap_config["node_offsets"] = offset_dict
    mmap_config["ntypes"] = list(g.node_type_to_id.keys())
    mmap_config["etypes"] = list(g.edge_type_to_id.keys())
    json.dump(mmap_config, open(conf_path, "w"))
    print(mmap_config)
    print("Done!")

    print("Saving split index...")
    torch.save(splitted_idx, split_idx_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for GraphBolt")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--store-path", type=str, default="./data")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    run(args, dataset, 0)
