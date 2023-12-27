import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ubuntu/OfflineSampling/examples')
import sklearn.cluster as cluster

from load_graph import *
from model import *
from sklearn.cluster import AgglomerativeClustering,SpectralClustering
import sklearn
import psutil
import time
import json
from dataset import OffgsDataset
import csv

import offgs
class TreeNode:
    def __init__(self, value, children=None):
        self.value = value  # The set of IDs at this node
        self.children = children if children is not None else []
        # When a TreeNode is created, update the total length
        TreeBuilder.total_length += len(value)
        Jaccard_TreeBuilder.total_length += len(value)
        ## print current value
        print('current value: ',len(value))
        # print jaccard similarity length
        # print('jaccard similarity length: ',Jaccard_TreeBuilder.total_length)  

class TreeBuilder:
    total_length = 0

    @staticmethod
    def build_k_ary_tree(sequences, max_depth, current_depth=0, ancestor_values=None):
        if ancestor_values is None:
            ancestor_values = set()
        if len(sequences) == 1:
            unique_values = set(sequences[0]) - ancestor_values
            return TreeNode(list(unique_values))

        # Determine the number of children based on the depth
        if len(sequences)>70:
            k=4
        elif len(sequences)>40:
            k=3
        else:
            k=2
        print('self sequences: '+str(len(sequences)))
        print(k)
        # Split the sequences into k parts
        parts = [sequences[i::k] for i in range(k)]
        # Compute the current node's value
        current_values = set.intersection(*map(set, sequences)) - ancestor_values
        new_ancestor_values = ancestor_values | current_values

        # Recursively build the child nodes
        children = [TreeBuilder.build_k_ary_tree(part, max_depth, current_depth + 1, new_ancestor_values) for part in parts if part]

        # Create the current node
        return TreeNode(list(current_values), children)

class Jaccard_TreeBuilder:
    total_length = 0
    leaf_node_cnt=0
    @staticmethod
    def jaccard_similarity(t1, t2):
        combined = torch.cat((t1, t2))
        uniques, counts = combined.unique(return_counts=True)
        intersection = uniques[counts > 1]
        similarity = len(intersection) / len(uniques)
        return similarity
    @staticmethod
    def calculate_similarity_matrix(sequences):
        size = len(sequences)
        ## change sequences to tensor in cuda
        # sequences = [torch.tensor(seq).to('cuda') for seq in sequences]
        similarity_matrix = torch.zeros(size, size).to('cuda')        
        for i in range(size):
            for j in range(size):
                if i != j:
                    similarity_matrix[i][j] = Jaccard_TreeBuilder.jaccard_similarity(sequences[i], sequences[j])
                else:
                    similarity_matrix[i][j] = 1  # Maximum similarity with itself
        return similarity_matrix

    @staticmethod
    def cluster_sequences(sequences, k, clustermethod='spectral'):
        similarity_matrix = Jaccard_TreeBuilder.calculate_similarity_matrix(sequences)
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        if clustermethod=='spectral':
            clustering = SpectralClustering(n_clusters=k, affinity='precomputed')
        else:
            clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete')
        labels = clustering.fit_predict(distance_matrix.cpu())

        clusters = [[] for _ in range(k)]
        for sequence, label in zip(sequences, labels):
            clusters[label].append(sequence)
        for i,cluster in enumerate(clusters):
            print('i: ',i,len(cluster))
        return clusters

    @staticmethod
    def build_k_ary_tree(sequences, max_depth, current_depth=0, ancestor_values=None):
        if ancestor_values is None:
            ancestor_values = set()
        if len(sequences) == 1:
            unique_values = set(sequences[0].tolist()) - ancestor_values
            TreeBuilder.total_length += len(unique_values)
            print('current value: ',len(unique_values))
            Jaccard_TreeBuilder.leaf_node_cnt+=1
            print('leaf node cnt: ',Jaccard_TreeBuilder.leaf_node_cnt)
            return TreeNode(list(unique_values))

        # Determine the number of children dynamically
        if len(sequences)>70:
            k=4
        elif len(sequences)>40:
            k=4
        elif len(sequences)>11:
            k=3
        else:
            k=2
        print(k)
        
        # Cluster the sequences into k groups based on similarity
        clusters = Jaccard_TreeBuilder.cluster_sequences(sequences, k)
        ## change the sequences in numpy and use numpy to calculate the intersection
        sequences = [seq.tolist() for seq in sequences]
        # Compute the current node's value
        current_values =  set.intersection(*map(set, sequences)) - ancestor_values
        ## print len current_values
        ## print len sequences
        print('len sequences: '+str(len(sequences)))
        new_ancestor_values = ancestor_values | current_values
        
        # Recursively build the child nodes
        children = [Jaccard_TreeBuilder.build_k_ary_tree(cluster, max_depth, current_depth + 1, new_ancestor_values) for cluster in clusters]
        
        # Create the current node
        print('len current values: '+str(len(current_values)))
        
        return TreeNode(list(current_values), children)

## add print tree function
def print_tree(node, depth=0):
    # Helper function to print the tree structure
    print(' ' * depth * 2, node.value)
    if node.left:
        print_tree(node.left, depth + 1)
    if node.right:
        print_tree(node.right, depth + 1)

def intersection(t1,t2):
        combined = torch.cat((t1, t2))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]
        intersection = uniques[counts > 1]
        del combined, uniques, counts
        return -intersection
def run(dataset, args):
    output_dir = f"{args.store_path}/{args.dataset}-{args.batchsize}--{args.fanout}"
    aux_dir = f"{output_dir}/cache-size-{args.feat_cache_size}"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(aux_dir):
        os.mkdir(aux_dir)

    features = dataset.mmap_features

    sorted_idx = torch.load(f"{output_dir}/meta_node_popularity.pt").cpu()
    node_cnt = torch.load(f"{output_dir}/node_counts.pt").cpu()
    feat_load_time, nid_load_time, difference_time, save_time = 0, 0, 0, 0
    clear_cache_time = 0

    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")

    start = time.time()

    tic = time.time()
    num_batches = (dataset.split_idx["train"].numel() + args.batchsize - 1) // args.batchsize
    table_size = 4 * dataset.num_nodes
    num_entries = min((args.feat_cache_size - table_size) // (4 * features.shape[1]), dataset.num_nodes)
    if num_entries > torch.iinfo(torch.int32).max:
        raise ValueError
    print(f"#Cached Entries: {num_entries}")
    cache_indices = sorted_idx[:num_entries]
    address_table = torch.full((dataset.num_nodes,), -1, dtype=torch.int32)
    address_table[cache_indices] = torch.arange(num_entries, dtype=torch.int32)
    torch.save(cache_indices, f"{aux_dir}/cached_nodes.pt")
    torch.save(address_table, f"{aux_dir}/address_table.pt")
    cache_init_time = time.time() - tic

    last_node_id=sorted_idx[num_entries]
    node_cnt = node_cnt[last_node_id]
    print(f"last_node_id: {last_node_id}, node_cnt: {node_cnt}")
    cache_indices = cache_indices.to("cuda")
    
    
    ## we can optimize the difference set process
    cold_index_strs=[]
    for i in trange(num_batches):
        tic = time.time()
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")
        clear_cache_time += time.time() - tic

        tic = time.time()
        input_nodes: torch.Tensor = torch.load(f"{output_dir}/in-nid-{i}.pt").to("cuda")
        nid_load_time += time.time() - tic

        tic = time.time()
        # input_nodes = blocks[0].srcdata[dgl.NID].to("cuda")
        # cold_nodes, rev_cold_idx = torch.ops.offgs._CAPI_Difference(input_nodes, cache_indices)
        # hot_nodes, rev_hot_idx = torch.ops.offgs._CAPI_Difference(input_nodes, cold_nodes)

        rev_hot_idx = torch.isin(input_nodes, cache_indices, assume_unique=True).nonzero(as_tuple=True)[0]
        hot_nodes = input_nodes[rev_hot_idx]
        rev_cold_idx = torch.isin(input_nodes, hot_nodes, assume_unique=True, invert=True).nonzero(as_tuple=True)[0]
        cold_nodes = input_nodes[rev_cold_idx]
        torch.cuda.synchronize()
        difference_time += time.time() - tic
        cold_index_strs.append(cold_nodes)
    sequences = [seq.tolist() for seq in cold_index_strs]
    switch_type_tree='random'
    ##TODO graph based method
    
    if switch_type_tree=='jaccard':
        ## change sequences to tensor in cuda
        sequences = [torch.tensor(seq).to('cuda') for seq in sequences]
        # Build the tree
        root = Jaccard_TreeBuilder.build_k_ary_tree(sequences,6)
        print('Total length of all current_values:', Jaccard_TreeBuilder.total_length)
    elif switch_type_tree=='random':
        # Build the tree
        root = TreeBuilder.build_k_ary_tree(sequences,6)

        # Access the total length
        print("Total length of all current_values:", TreeBuilder.total_length)
    
    pre_total_len=0
    for seq in sequences:
        pre_total_len+=len(seq)
    print('pre total length: '+str(pre_total_len))
    
#Total length of all current_values: 39372966
#pre total length: 44017255
        # tic = time.time()
        # packed_feats: torch.Tensor = features[cold_nodes.long().cpu()]
        # # packed_feats = torch.ops.offgs._CAPI_GatherMemMap(features, cold_nodes.cpu(), dataset.num_features)
        # # packed_feats = torch.ops.offgs._CAPI_GatherPRead(dataset.features_path, cold_nodes.cpu(), dataset.num_features)
        # feat_load_time += time.time() - tic

        # tic = time.time()
        # aux_data = torch.cat([packed_feats.flatten(), cold_nodes.cpu(), hot_nodes.cpu(), rev_hot_idx.cpu(), rev_cold_idx.cpu()])
        # stored_data = np.memmap(f"{aux_dir}/train-aux-{i}.npy", mode='w+', shape=aux_data.numel() + 5, dtype=np.float32)
        # stored_data[:5] = [packed_feats.numel(), cold_nodes.numel(), hot_nodes.numel(), rev_hot_idx.numel(), rev_cold_idx.numel()]
        # stored_data[5:] = aux_data
        # stored_data.flush()
        # save_time += time.time() - tic

    total_time = time.time() - start
    # with open("logs/pack_decompose.csv", "a") as f:
    #     writer = csv.writer(f, lineterminator="\n")
    #     log_info = [
    #         round(cache_init_time, 3),
    #         round(blocks_load_time, 3),
    #         round(difference_time, 3),
    #         round(feat_load_time, 3),
    #         round(save_time, 3),
    #         round(total_time, 3),
    #     ]
    #     writer.writerow(log_info)

    print(
        f"Init Cache Indices Time: {cache_init_time:.3f}\t"
        f"Drop Cache Time: {clear_cache_time:.3f}\t"
        f"NID Load Time: {nid_load_time:.3f}\t"
        f"Set Difference Time: {difference_time:.3f}\t"
        f"Feat Load Time: {feat_load_time:.3f}\t"
        f"Save Time: {save_time:.3f}\t"
        f"Total Time: {(total_time - clear_cache_time):.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-products", help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=5000, help="batch size for training")
    parser.add_argument("--fanout", type=str, default="10,10", help="sampling fanout")
    parser.add_argument("--store-path", default="/nvme2n1", help="path to store subgraph")
    parser.add_argument("--feat-cache-size", type=int, default=200000000, help="cache size in bytes")
    args = parser.parse_args()
    print(args)

    # --- load data --- #
    dataset_path = f"{args.store_path}/{args.dataset}-offgs"
    dataset = OffgsDataset(dataset_path)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem, "GB")

    run(dataset, args)
