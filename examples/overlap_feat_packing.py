import torch
from dgl.dataloading import DataLoader, NeighborSampler
import argparse
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from load_graph import *
from model import *
import psutil
import time
import json
from dataset import OffgsDataset
import csv

import offgs
def find_unique_overlaps(sequences):
    ## convert to list
    sequences = [seq.tolist() for seq in sequences]
    ## print every sequence len
    for i in range(len(sequences)):
        print(f"Sequence {i} len: {len(sequences[i])}")
    # Convert all index strings to sets to remove duplicates and for set operations
    sets = [set(seq) for seq in sequences]
    num_seqs = len(sets)
    unique_overlaps = []
    remaining_parts = []
    overlap_matrix_len = [[None for _ in range(num_seqs)] for _ in range(num_seqs)]
    overlap_matrix_index = [[None for _ in range(num_seqs)] for _ in range(num_seqs)]

    # Calculate overlaps for all possible pair combinations
    for i in range(len(sets)):
        remaining = sets[i].copy()  # Copy the current set to track remaining parts
        for j in range(i+1, len(sets)):
            # Find the intersection of two sets
            overlap = sets[i] & sets[j]
            
            # Eliminate elements that have already overlapped
            unique_overlap = set()
            for item in overlap:
                if all(item not in s for k, s in enumerate(sets) if k not in [i, j]):
                    unique_overlap.add(item)

            # Record the size of elements that only overlap once in an upper triangular matrix
            overlap_size = len(unique_overlap)
            overlap_matrix_len[i][j] = overlap_size 
            overlap_matrix_index[i][j] = unique_overlap

            # Add elements that only overlap once to the result list
            unique_overlaps.append(((i, j), unique_overlap))

            # Remove this overlap from both the current set and the corresponding set
            sets[i] -= unique_overlap
            sets[j] -= unique_overlap
            remaining -= unique_overlap

        # Add the remaining parts
        remaining_parts.append((i, remaining))
    
    # Set diagonal entries to the size of the remaining parts of each sequence
    for i in range(len(sets)):
        overlap_matrix_len[i][i] = len(remaining_parts[i][1])
        overlap_matrix_index[i][i] = remaining_parts[i][1]

    # Return unique overlaps, remaining parts of each sequence, and the upper triangular matrices
    return unique_overlaps, remaining_parts, overlap_matrix_len, overlap_matrix_index
def compute_cumulative_upper_triangular(matrix):
    matrix_np = np.array(matrix, dtype=np.object)
    matrix_np[matrix_np == None] = 0
    matrix_np = matrix_np.astype(int)

    cumulative_matrix = np.zeros_like(matrix_np)
    for i in range(matrix_np.shape[0]):
        for j in range( matrix_np.shape[1]):
            cumulative_matrix[i][j] = matrix_np[i][j]
            if j > 0:
                cumulative_matrix[i][j] += cumulative_matrix[i][j-1]
            if i>0 and j== 0:
                cumulative_matrix[i][j] += cumulative_matrix[i-1][-1]
           

    return cumulative_matrix

def run(dataset, args):
    output_dir = f"{args.store_path}/{args.dataset}-{args.fanout}"
    aux_dir = f"{output_dir}/cache-size-{args.feat_cache_size}"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(aux_dir):
        os.mkdir(aux_dir)

    features = dataset.mmap_features

    sorted_idx = torch.load(f"{output_dir}/meta_node_popularity.pt").cpu()

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
    overlaps, remainings, overlap_matrix,overlap_matrix_index = find_unique_overlaps(cold_index_strs)
    for idx_pair, unique_overlap in overlaps:
        print(f"Unique overlap between sequence {idx_pair[0]} and {idx_pair[1]}: {sorted(list(unique_overlap))}")

    for idx, remaining in remainings:
        print(f"Remaining elements of sequence {idx}: {sorted(list(remaining))}")

    print("Overlap matrix (upper triangular):")
    for row in overlap_matrix:
        print(row)
    cumulative_matrix = compute_cumulative_upper_triangular(overlap_matrix)-1
    print("Cumulative matrix (upper triangular):")
    for row in cumulative_matrix:
        print(list(row))
    # Function to extract features based on the overlap index matrix
    def extract_features(overlap_matrix_index, feature):
        # Create a list to hold the selected features
        selected_features = []

        # Convert the PyTorch tensor to a NumPy array
        feature_np = feature.numpy()

        # Traverse the index matrix of the upper triangular matrix
        for i in range(len(overlap_matrix_index)):
            for j in range(i, len(overlap_matrix_index[i])):
                if overlap_matrix_index[i][j] is not None:
                    # Iterate through each index in the set
                    for index in overlap_matrix_index[i][j]:
                        # Check if the index is within a valid range
                        if index < len(feature_np):
                            # Retrieve the feature
                            selected_features.append(feature_np[index])

        # Convert the list to a numpy array
        return np.vstack(selected_features)

    # Example usage
    extracted_features = extract_features(overlap_matrix_index, feature)
    print(extracted_features)
    print(extracted_features.shape)
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
    parser.add_argument("--batchsize", type=int, default=1000, help="batch size for training")
    parser.add_argument("--fanout", type=str, default="10,10,10", help="sampling fanout")
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
