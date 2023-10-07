import dgl
import torch
import time
import offgs
import numpy as np
from tqdm import tqdm, trange
import os

# with open("/proc/sys/vm/drop_caches", "w") as stream:
#     stream.write("1\n")

# tic = time.time()
# data, _ = dgl.load_graphs("/nvme1n1/dataset/friendster/friendster.bin")
# print(time.time() - tic)

# g: dgl.DGLGraph = data[0].long()

# with open("/proc/sys/vm/drop_caches", "w") as stream:
#     stream.write("1\n")

# tic = time.time()
# torch.save(g, "/nvme1n1/friendster.pt")
# print(time.time() - tic)

# with open("/proc/sys/vm/drop_caches", "w") as stream:
#     stream.write("1\n")

# tic = time.time()
# g = torch.load("/nvme1n1/friendster.pt")
# print(time.time() - tic)


# print("Saving features...")
# packed_feats = torch.tensor([0.51, 0.62], dtype=torch.float32)
# cold_nodes = torch.tensor([1, 2], dtype=torch.int64)
# hot_nodes = torch.tensor([3, 4], dtype=torch.int64)
# rev_hot_idx = torch.tensor([5, 6], dtype=torch.int64)
# rev_cold_idx = torch.tensor([7, 8], dtype=torch.int64)
# aux_data = torch.cat([packed_feats.flatten(), cold_nodes.cpu(), hot_nodes.cpu(), rev_hot_idx.cpu(), rev_cold_idx.cpu()])
# stored_data = np.memmap("/nvme2n1/feature_test.npy", mode="w+", shape=aux_data.numel() + 5, dtype=np.float32)
# stored_data[:5] = [packed_feats.numel(), cold_nodes.numel(), hot_nodes.numel(), rev_hot_idx.numel(), rev_cold_idx.numel()]
# stored_data[5:] = aux_data
# stored_data.flush()
# print("Done!")
# print(stored_data)

# cold_feats, cold_nodes, hot_nodes, rev_hot_idx, rev_cold_idx = torch.ops.offgs._CAPI_LoadFeats("/nvme2n1/feature_test.npy", 1, 8, 64)
# print(cold_feats)
# print(cold_nodes)
# print(hot_nodes)
# print(rev_hot_idx)
# print(rev_cold_idx)

# cold_feats, cold_nodes, hot_nodes, rev_hot_idx, rev_cold_idx = torch.ops.offgs._CAPI_LoadFeats_ODirect("/nvme2n1/feature_test.npy", 1, 8, 64)
# print(cold_feats)
# print(cold_nodes)
# print(hot_nodes)
# print(rev_hot_idx)
# print(rev_cold_idx)

with open("/proc/sys/vm/drop_caches", "w") as stream:
    stream.write("1\n")

tic = time.time()
feats = torch.ops.offgs._CAPI_LoadTensor("/nvme2n1/ogbn-products-10,10,10/cache-size-200000000/train-aux-0.npy")
print(time.time() - tic)
# print(features.size == feats.numpy().size)
# print(features == feats.numpy())
# print(features)
# print(feats.numpy())

with open("/proc/sys/vm/drop_caches", "w") as stream:
    stream.write("1\n")

tic = time.time()
feats = torch.ops.offgs._CAPI_LoadFeats("/nvme2n1/ogbn-products-10,10,10/cache-size-200000000/train-aux-0.npy", 100, 8, 64)
print(time.time() - tic)

with open("/proc/sys/vm/drop_caches", "w") as stream:
    stream.write("1\n")

tic = time.time()
feats = torch.ops.offgs._CAPI_LoadFeats_ODirect("/nvme2n1/ogbn-products-10,10,10/cache-size-200000000/train-aux-0.npy", 100, 8, 64)
print(time.time() - tic)

# for omp_threads in [1, 4, 8, 16, 32, 64, 128, 256]:
#     for num_align in [1, 2, 4, 6, 8, 10, 12, 16]:
#         with open("/proc/sys/vm/drop_caches", "w") as stream:
#             stream.write("1\n")

#         print(omp_threads, num_align)
#         tic = time.time()
#         feats = torch.ops.offgs._CAPI_LoadFeats("/nvme2n1/ogbn-products-10,10,10/cache-size-200000000/train-aux-0.npy", 100, num_align, omp_threads)
#         print(time.time() - tic)

for epoch in range(3):
    clear_cache, feats_load_time = 0, 0
    total_io_vol = 0

    torch.cuda.synchronize()
    start = time.time()

    for i in trange(197):
        torch.cuda.synchronize()
        tic = time.time()
        # Same effect of `sysctl -w vm.drop_caches=1`
        # Requires sudo
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")
        clear_cache += time.time() - tic

        tic = time.time()
        cold_feats, cold_nodes, hot_nodes, rev_hot_idx, rev_cold_idx = torch.ops.offgs._CAPI_LoadFeats_ODirect(
            f"/nvme2n1/ogbn-products-10,10,10/cache-size-200000000/train-aux-{i}.npy", 100, 8, 64
        )
        feats_load_time += time.time() - tic

    torch.cuda.synchronize()
    epoch_time = time.time() - start
    print(
        f"IO Volume: {total_io_vol}\t"
        f"Feature Load Time: {feats_load_time:.3f}\t"
        f"Epoch Time: {(epoch_time - clear_cache):.3f}\t"
        f"Drop Cache Time: {clear_cache:.3f}"
    )
