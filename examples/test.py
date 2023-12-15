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

# cold_feats, cold_nodes, hot_nodes, rev_hot_idx, rev_cold_idx = torch.ops.offgs._CAPI_LoadFeats_ODirect("/nvme2n1/feature_test.npy", 1, 10240)
# print(cold_feats)
# print(cold_nodes)
# print(hot_nodes)
# print(rev_hot_idx)
# print(rev_cold_idx)

# with open("/proc/sys/vm/drop_caches", "w") as stream:
#     stream.write("1\n")

# tic = time.time()
# feats = torch.ops.offgs._CAPI_LoadTensor("/nvme2n1/ogbn-products-10,10,10/cache-size-200000000/train-aux-0.npy")
# print(time.time() - tic)
# print(features.size == feats.numpy().size)
# print(features == feats.numpy())
# print(features)
# print(feats.numpy())

# with open("/proc/sys/vm/drop_caches", "w") as stream:
#     stream.write("1\n")

# tic = time.time()
# feats = torch.ops.offgs._CAPI_LoadFeats("/nvme2n1/ogbn-products-10,10,10/cache-size-200000000/train-aux-0.npy", 100, 8, 64)
# print(time.time() - tic)

# with open("/proc/sys/vm/drop_caches", "w") as stream:
#     stream.write("1\n")

# tic = time.time()
# feats = torch.ops.offgs._CAPI_LoadFeats_ODirect("/nvme2n1/ogbn-products-10,10,10/cache-size-200000000/train-aux-0.npy", 100, 8, 64)
# print(time.time() - tic)

# for num_align in [1, 2, 4, 6, 8, 10, 12, 16, 32, 64, 128, 256, 512, 1024, 5120, 10240]:
# for num_align in [8, 10240]:
#     print(num_align, (num_align * 4) / 1024, "MB")
#     tic = time.time()
#     res = torch.ops.offgs._CAPI_LoadFeats_ODirect("/nvme2n1/ogbn-products-10,10,10/cache-size-200000000/train-aux-0.npy", 100, num_align)
#     # feats = torch.ops.offgs._CAPI_LoadFeats_ODirect("/nvme2n1/test", 100, num_align)
#     print(time.time() - tic)
#     print("IO time:", res[-1])

# for epoch in range(3):
#     clear_cache, feats_load_time = 0, 0
#     total_io_vol = 0

#     # tic = time.time()
#     # # Same effect of `sysctl -w vm.drop_caches=1`
#     # # Requires sudo
#     # with open("/proc/sys/vm/drop_caches", "w") as stream:
#     #     stream.write("1\n")
#     # clear_cache += time.time() - tic

#     torch.cuda.synchronize()
#     start = time.time()
#     for i in trange(1208):
#         torch.cuda.synchronize()

#         tic = time.time()
#         cold_feats, cold_nodes, hot_nodes, rev_hot_idx, rev_cold_idx = torch.ops.offgs._CAPI_LoadFeats_ODirect(
#             f"/nvme2n1/ogbn-papers100M-10,10,10/cache-size-10000000000/train-aux-{11}.npy", 128, 8
#         )
#         feats_load_time += time.time() - tic

#         print(cold_feats)
#         print(cold_nodes)
#         print(hot_nodes)
#         print(rev_cold_idx)
#         print(rev_hot_idx)
#         print((rev_hot_idx < 0).any())
#         exit()

#     torch.cuda.synchronize()
#     epoch_time = time.time() - start
#     print(
#         f"IO Volume: {total_io_vol}\t"
#         f"Feature Load Time: {feats_load_time:.3f}\t"
#         f"Epoch Time: {(epoch_time - clear_cache):.3f}\t"
#         f"Drop Cache Time: {clear_cache:.3f}"
#     )

# with open("/proc/sys/vm/drop_caches", "w") as stream:
#     stream.write("1\n")

# total_time = 0
# volume = 0
# for i in trange(200):
#     # with open("/proc/sys/vm/drop_caches", "w") as stream:
#     #     stream.write("1\n")

#     size = np.random.randint(800000, 1000000)
#     cold_size = int(size * 0.8)
#     hot_size = size - cold_size
#     rev_cold_idx = torch.randint(0, size, (cold_size,), dtype=torch.int32)
#     cold_feats = torch.randn((cold_size, 100), dtype=torch.float32)
#     # rev_hot_idx = torch.randint(0, size, (hot_size,), dtype=torch.int32)
#     # hot_feats = torch.randn((hot_size, 100), dtype=torch.float32)
#     x = torch.empty((size, 100), dtype=torch.float32)

#     tic = time.time()
#     x[rev_cold_idx] = cold_feats
#     # x[rev_hot_idx] = cached_feats[address_table[hot_nodes]]
#     # x[rev_hot_idx] = hot_feats
#     total_time += time.time() - tic

#     volume += cold_feats.shape[0] * cold_feats.shape[1]

# print(total_time)
# print((volume * 4) / (1024 * 1024 * 1024), "GB")

# size = 430385197 // 641

# for i in trange(10):
#     a = torch.rand((size, 128), dtype=torch.float32)
#     a = a.to("cuda")

# total_time = 0
# for i in trange(64):
#     a = torch.empty((size, 128), dtype=torch.float32, pin_memory=True)
#     torch.cuda.synchronize()
#     tic = time.time()
#     a = a.to("cuda")
#     torch.cuda.synchronize()
#     total_time += time.time() - tic
# print(total_time)

# temp = torch.rand(1000000000, dtype=torch.float32)
# torch.save(temp, "/nvme1n1/torch_testfile.pt")

with open("/proc/sys/vm/drop_caches", "w") as stream:
    stream.write("3\n")

tic = time.time()
temp = torch.load("/nvme1n1/torch_testfile.pt")
print(time.time() - tic)


# tic = time.time()
# for i in trange(640):
#     blocks = torch.load(f"/nvme1n1/friendster-1024-10,10,10/train-{i}.pt")
#     output_nodes = torch.load(f"/nvme1n1/friendster-1024-10,10,10/out-nid-{i}.pt")
# print(time.time() - tic)
