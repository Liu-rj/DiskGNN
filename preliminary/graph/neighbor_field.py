import torch
import dgl
import time


num_nodes = 10000
# warmup the GPU
for degree in [20, 200]:
    fanout = 10
    dst = torch.repeat_interleave(torch.arange(0, num_nodes, dtype=torch.int64), degree)
    src = torch.randint(0, num_nodes, (dst.numel(),), dtype=torch.int64)
    graph = dgl.graph((src, dst)).formats("csc").to("cuda")
    for i in range(10):
        seeds = torch.randint(0, num_nodes, (1024,), dtype=torch.int64, device="cuda")
        subg = graph.sample_neighbors(seeds, fanout, replace=True)


for degree in [20, 200, 2000, 20000]:
    fanout = 10
    print(f"degree {degree} in-GPU")
    dst = torch.repeat_interleave(torch.arange(0, num_nodes, dtype=torch.int64), degree)
    src = torch.randint(0, num_nodes, (dst.numel(),), dtype=torch.int64)
    graph = dgl.graph((src, dst)).formats("csc").to("cuda")
    for i in range(10):
        seeds = torch.randint(0, num_nodes, (1024,), dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        start = time.time()
        subg = graph.sample_neighbors(seeds, fanout, replace=True)
        torch.cuda.synchronize()
        end = time.time()
        print(end - start)

for degree in [20, 200, 2000, 20000]:
    fanout = 10
    print(f"degree {degree} UVA")
    dst = torch.repeat_interleave(torch.arange(0, num_nodes, dtype=torch.int64), degree)
    src = torch.randint(0, num_nodes, (dst.numel(),), dtype=torch.int64)
    graph = dgl.graph((src, dst)).formats("csc")
    graph.pin_memory_()
    for i in range(10):
        seeds = torch.randint(0, num_nodes, (1024,), dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        start = time.time()
        subg = graph.sample_neighbors(seeds, fanout, replace=True)
        torch.cuda.synchronize()
        end = time.time()
        print(end - start)
