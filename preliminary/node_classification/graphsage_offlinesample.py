import torch
import dgl
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from load_graph import *
from model import *
import psutil
import os


def train_dgl(dataset, args):
    device = args.device
    fanout = [int(x) for x in args.fanout.split(",")]
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats("csc")
    train_nid, val_nid, _ = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    # features, labels = features.pin_memory(), labels.pin_memory()

    g = g.to("cpu") if args.use_uva else g.to(device)
    train_nid, val_nid = train_nid.to(device), val_nid.to(device)
    num_layers = len(fanout)
    # if args.model == "SAGE":
    #     model = SAGEModel(features.shape[1], 512, n_classes, num_layers).to(device)
    # elif args.model == "GAT":
    #     model = GAT(features.shape[1], 512, n_classes, [8, 2]).to(device)
    sampler = NeighborSampler(fanout)
    train_dataloader = DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        device=torch.device(device),
        use_uva=args.use_uva,
    )
    # val_dataloader = DataLoader(
    #     g,
    #     val_nid,
    #     sampler,
    #     batch_size=args.batchsize,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=args.num_workers,
    #     device=torch.device(device),
    #     use_uva=args.use_uva,
    # )
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    sampled_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device=device)
    torch.cuda.synchronize()
    process = psutil.Process(os.getpid())
    before_mem = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Start Pre-sample")
    tic = time.time()
    # blocks_pool = []
    for i in tqdm(range(args.num_sample)):
        input_nodes, output_nodes, blocks = next(iter(train_dataloader))
        sampled_counts[input_nodes] += 1
        # blocks = [block.to("cpu") for block in blocks]
        # blocks_pool.append(blocks)
    torch.cuda.synchronize()
    presample_mem = process.memory_info().rss / (1024 * 1024 * 1024) - before_mem
    print("Memory of pre-sampled sub-graphs:", presample_mem, "GB")

    sampled_counts = sampled_counts.to("cpu")
    ordered_counts, idx = torch.sort(sampled_counts, dim=0, descending=True)
    cache_percent = 0.01
    print(
        f"Fanout: {args.fanout}, cache percent.: {cache_percent}, feature percent.:",
        torch.sum(ordered_counts[int(ordered_counts.shape[0] * cache_percent) :]).item()
        / g.num_nodes(),
    )
    cache_percent = 0.05
    print(
        f"Fanout: {args.fanout}, cache percent.: {cache_percent}, feature percent.:",
        torch.sum(ordered_counts[int(ordered_counts.shape[0] * cache_percent) :]).item()
        / g.num_nodes(),
    )
    cache_percent = 0.1
    print(
        f"Fanout: {args.fanout}, cache percent.: {cache_percent}, feature percent.:",
        torch.sum(ordered_counts[int(ordered_counts.shape[0] * cache_percent) :]).item()
        / g.num_nodes(),
    )
    cache_percent = 0.2
    print(
        f"Fanout: {args.fanout}, cache percent.: {cache_percent}, feature percent.:",
        torch.sum(ordered_counts[int(ordered_counts.shape[0] * cache_percent) :]).item()
        / g.num_nodes(),
    )
    cache_percent = 0.3
    print(
        f"Fanout: {args.fanout}, cache percent.: {cache_percent}, feature percent.:",
        torch.sum(ordered_counts[int(ordered_counts.shape[0] * cache_percent) :]).item()
        / g.num_nodes(),
    )
    cache_percent = 0.4
    print(
        f"Fanout: {args.fanout}, cache percent.: {cache_percent}, feature percent.:",
        torch.sum(ordered_counts[int(ordered_counts.shape[0] * cache_percent) :]).item()
        / g.num_nodes(),
    )

    # before_mem = process.memory_info().rss / (1024 * 1024 * 1024)
    # for i in range(len(blocks_pool)):
    #     blocks = blocks_pool[i]
    #     x = features[blocks[0].srcdata[dgl.NID]].pin_memory()
    #     y = labels[blocks[-1].dstdata[dgl.NID]].pin_memory()
    #     blocks_pool[i] = (blocks, x, y)
    # presample_mem = process.memory_info().rss / (1024 * 1024 * 1024) - before_mem
    # print("Memory of pre-sampled node features:", presample_mem, "GB")

    # presample_time = time.time() - tic
    # print("Pre-sampling time:", presample_time)
    # buffersize = len(blocks_pool)

    # static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    # print("memory allocated before training:", static_memory, "GB")

    # # To warm-up
    # model.train()
    # id = np.random.randint(0, buffersize)
    # blocks, x, y = blocks_pool[id]
    # blocks = [block.to("cuda") for block in blocks]
    # x = x.to("cuda", non_blocking=True)
    # y = y.to("cuda", non_blocking=True)
    # y_hat = model(blocks, x)

    # epoch_time = []
    # acc_list = []
    # torch.cuda.synchronize()
    # start = time.time()
    # model.train()
    # for i in tqdm(range(args.num_batches)):
    #     id = np.random.randint(0, buffersize)
    #     blocks, x, y = blocks_pool[id]
    #     blocks = [block.to("cuda") for block in blocks]
    #     x = x.to("cuda", non_blocking=True)
    #     y = y.to("cuda", non_blocking=True)
    #     y_hat = model(blocks, x)
    #     is_labeled = y == y
    #     y = y[is_labeled].long()
    #     y_hat = y_hat[is_labeled]
    #     loss = F.cross_entropy(y_hat, y)
    #     opt.zero_grad()
    #     loss.backward()
    #     opt.step()

    #     if i % 100 == 0:
    #         model.eval()
    #         val_pred = []
    #         val_labels = []
    #         with torch.no_grad():
    #             for it, (input_nodes, output_nodes, blocks) in enumerate(
    #                 tqdm(val_dataloader)
    #             ):
    #                 x = gather_pinned_tensor_rows(features, input_nodes)
    #                 y = gather_pinned_tensor_rows(labels, output_nodes)

    #                 y_pred = model(blocks, x)
    #                 val_pred.append(y_pred)
    #                 val_labels.append(y)
    #         pred = torch.cat(val_pred)
    #         label = torch.cat(val_labels)
    #         acc = (pred.argmax(1) == label).float().mean().item()
    #         acc_list.append(acc)
    #         model.train()

    #         print("Batch {:05d} | Val ACC {:.4f} s".format(i, acc))

    # torch.cuda.synchronize()
    # total_time = time.time() - start

    # torch.save(
    #     model.state_dict(),
    #     "models/graphsage_{}_{}_{}_{}_{}.pt".format(
    #         args.dataset,
    #         model.__class__.__name__,
    #         args.fanout.replace(",", "-"),
    #         args.num_sample,
    #         time.ctime().replace(" ", "_"),
    #     ),
    # )

    # print("Total Elapse Time:", total_time)
    # # print("Average Epoch Time:", np.mean(epoch_time[3:]))
    # s1 = pd.Series(acc_list, name="acc")
    # s2 = pd.Series(epoch_time, name="epoch train time/s")
    # s3 = pd.Series([total_time], name="total time/s")
    # s4 = pd.Series([static_memory], name="GPU graph mem/GB")
    # s5 = pd.Series([presample_time], name="presampling time/s")
    # s6 = pd.Series([presample_mem], name="presampling mem/GB")
    # df = pd.concat([s1, s2, s3, s4, s5, s6], axis=1)
    # df.to_csv(
    #     "data/graphsage_{}_{}_{}_{}_{}.csv".format(
    #         args.dataset,
    #         model.__class__.__name__,
    #         args.fanout.replace(",", "-"),
    #         args.num_sample,
    #         time.ctime().replace(" ", "_"),
    #     ),
    #     index=False,
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training model on gpu or cpu",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="#workers",
    )
    parser.add_argument("--dataset", default="ogbn-products", help="dataset")
    parser.add_argument("--batchsize", type=int, default=1024, help="batch size")
    parser.add_argument("--num-batches", type=int, default=2000, help="#batches")
    parser.add_argument("--num-sample", type=int, default=200, help="pre-sample size")
    parser.add_argument("--use-uva", type=bool, default=False, help="use uva")
    parser.add_argument(
        "--fanout", type=str, default="10,10,10", help="sampling fanout"
    )
    parser.add_argument("--model", type=str, default="SAGE", help="training model")
    parser.add_argument(
        "--path",
        type=str,
        default="/efs/rjliu/dataset/igb_large",
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
        "--num_classes",
        type=int,
        default=19,
        choices=[19, 2983],
        help="number of classes",
    )
    parser.add_argument(
        "--in_memory",
        type=int,
        default=1,
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
    print(args)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    if args.dataset == "reddit":
        dataset = load_reddit()
    elif args.dataset.startswith("ogbn"):
        dataset = load_ogb(args.dataset, "/home/ubuntu/dataset")
    elif args.dataset == "friendster":
        dataset = load_dglgraph("/home/ubuntu/dataset/friendster/friendster.bin")
    elif args.dataset.startswith("igb"):
        dataset = load_igb(args)
    else:
        raise NotImplementedError
    print(dataset[0])
    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Graph total memory:", mem1 - mem, "GB")
    train_dgl(dataset, args)
