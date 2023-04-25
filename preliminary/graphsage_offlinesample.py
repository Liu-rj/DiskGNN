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
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats('csc')
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    features, labels = features.pin_memory(), labels.pin_memory()

    g = g.to('cpu') if args.use_uva else g.to(device)
    train_nid, val_nid = train_nid.to(device), val_nid.to(device)
    num_layers = 3
    model = SAGEModel(features.shape[1], 256, n_classes, num_layers).to(device)
    sampler = NeighborSampler([10, 10, 10])
    train_dataloader = DataLoader(g, train_nid, sampler, batch_size=args.batchsize,
                                  shuffle=True,  drop_last=False, num_workers=args.num_workers,
                                  device=torch.device(device), use_uva=args.use_uva)
    val_dataloader = DataLoader(g, val_nid, sampler, batch_size=args.batchsize,
                                shuffle=True, drop_last=False, num_workers=args.num_workers,
                                device=torch.device(device), use_uva=args.use_uva)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    torch.cuda.synchronize()
    process = psutil.Process(os.getpid())
    before_mem = process.memory_info().rss / (1024 * 1024 * 1024)
    print('Start Pre-sample')
    tic = time.time()
    blocks_pool = []
    if args.num_sample < 1:
        for i in tqdm(range(int(args.num_sample * len(train_dataloader)))):
            input_nodes, output_nodes, blocks = next(iter(train_dataloader))
            blocks = [block.to('cpu') for block in blocks]
            x = features[blocks[0].srcdata[dgl.NID]].pin_memory()
            y = labels[blocks[-1].dstdata[dgl.NID]].pin_memory()
            blocks_pool.append((blocks, x, y))
    else:
        for i in range(int(args.num_sample)):
            for step, (input_nodes, output_nodes, blocks) in enumerate(tqdm(train_dataloader)):
                blocks = [block.to('cpu') for block in blocks]
                x = features[blocks[0].srcdata[dgl.NID]].pin_memory()
                y = labels[blocks[-1].dstdata[dgl.NID]].pin_memory()
                blocks_pool.append((blocks, x, y))
    torch.cuda.synchronize()
    presample_time = time.time() - tic
    print('Pre-sampling time:', presample_time)
    presample_mem = process.memory_info().rss / (1024 * 1024 * 1024) - before_mem
    print('Memory of pre-sampled sub-graphs:', presample_mem, 'GB')
    buffersize = len(blocks_pool)

    static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    print('memory allocated before training:', static_memory, 'GB')

    # To warm-up
    model.train()
    for batch in tqdm(range(len(train_dataloader))):
        id = np.random.randint(0, buffersize)
        blocks, x, y = blocks_pool[id]
        blocks = [block.to('cuda') for block in blocks]
        x = x.to('cuda', non_blocking=True)
        y = y.to('cuda', non_blocking=True)
        y_hat = model(blocks, x)

    epoch_time = []
    acc_list = []
    start = time.time()
    for epoch in range(args.num_epoch):
        torch.cuda.synchronize()
        tic = time.time()
        model.train()
        for batch in tqdm(range(len(train_dataloader))):
            id = np.random.randint(0, buffersize)
            blocks, x, y = blocks_pool[id]
            blocks = [block.to('cuda') for block in blocks]
            x = x.to('cuda', non_blocking=True)
            y = y.to('cuda', non_blocking=True)
            y_hat = model(blocks, x)
            is_labeled = y == y
            y = y[is_labeled].long()
            y_hat = y_hat[is_labeled]
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        epoch_time.append(time.time() - tic)

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            for it, (input_nodes, output_nodes, blocks) in enumerate(tqdm(val_dataloader)):
                x = gather_pinned_tensor_rows(features, input_nodes)
                y = gather_pinned_tensor_rows(labels, output_nodes)

                y_pred = model(blocks, x)
                val_pred.append(y_pred)
                val_labels.append(y)
        pred = torch.cat(val_pred)
        label = torch.cat(val_labels)
        acc = (pred.argmax(1) == label).float().mean().item()
        acc_list.append(acc)

        print("Epoch {:05d} | Val ACC {:.4f} | Epoch Time {:.4f} s".format(
            epoch, acc, epoch_time[-1]))

    torch.cuda.synchronize()
    total_time = time.time() - start

    torch.save(model.state_dict(), 'models/graphsage_{}_{}_{}.pt'.format(
        args.dataset, args.num_sample, time.ctime().replace(' ', '_')))

    print('Total Elapse Time:', total_time)
    print('Average Epoch Time:', np.mean(epoch_time[3:]))
    s1 = pd.Series(acc_list, name='acc')
    s2 = pd.Series(epoch_time, name='time/s')
    s3 = pd.Series([total_time], name='total time/s')
    s4 = pd.Series([static_memory], name='static mem/GB')
    s5 = pd.Series([presample_time], name='presampling time/s')
    s6 = pd.Series([presample_mem], name='presampling mem/GB')
    df = pd.concat([s1, s2, s3, s4, s5, s6], axis=1)
    df.to_csv('data/graphsage_{}_{}_{}.csv'.format(
              args.dataset, args.num_sample, time.ctime().replace(' ', '_')), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument("--dataset", default='reddit',
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=512,
                        help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    parser.add_argument("--num-epoch", type=int, default=50,
                        help="numbers of epoch in training")
    parser.add_argument("--num-sample", type=float, default=1,
                        help="numbers of epoch in training")
    parser.add_argument("--use-uva", type=bool, default=False,
                        help="use uva for sampling or not")
    args = parser.parse_args()
    if args.dataset == 'reddit':
        dataset = load_reddit()
    else:
        dataset = load_ogb(args.dataset)
    print(dataset[0])
    train_dgl(dataset, args)
