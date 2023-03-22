import torch
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
from ctypes import *
from ctypes.util import *
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from load_graph import *
from model import *


def train_dgl(dataset, args):
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats('csc')
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    features, labels = features.pin_memory(), labels.pin_memory()

    g = g.to('cuda')
    train_nid, val_nid = train_nid.to('cuda'), val_nid.to('cuda')
    num_layers = 3
    model = SAGEModel(features.shape[1], 64, n_classes, num_layers).to('cuda')
    sampler = NeighborSampler([5, 10, 15])
    train_dataloader = DataLoader(g, train_nid, sampler, batch_size=args.batchsize,
                                  shuffle=True,  drop_last=False, num_workers=args.num_workers)
    val_dataloader = DataLoader(g, val_nid, sampler, batch_size=args.batchsize,
                                shuffle=True, drop_last=False, num_workers=args.num_workers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    print('memory allocated before training:', static_memory, 'GB')

    # To warm-up
    model.train()
    for step, (input_nodes, output_nodes, blocks) in enumerate(tqdm(train_dataloader)):
        x = gather_pinned_tensor_rows(features, input_nodes)
        y = gather_pinned_tensor_rows(labels, output_nodes)
        y_hat = model(blocks, x)

    epoch_time = []
    acc_list = []
    start = time.time()
    for epoch in range(args.num_epoch):
        torch.cuda.synchronize()
        tic = time.time()
        model.train()
        for step, (input_nodes, output_nodes, blocks) in enumerate(tqdm(train_dataloader)):
            x = gather_pinned_tensor_rows(features, input_nodes)
            y = gather_pinned_tensor_rows(labels, output_nodes)
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

    print('Total Elapse Time:', total_time)
    print('Average Epoch Time:', np.mean(epoch_time[3:]))
    s1 = pd.Series(acc_list, name='acc')
    s2 = pd.Series(epoch_time, name='time/s')
    s3 = pd.Series([total_time], name='total time/s')
    s4 = pd.Series([static_memory], name='static mem/GB')
    df = pd.concat([s1, s2, s3, s4], axis=1)
    df.to_csv('data/graphsage_{}_{}.csv'.format(args.dataset,
              time.ctime().replace(' ', '_')), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument("--dataset", default='reddit', choices=['reddit', 'products', 'papers100m'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=512,
                        help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    parser.add_argument("--num-epoch", type=int, default=50,
                        help="numbers of epoch in training")
    args = parser.parse_args()
    print(args)
    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogb('ogbn-products')
    elif args.dataset == 'papers100m':
        dataset = load_ogb('ogbn-papers100M')
    print(dataset[0])
    train_dgl(dataset, args)
