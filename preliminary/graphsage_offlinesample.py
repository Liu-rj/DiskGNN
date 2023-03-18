import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import DataLoader, NeighborSampler
import time
import argparse
from ctypes import *
from ctypes.util import *
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from dgl.nn.pytorch import SAGEConv
import torch.nn as nn
import pandas as pd


class GraphSAGE_DGL(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hid_size, 'mean'))
        for i in range(num_layers - 2):
            self.layers.append(SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def load_ogb(name):
    data = DglNodePropPredDataset(name=name, root="/home/ubuntu/dataset")
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    g = g.long()
    feat = g.ndata['feat']
    labels = labels[:, 0]
    n_classes = len(torch.unique(
        labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx


def train_dgl(dataset, args):
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats('csc')
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']

    g, train_nid, val_nid = g.to('cuda'), train_nid.to(
        'cuda'), val_nid.to('cuda')
    features = features.to('cuda')
    labels = labels.to('cuda')
    num_layers = 3
    model = GraphSAGE_DGL(
        features.shape[1], 64, n_classes, num_layers).to('cuda')
    sampler = NeighborSampler([5, 10, 15])
    train_dataloader = DataLoader(g, train_nid, sampler, batch_size=args.batchsize,
                                  shuffle=True,  drop_last=False, num_workers=args.num_workers)
    val_dataloader = DataLoader(g, val_nid, sampler, batch_size=args.batchsize, shuffle=True,
                                drop_last=False, num_workers=args.num_workers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    torch.cuda.synchronize()
    tic = time.time()
    blocks_pool = []
    for i in range(args.num_sample):
        for step, (input_nodes, output_nodes, blocks) in enumerate(tqdm(train_dataloader)):
            blocks = [block.to('cpu') for block in blocks]
            blocks_pool.append(blocks)
    torch.cuda.synchronize()
    presample_time = time.time() - tic
    buffersize = len(blocks_pool)

    static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    print('memory allocated before training:', static_memory, 'GB')
    epoch_time = []
    acc_list = []
    for epoch in range(args.num_epoch):
        torch.cuda.synchronize()
        tic = time.time()
        model.train()
        for batch in tqdm(range(len(train_dataloader))):
            id = np.random.randint(0, buffersize)
            blocks = blocks_pool[id]
            blocks = [block.to('cuda') for block in blocks]
            x = features[blocks[0].srcdata[dgl.NID]]
            y = labels[blocks[-1].dstdata[dgl.NID]]
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
                x = features[input_nodes]
                y = labels[output_nodes]

                y_pred = model(blocks, x)
                val_pred.append(y_pred)
                val_labels.append(y)
        pred = torch.cat(val_pred)
        label = torch.cat(val_labels)
        acc = (pred.argmax(1) == label).float().mean().item()
        acc_list.append(acc)

        print("Epoch {:05d} | Val ACC {:.4f} | Epoch Time {:.4f} s".format(
            epoch, acc, epoch_time[-1]))

    print('Average Epoch Time:', np.mean(epoch_time[3:]))
    s1 = pd.Series(acc_list, name='acc')
    s2 = pd.Series(epoch_time, name='time/s')
    s3 = pd.Series([static_memory], name='static mem/GB')
    s4 = pd.Series([presample_time], name='presampling time/s')
    df = pd.concat([s1, s2, s3, s4], axis=1)
    df.to_csv('data/graphsage_{}.csv'.format(args.num_sample), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument("--dataset", default='products', choices=['products', 'papers100m'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=512,
                        help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    parser.add_argument("--num-epoch", type=int, default=50,
                        help="numbers of epoch in training")
    parser.add_argument("--num-sample", type=int, default=1,
                        help="numbers of epoch in training")
    args = parser.parse_args()
    print(args)
    if args.dataset == 'products':
        dataset = load_ogb('ogbn-products')
    elif args.dataset == 'papers100m':
        dataset = load_ogb('ogbn-papers100M')
    print(dataset[0])
    train_dgl(dataset, args)
