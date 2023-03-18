import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import time
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.nn.pytorch import GraphConv
import dgl.function as fn
import pandas as pd


class LADIESSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, weight='w', out_weight='w', replace=False, W=None):
        super().__init__()
        self.fanouts = fanouts
        self.edge_weight = weight
        self.output_weight = out_weight
        self.replace = replace
        self.return_eids = False
        self.W = W

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        blocks = []
        output_nodes = seed_nodes
        for fanout in self.fanouts:
            subg = dgl.in_subgraph(g, seed_nodes)
            # layer-wise sample
            edges = subg.edges()
            nodes = torch.unique(edges[0])
            num_pick = np.min([nodes.shape[0], fanout])
            reversed_subg = dgl.reverse(subg, copy_edata=True)
            weight = self.W[reversed_subg.edata[dgl.EID]]
            probs = dgl.ops.copy_e_sum(reversed_subg, weight ** 2)
            node_probs = probs[nodes]
            idx = torch.multinomial(node_probs, num_pick, replacement=False)
            selected = nodes[idx]
            ################
            selected = torch.cat((seed_nodes, selected)).unique()
            subg = dgl.out_subgraph(subg, selected)
            weight = weight[subg.edata[dgl.EID]]
            W_tilde = dgl.ops.e_div_u(subg, weight, probs)
            W_tilde_sum = dgl.ops.copy_e_sum(subg, W_tilde)
            W_tilde = dgl.ops.e_div_v(subg, W_tilde, W_tilde_sum)
            block = dgl.to_block(subg, seed_nodes)
            block.edata[self.output_weight] = W_tilde[block.edata[dgl.EID]]
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks


def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'u'))
        g.ndata['u'] = g_rev.ndata['u']
        g.apply_edges(lambda edges: {
                      'w': edges.data[weight] / torch.sqrt(edges.src['u'] * edges.dst['v'])})
        return g.edata['w']


class Model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_feats, n_hidden))
        for i in range(n_layers - 2):
            self.convs.append(GraphConv(n_hidden, n_hidden))
        self.convs.append(GraphConv(n_hidden, n_classes))

    def forward(self, blocks, x):
        for i, (conv, block) in enumerate(zip(self.convs, blocks)):
            x = conv(block, x, edge_weight=block.edata['w'])
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x


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


def train(dataset, args):
    device = args.device
    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    g = g.to(device)
    train_nid, val_nid = train_nid.to(device), val_nid.to(device)
    W = normalized_laplacian_edata(g)
    g = g.formats('csc')
    features, labels = features.to(device), labels.to(device)

    fanout = [1000, 1000, 1000]
    model = Model(features.shape[1], 64, n_classes, 3).to('cuda')
    sampler = LADIESSampler(fanout, weight='weight',
                            out_weight='w', replace=False, W=W)
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
    df.to_csv('data/ladies_{}.csv'.format(args.num_sample), index=False)


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
    train(dataset, args)
