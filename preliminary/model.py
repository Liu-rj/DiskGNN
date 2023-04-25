import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, SAGEConv
import dgl.function as fn
from tqdm import tqdm


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


class GCNModel(nn.Module):
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


class SAGEModel(nn.Module):
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

    # def inference(self, g, x, device):
    #     """
    #     Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
    #     g : the entire graph.
    #     x : the input of entire node set.
    #     The inference code is written in a fashion that it could handle any number of nodes and
    #     layers.
    #     """
    #     # During inference with sampling, multi-layer blocks are very inefficient because
    #     # lots of computations in the first few layers are repeated.
    #     # Therefore, we compute the representation of all nodes layer by layer.  The nodes
    #     # on each layer are of course splitted in batches.
    #     # TODO: can we standardize this?
    #     for l, layer in enumerate(self.layers):
    #         y = torch.zeros(
    #             g.num_nodes(),
    #             self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
    #         ).to(device)

    #         sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    #         dataloader = dgl.dataloading.DataLoader(
    #             g,
    #             torch.arange(g.num_nodes()),
    #             sampler,
    #             batch_size=args.batch_size,
    #             shuffle=True,
    #             drop_last=False,
    #             num_workers=args.num_workers,
    #         )

    #         for input_nodes, output_nodes, blocks in tqdm(dataloader):
    #             block = blocks[0].int().to(device)

    #             h = x[input_nodes]
    #             h = layer(block, h)
    #             if l != len(self.layers) - 1:
    #                 h = F.relu(h)
    #                 h = self.dropout(h)

    #             y[output_nodes] = h

    #         x = y
    #     return y
