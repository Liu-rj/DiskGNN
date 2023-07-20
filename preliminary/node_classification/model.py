import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, SAGEConv
import dgl.function as fn
from tqdm import tqdm
from dgl.nn.pytorch import GATConv


def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = "W"
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, "v"))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, "u"))
        g.ndata["u"] = g_rev.ndata["u"]
        g.apply_edges(
            lambda edges: {
                "w": edges.data[weight] / torch.sqrt(edges.src["u"] * edges.dst["v"])
            }
        )
        return g.edata["w"]


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
            x = conv(block, x, edge_weight=block.edata["w"])
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x


class SAGEModel(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hid_size, "mean"))
        for i in range(num_layers - 2):
            self.layers.append(SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        num_layers = len(heads)
        # two-layer GAT
        self.gat_layers.append(
            GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.2,
                attn_drop=0.2,
                activation=F.elu,
            )
        )
        for i in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    hid_size * heads[i],
                    hid_size,
                    heads[i + 1],
                    feat_drop=0.2,
                    attn_drop=0.2,
                    activation=F.elu,
                )
            )
        self.gat_layers.append(
            GATConv(
                hid_size * heads[-2],
                out_size,
                heads[-1],
                feat_drop=0.2,
                attn_drop=0.2,
                activation=None,
            )
        )

    def forward(self, blocks, inputs):
        h = inputs
        for i, (layer, block) in enumerate(zip(self.gat_layers, blocks)):
            h = layer(block, h)
            if i == len(self.gat_layers) - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h
