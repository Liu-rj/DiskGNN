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


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden))
        for i in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden))
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, h):
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hid_size, "mean"))
        for i in range(num_layers - 2):
            self.layers.append(SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class SAGE_Shadow(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hid_size, "mean"))
        for i in range(num_layers - 2):
            self.layers.append(SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(dropout)

    def forward(self, subg, h):
        for l, layer in enumerate(self.layers):
            h = layer(subg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads, num_layers, dropout):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            GATConv(
                in_size,
                hid_size,
                heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=F.relu,
            )
        )
        for i in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    hid_size * heads,
                    hid_size,
                    heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=F.elu,
                )
            )
        self.gat_layers.append(
            GATConv(
                hid_size * heads,
                out_size,
                heads,
                feat_drop=dropout,
                attn_drop=dropout,
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
