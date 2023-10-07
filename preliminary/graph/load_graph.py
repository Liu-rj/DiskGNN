import torch
import dgl
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from igb.dataloader import IGB260MDGLDataset
from ogb.lsc import MAG240MDataset
import os


def load_reddit():
    data = RedditDataset(self_loop=True)
    g = data[0].long()
    n_classes = data.num_classes
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    feat = g.ndata["feat"]
    labels = g.ndata["label"]
    g.ndata.clear()
    return g, feat, labels, n_classes, splitted_idx


def load_ogb(name, root):
    data = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    g = g.long()
    feat = g.ndata["feat"]
    labels = labels[:, 0]
    n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g.ndata.clear()
    return g, feat, labels, n_classes, splitted_idx


def load_igb(args):
    data = IGB260MDGLDataset(args)
    g = data[0].long()
    n_classes = args.num_classes
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    feat = g.ndata["feat"]
    labels = g.ndata["label"]
    g.ndata.clear()
    return g, feat, labels, n_classes, splitted_idx


def load_dglgraph(root: str):
    data, _ = dgl.load_graphs(root)
    g = data[0].long()
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    g.ndata.clear()
    return g, None, None, None, splitted_idx


def load_mag240m(dir: str):
    __meta__ = torch.load(os.path.join(dir, "meta.pt"))
    __split__ = torch.load(os.path.join(dir, "split_dict.pt"))
    (g,), _ = dgl.load_graphs(os.path.join(dir, "graph.dgl"))
    g = g.long()
    paper_offset = __meta__["author"] + __meta__["institution"]
    num_nodes = paper_offset + __meta__["paper"]
    num_features = 768
    train_idx = torch.LongTensor(__split__["train"]) + paper_offset
    valid_idx = torch.LongTensor(__split__["valid"]) + paper_offset
    test_idx = torch.LongTensor(__split__["test"]) + paper_offset
    splitted_idx = {"train": train_idx, "test": test_idx, "valid": valid_idx}
    g.ndata.clear()
    return g, None, None, None, splitted_idx
