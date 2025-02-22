import torch
import dgl
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
import numpy as np
import torch.nn.functional as F
from offgs.utils import SAGE, GAT, GCN
from offgs.dataset import OffgsDataset
import psutil
import csv
import os
from tqdm import tqdm


def evaluate(
    model, val_dataloader, test_dataloader, features, labels, label_offset, device
):
    # --- valid ---#
    model.eval()
    valid_correct, valid_tot, val_acc = 0, 0, 0
    for i, (input_nodes, output_nodes, blocks) in enumerate(
        tqdm(val_dataloader, ncols=100)
    ):
        blocks = [block.to(device) for block in blocks]
        x = gather_pinned_tensor_rows(features, input_nodes.to(device))
        # x = features[input_nodes.cpu()].to(device)
        y = gather_pinned_tensor_rows(labels, output_nodes - label_offset).long()
        pred = model(blocks, x)
        correct = (pred.argmax(dim=1) == y).sum().item()
        total = y.shape[0]
        valid_correct += correct
        valid_tot += total
    val_acc = valid_correct / valid_tot

    # --- test --- #
    model.eval()
    test_correct, test_tot, test_acc = 0, 0, 0
    if args.dataset != "mag240m":
        for i, (input_nodes, output_nodes, blocks) in enumerate(
            tqdm(test_dataloader, ncols=100)
        ):
            blocks = [block.to(device) for block in blocks]
            x = gather_pinned_tensor_rows(features, input_nodes.to(device))
            # x = features[input_nodes.cpu()].to(device)
            y = gather_pinned_tensor_rows(labels, output_nodes - label_offset).long()
            pred = model(blocks, x)
            correct = (pred.argmax(dim=1) == y).sum().item()
            total = y.shape[0]
            test_correct += correct
            test_tot += total
        test_acc = test_correct / test_tot
    return val_acc, test_acc


def train(
    args,
    dataset: OffgsDataset,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
):
    device = torch.device(f"cuda:{args.device}")
    fanout = [int(x) for x in args.fanout.split(",")]
    acc_logfile = f"logs/online_{args.dataset}_{args.fanout}_{args.model}_{args.hidden}_{args.dropout}_{args.ratio}.csv"

    labels = dataset.labels.pin_memory()
    print("features shape:", features.shape)
    # features = features.pin_memory()
    label_offset = dataset.conf["label_offset"]
    print("label offset:", label_offset)

    sampler = NeighborSampler(fanout)
    if args.model == "SAGE":
        model = SAGE(
            dataset.num_features,
            args.hidden,
            dataset.num_classes,
            len(fanout),
            args.dropout,
        ).to(device)
    elif args.model == "GAT":
        model = GAT(
            dataset.num_features,
            args.hidden,
            dataset.num_classes,
            4,
            len(fanout),
            args.dropout,
        ).to(device)
    elif args.model == "GCN":
        model = GCN(
            dataset.num_features,
            args.hidden,
            dataset.num_classes,
            len(fanout),
            args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_nid = (
        dataset.split_idx["train"]
        if args.ratio == 1.0
        else torch.load(f"{args.dir}/{args.dataset}-offgs/train_idx_{args.ratio}.pt")
    )

    train_dataloader = DataLoader(
        graph,
        train_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        device=torch.device(device),
        use_uva=True,
    )

    val_dataloader = DataLoader(
        graph,
        dataset.split_idx["valid"],
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        device=torch.device(device),
        use_uva=True,
    )

    test_dataloader = None
    if args.dataset != "mag240m":
        test_dataloader = DataLoader(
            graph,
            dataset.split_idx["test"],
            sampler,
            batch_size=args.batchsize,
            shuffle=True,
            drop_last=False,
            device=torch.device(device),
            use_uva=True,
        )

    best_val_acc, best_test_acc, best_epoch = 0, 0, 0
    val_acc, test_acc = evaluate(
        model,
        val_dataloader,
        test_dataloader,
        features,
        labels,
        label_offset,
        device,
    )

    print(
        f"Epoch: 0\t"
        f"Valid acc: {val_acc * 100:.2f}%\t"
        f"Test acc: {test_acc * 100:.2f}%\t"
    )

    with open(acc_logfile, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            0,
            round(-1, 3),
            round(-1, 2),
            round(val_acc * 100, 2),
            round(test_acc * 100, 2),
            round(best_val_acc * 100, 2),
            round(best_test_acc * 100, 2),
            best_epoch,
        ]
        writer.writerow(log_info)

    epoch_info_recorder = [[] for i in range(10)]
    for epoch in range(args.num_epoch):
        train_correct, train_tot, train_acc = 0, 0, 0
        tot_loss = 0
        info_recorder = [0] * 9
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            tqdm(train_dataloader, ncols=100)
        ):
            tic = time.time()
            x = gather_pinned_tensor_rows(features, input_nodes.to(device))
            # x = features[input_nodes].to(device)
            torch.cuda.synchronize()
            info_recorder[2] += time.time() - tic  # feat assemble
            info_recorder[8] += x.shape[0]  # input node num

            tic = time.time()
            blocks = [block.to(device) for block in blocks]
            y = gather_pinned_tensor_rows(labels, output_nodes - label_offset).long()
            torch.cuda.synchronize()
            info_recorder[3] += time.time() - tic  # graph & label transfer

            tic = time.time()
            pred = model(blocks, x)
            loss = F.cross_entropy(pred, y)
            tot_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            info_recorder[5] += time.time() - tic

            correct = (pred.argmax(dim=1) == y).sum().item()
            total = y.shape[0]
            train_correct += correct
            train_tot += total

            if epoch == 0 and it in [
                len(train_dataloader) // 500,
                len(train_dataloader) // 100,
                len(train_dataloader) // 10,
                len(train_dataloader) // 5,
                len(train_dataloader) // 2,
            ]:
                train_acc = train_correct / train_tot
                val_acc, test_acc = evaluate(
                    model,
                    val_dataloader,
                    test_dataloader,
                    features,
                    labels,
                    label_offset,
                    device,
                )
                model.train()

                with open(acc_logfile, "a") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    log_info = [
                        round((it + 1) / len(train_dataloader), 3),
                        round(tot_loss, 3),
                        round(train_acc * 100, 2),
                        round(val_acc * 100, 2),
                        round(test_acc * 100, 2),
                        round(best_val_acc * 100, 2),
                        round(best_test_acc * 100, 2),
                        best_epoch,
                    ]
                    writer.writerow(log_info)

        train_acc = train_correct / train_tot
        val_acc, test_acc = evaluate(
            model,
            val_dataloader,
            test_dataloader,
            features,
            labels,
            label_offset,
            device,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch

        with open(acc_logfile, "a") as f:
            writer = csv.writer(f, lineterminator="\n")
            log_info = [
                epoch,
                round(tot_loss, 3),
                round(train_acc * 100, 2),
                round(val_acc * 100, 2),
                round(test_acc * 100, 2),
                round(best_val_acc * 100, 2),
                round(best_test_acc * 100, 2),
                best_epoch,
            ]
            writer.writerow(log_info)

        print(
            f"Epoch: {epoch}\t"
            f"Graph Load Time: {info_recorder[0]:.3f}\t"
            f"Feature Load Time: {info_recorder[1]:.3f}\t"
            f"Feat Assemble Time: {info_recorder[2]:.3f}\t"
            f"Sample and Graph Transfer Time : {info_recorder[3]:.3f}\t"
            f"Feat Transfer Time: {info_recorder[4]:.3f}\t"
            f"Train Time: {info_recorder[5]:.3f}\t"
            f"Sample init time: {info_recorder[6]:.3f}\t"
            f"Epoch Time: {np.sum(info_recorder[:7]):.3f}\t"
            f"Cold Feats Num: {info_recorder[7]}\t"
            f"Block Input Node Num: {info_recorder[8]}\t"
            f"Loss: {tot_loss:.10f}\t"
            f"Valid acc: {val_acc * 100:.2f}%\t"
            f"Test acc: {test_acc * 100:.2f}%\t"
        )
        for i, info in enumerate(info_recorder):
            epoch_info_recorder[i].append(info)
        epoch_info_recorder[-1].append(np.sum(info_recorder[:7]))

    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.fanout,
            args.batchsize,
            args.model,
            args.hidden,
            args.dropout,
            args.num_epoch,
            best_epoch,
            round(best_val_acc * 100, 2),
            round(best_test_acc * 100, 2),
        ]
        for epoch_info in epoch_info_recorder:
            log_info.append(round(np.mean(epoch_info[1:]), 2))
        writer.writerow(log_info)


def start(args):
    dataset_dir = f"{args.dir}/{args.dataset}-offgs"

    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    dataset = OffgsDataset(dataset_dir)
    graph = dataset.graph
    features = dataset.features.pin_memory()
    print(graph)

    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Memory Occupation:", mem1 - mem, "GB")

    train(args, dataset, graph, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--dataset", default="ogbn-products")
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--model", type=str, default="SAGE")
    parser.add_argument("--dir", type=str, default="/nvme1n1/offgs_dataset")
    parser.add_argument("--num-epoch", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--log", type=str, default="../logs/train_online_acc.csv")
    args = parser.parse_args()
    print(args)

    start(args)
