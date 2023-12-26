import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata

        for i in s:
            unicodedata.numeric(i)
        return True
    except (TypeError, ValueError):
        pass
    return False


file_names = os.listdir("data/")

from collections import defaultdict

dfs = defaultdict(list)
for name in file_names:
    key = name.split("_")[:3]
    if len(key) > 1:
        if is_number(key[2]):
            dfs["_".join(key)].append(pd.read_csv("data/" + name))
        else:
            dfs["_".join(key[:2])].append(pd.read_csv("data/" + name))


def anlyze_data(dfs, epoch, prefix, title):
    plt.figure(figsize=(15, 8))
    x = range(epoch)
    keys = sorted(dfs.keys())
    for key in keys:
        if key.startswith(prefix):
            df_acc = pd.DataFrame()
            epochtrain_time = []
            presample_time = []
            e2e_time = []
            smallest_acc = 1
            largest_acc = 0
            for df in dfs[key]:
                df_acc[key] = df["acc"][:epoch]
                epochtrain_time.append(df["time/s"].mean())
                e2e_time.append(df["time/s"].sum())
                if key != prefix:
                    presample_time.append(df["presampling time/s"][0])
                smallest_acc = min(smallest_acc, df["acc"][epoch - 1])
                largest_acc = max(largest_acc, df["acc"][epoch - 1])
            mean_acc = df_acc.mean(axis=1)
            print(key)
            print("epoch train:", np.mean(epochtrain_time))
            if key != prefix:
                print("pre sample:", np.mean(presample_time))
            print("total train", np.mean(e2e_time))
            print("mean acc:", mean_acc[epoch - 1])
            print("worst acc:", smallest_acc)
            print("best acc:", largest_acc)
            print("below mean:", mean_acc[epoch - 1] - smallest_acc)
            print("above mean:", largest_acc - mean_acc[epoch - 1])
            if key == prefix:
                label = "Online Sampling"
            else:
                label = "Offline Sampling ({} Epoch)".format(key.split("_")[-1])
            plt.plot(x, mean_acc, label=label)
    plt.grid(alpha=0.2)
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.title(title)
    plt.legend()
    plt.show()


anlyze_data(dfs, 50, "graphsage_ogbn-arxiv", "GraphSAGE (ogbn-arxiv)")

anlyze_data(dfs, 50, "graphsage_products", "GraphSAGE (ogbn-products)")

anlyze_data(dfs, 50, "graphsage_reddit", "GraphSAGE (reddit)")

anlyze_data(dfs, 50, "ladies_products", "LADIES (ogbn-products)")

anlyze_data(dfs, 50, "ladies_reddit", "LADIES (reddit)")

df = pd.read_csv("data/ladies.csv")
df1 = pd.read_csv("data/ladies_1.csv")
df2 = pd.read_csv("data/ladies_2.csv")
df3 = pd.read_csv("data/ladies_3.csv")

epoch = 50
plt.figure(figsize=(15, 8))
x = range(epoch)
plt.plot(x, df["acc"][:epoch], label="Online Sampling", color="r")
plt.plot(x, df1["acc"][:epoch], label="Offline Sampling (1 Epoch)", color="g")
plt.plot(x, df2["acc"][:epoch], label="Offline Sampling (2 Epoch)", color="b")
plt.plot(x, df3["acc"][:epoch], label="Offline Sampling (3 Epoch)", color="black")
plt.grid(alpha=0.2)
plt.ylabel("Validation Accuracy")
plt.xlabel("Epoch")
plt.title("LADIES (ogbn-products)")
plt.legend()
plt.show()

print(df["time/s"][3:].mean())
print(df1["time/s"][3:].mean())
print(df2["time/s"][3:].mean())
print(df3["time/s"][3:].mean())

print(df1["presampling time/s"][0])
print(df2["presampling time/s"][0])
print(df3["presampling time/s"][0])

font_size = 20
marker_size = 10
plt.figure(figsize=(15, 8))
x = [0.1, 0.2, 0.3, 0.4]
fanout_10 = [0.935, 0.494, 0.238, 0.093]
fanout_15 = [1.441, 0.792, 0.399, 0.165]
fanout_30 = [2.555, 1.425, 0.734, 0.330]
fanout_50 = [3.340, 1.851, 0.951, 0.425]
plt.plot(x, fanout_10, label="fanout=10", color="r", marker="o", markersize=marker_size)
plt.plot(x, fanout_15, label="fanout=15", color="g", marker="v", markersize=marker_size)
plt.plot(x, fanout_30, label="fanout=30", color="b", marker="x", markersize=marker_size)
plt.plot(
    x, fanout_50, label="fanout=50", color="black", marker="1", markersize=marker_size
)
plt.grid(alpha=0.2)
plt.xticks(x, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylabel("packed node feature / #nodes", fontsize=font_size)
plt.xlabel("cache percentage", fontsize=font_size)
plt.title("Cache effect on node feature redundancy", fontsize=font_size)
plt.legend(fontsize=font_size)
plt.savefig("imgs/feature_redundancy.png", bbox_inches="tight")
plt.show()


font_size = 20
marker_size = 10
plt.figure(figsize=(15, 8))
x = [0.01, 0.05, 0.1, 0.2]
fanout_10_10 = [0.0079, 0, 0, 0]
fanout_10_10_10 = [0.0358, 0, 0, 0]
plt.plot(
    x, fanout_10_10, label="fanout=10,10", color="r", marker="o", markersize=marker_size
)
plt.plot(
    x,
    fanout_10_10_10,
    label="fanout=10,10,10",
    color="g",
    marker="v",
    markersize=marker_size,
)
plt.grid()
plt.xticks(x, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylabel("packed node feature / #nodes", fontsize=font_size)
plt.xlabel("cache percentage", fontsize=font_size)
plt.title(
    "Cache effect on node feature redundancy (Ogbn-papers100M)", fontsize=font_size
)
plt.legend(fontsize=font_size)
plt.savefig("imgs/feature_redundancy_subgraph.png", bbox_inches="tight")
plt.show()


font_size = 20
marker_size = 10
plt.figure(figsize=(15, 8))
x = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
fanout_10_10 = [0.169, 0.093, 0.0431, 0.0, 0.0, 0.0]
fanout_10_10_10 = [1.732, 1.186, 0.8033, 0.3769, 0.1525, 0.0521]
plt.plot(
    x, fanout_10_10, label="fanout=10,10", color="r", marker="o", markersize=marker_size
)
plt.plot(
    x,
    fanout_10_10_10,
    label="fanout=10,10,10",
    color="g",
    marker="v",
    markersize=marker_size,
)
plt.grid()
plt.xticks(x, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylabel("packed node feature / #nodes", fontsize=font_size)
plt.xlabel("cache percentage", fontsize=font_size)
plt.title("Cache effect on node feature redundancy (Friendster)", fontsize=font_size)
plt.legend(fontsize=font_size)
plt.savefig("imgs/feature_redundancy_subgraph_friendster.png", bbox_inches="tight")
plt.show()


font_size = 20
marker_size = 10
plt.figure(figsize=(15, 8))
x = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
fanout_10_10 = [0.0183, 0.0, 0.0, 0.0, 0.0, 0.0]
fanout_10_10_10 = [0.217, 0.108, 0.0424, 0.0, 0.0, 0.0]
plt.plot(
    x, fanout_10_10, label="fanout=10,10", color="r", marker="o", markersize=marker_size
)
plt.plot(
    x,
    fanout_10_10_10,
    label="fanout=10,10,10",
    color="g",
    marker="v",
    markersize=marker_size,
)
plt.grid()
plt.xticks(x, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylabel("packed node feature / #nodes", fontsize=font_size)
plt.xlabel("cache percentage", fontsize=font_size)
plt.title("Cache effect on node feature redundancy (IGB-full)", fontsize=font_size)
plt.legend(fontsize=font_size)
plt.savefig("graph/imgs/feature_redundancy_subgraph_igb_full.png", bbox_inches="tight")
plt.show()


font_size = 20
marker_size = 10
plt.figure(figsize=(15, 8))
x = [0.1, 0.2, 0.3, 0.4]
y = [8.326, 5.138, 3.117, 1.843]
plt.plot(
    x, y, label="gather all neighbors", color="r", marker="o", markersize=marker_size
)
plt.grid()
plt.xticks(x, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylabel("packed node feature / #nodes", fontsize=font_size)
plt.xlabel("cache percentage", fontsize=font_size)
plt.title("Cache effect on node feature redundancy (Papers100M)", fontsize=font_size)
plt.legend(fontsize=font_size)
plt.savefig(
    "graph/imgs/feature_redundancy_allsample_papers100M.png", bbox_inches="tight"
)
plt.show()


font_size = 20
marker_size = 10
plt.figure(figsize=(15, 8))
x = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
pre_sample_001 = [0.016, 0.0, 0.0, 0.0, 0.0, 0.0]
pre_sample_005 = [0.396, 0.102, 0.0171, 0.0, 0.0, 0.0]
pre_sample_01 = [0.988, 0.566, 0.3535, 0.1437, 0.0437, 0.0]
pre_sample_02 = [2.572, 1.838, 1.3353, 0.7324, 0.3872, 0.1804]
pre_sample_03 = [5.066, 3.855, 2.9213, 1.6906, 0.9266, 0.4528]
pre_sample_04 = [7.494, 5.748, 4.3571, 2.4959, 1.3434, 0.6497]
plt.plot(
    x,
    pre_sample_001,
    label="pre-sample 1% hot nodes",
    marker="o",
    markersize=marker_size,
)
plt.plot(
    x,
    pre_sample_005,
    label="pre-sample 5% hot nodes",
    marker="v",
    markersize=marker_size,
)
plt.plot(
    x,
    pre_sample_01,
    label="pre-sample 10% hot nodes",
    marker="1",
    markersize=marker_size,
)
plt.plot(
    x,
    pre_sample_02,
    label="pre-sample 20% hot nodes",
    marker="s",
    markersize=marker_size,
)
plt.plot(
    x,
    pre_sample_03,
    label="pre-sample 30% hot nodes",
    marker="p",
    markersize=marker_size,
)
plt.plot(
    x,
    pre_sample_04,
    label="pre-sample 40% hot nodes",
    marker="|",
    markersize=marker_size,
)
plt.grid()
plt.xticks(x, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylim(-0.1, 3)
plt.ylabel("packed node feature / #nodes", fontsize=font_size)
plt.xlabel("cache percentage", fontsize=font_size)
plt.title("Cache effect on node feature redundancy (Papers100M)", fontsize=font_size)
plt.legend(fontsize=font_size)
plt.savefig(
    "graph/imgs/feature_redundancy_hotnodes_allsample_papers100M.png",
    bbox_inches="tight",
)
plt.show()


font_size = 20
marker_size = 10
plt.figure(figsize=(15, 8))
x = [0.1, 0.2, 0.3, 0.4]
y = [21.290, 10.973, 5.8818, 3.2594]
plt.plot(
    x, y, label="gather all neighbors", color="r", marker="o", markersize=marker_size
)
plt.grid()
plt.xticks(x, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylabel("packed node feature / #nodes", fontsize=font_size)
plt.xlabel("cache percentage", fontsize=font_size)
plt.title("Cache effect on node feature redundancy (Friendster)", fontsize=font_size)
plt.legend(fontsize=font_size)
plt.savefig(
    "graph/imgs/feature_redundancy_allsample_friendster.png", bbox_inches="tight"
)
plt.show()


font_size = 20
marker_size = 10
x = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
pre_sample_001 = [6.148, 3.466, 2.0758, 0.8439, 0.3480, 0.1267]
pre_sample_005 = [15.03, 9.145, 5.8029, 2.5947, 1.1944, 0.5465]
pre_sample_01 = [23.148, 14.537, 9.42434, 4.34103, 2.05332, 0.97504]
pre_sample_02 = [32.948, 21.286, 14.0709, 6.66869, 3.23461, 1.57927]
pre_sample_03 = [38.251, 25.079, 16.7743, 8.10244, 3.98732, 1.97221]
pre_sample_04 = [41.189, 27.259, 18.3841, 9.01350, 4.49374, 2.24432]
plt.figure(figsize=(15, 8))
plt.plot(
    x,
    pre_sample_001,
    label="pre-sample 1% hot nodes",
    marker="o",
    markersize=marker_size,
)
plt.plot(
    x,
    pre_sample_005,
    label="pre-sample 5% hot nodes",
    marker="v",
    markersize=marker_size,
)
plt.plot(
    x,
    pre_sample_01,
    label="pre-sample 10% hot nodes",
    marker="1",
    markersize=marker_size,
)
plt.plot(
    x,
    pre_sample_02,
    label="pre-sample 20% hot nodes",
    marker="s",
    markersize=marker_size,
)
plt.plot(
    x,
    pre_sample_03,
    label="pre-sample 30% hot nodes",
    marker="p",
    markersize=marker_size,
)
plt.plot(
    x,
    pre_sample_04,
    label="pre-sample 40% hot nodes",
    marker="|",
    markersize=marker_size,
)
plt.grid()
plt.xticks(x, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylim(-0.1, 3)
plt.ylabel("packed node feature / #nodes", fontsize=font_size)
plt.xlabel("cache percentage", fontsize=font_size)
plt.title("Cache effect on node feature redundancy (Friendster)", fontsize=font_size)
plt.legend(fontsize=font_size)
plt.savefig(
    "graph/imgs/feature_redundancy_hotnodes_allsample_friendster.png",
    bbox_inches="tight",
)
plt.show()
