import matplotlib.pyplot as plt

avg_degree = {
    "ogbn-products": 51.517,
    "ogbn-papers100M": 14.548,
    "friendster": 27.528,
    "igb_tiny": 5.47,
    "igb_large": 12.20,
    "igb_full": 14.90,
}
cache_ratio = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
presample_ratio = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]


def plot_speedup(speedup_all: list, title: str):
    font_size = 20
    marker_size = 10
    markers = ["o", "v", "1", "s", "p", "|", "_"]
    plt.figure(figsize=(15, 8))
    for i, speedup in enumerate(speedup_all):
        plt.plot(cache_ratio, speedup, label=f"{presample_ratio[i] * 100}%", marker=markers[i], markersize=marker_size)
    plt.xticks([0, 0.05, 0.1, 0.2, 0.3, 0.4], fontsize=font_size)
    plt.xticks([0.01], minor=True)
    plt.yticks(fontsize=font_size)
    plt.grid(which="major", linestyle="-.")
    plt.grid(which="minor", linestyle="-.")
    plt.ylabel("speedup over online sampling", fontsize=font_size)
    plt.xlabel("cache ratio", fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.legend(fontsize=15, ncols=4)
    file_name = title.replace(":", "").replace(",", "").replace(" ", "_")
    plt.savefig(f"imgs/{file_name}.png", bbox_inches="tight")
    plt.show()


def plot_nfeat_access(online, offline, title):
    font_size = 20
    marker_size = 10
    markers = ["o", "v", "1", "s", "p", "|", "_"]
    plt.figure(figsize=(15, 8))
    plt.plot(cache_ratio, online, label="online", marker="*", markersize=marker_size)
    for i, line in enumerate(offline):
        plt.plot(cache_ratio, line, label=f"{presample_ratio[i] * 100}%", marker=markers[i], markersize=marker_size)
    plt.xticks([0, 0.05, 0.1, 0.2, 0.3, 0.4], fontsize=font_size)
    plt.xticks([0.01], minor=True)
    plt.yticks(fontsize=font_size)
    plt.grid(which="major", linestyle="-.")
    plt.grid(which="minor", linestyle="-.")
    plt.ylabel("#Nfeat accesses", fontsize=font_size)
    plt.xlabel("cache ratio", fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.legend(fontsize=15, ncols=4)
    file_name = title.replace(":", "").replace(",", "").replace(" ", "_")
    plt.savefig(f"imgs/{file_name}.png", bbox_inches="tight")
    plt.show()


with open("../graph/logs/log_07_25.txt", "r") as f:
    namespace = f.readline()
    while namespace:
        dataset = namespace.split("'")[1]
        for i in range(5):
            f.readline()

        # online
        online_adj_access = int(f.readline().split(": ")[-1])
        online_nfeat_access = []
        for i in range(len(cache_ratio)):
            online_nfeat_access.append(int(f.readline().split(": ")[-1]))

        # node-wise offline
        presample_hit, presample_miss = [], []
        cold_nfeat_access, nfeat_blowup = [], []
        for i in range(len(presample_ratio)):
            content = f.readline().split(", ")
            presample_hit.append(int(content[1].split(": ")[-1]))
            presample_miss.append(int(content[2].split(": ")[-1]))
            cold_access, blowup = [], []
            for j in range(len(cache_ratio)):
                content = f.readline().split(", ")
                cold_access.append(int(content[2].split(": ")[-1]))
                blowup.append(float(content[3].split(": ")[-1]))
            cold_nfeat_access.append(cold_access)
            nfeat_blowup.append(blowup)

        plot_nfeat_access(online_nfeat_access, cold_nfeat_access, f"{dataset}: nfeat access")

        # Graph in memory
        speedup_all = []
        for i in range(len(presample_ratio)):
            speedup = []
            for j in range(len(cache_ratio)):
                cost_online = online_nfeat_access[j]
                cost_offline = presample_hit[i] * 8 + cold_nfeat_access[i][j]
                speedup.append(cost_online / cost_offline)
            speedup_all.append(speedup)
        plot_speedup(speedup_all, f"{dataset}: graph in memory")

        # Graph in memory, overlap presample
        speedup_all = []
        for i in range(len(presample_ratio)):
            speedup = []
            for j in range(len(cache_ratio)):
                cost_online = online_nfeat_access[j]
                cost_offline = cold_nfeat_access[i][j]
                speedup.append(cost_online / cost_offline)
            speedup_all.append(speedup)
        plot_speedup(speedup_all, f"{dataset}: graph in memory, overlap persample")

        # Graph in disk
        speedup_all = []
        for i in range(len(presample_ratio)):
            speedup = []
            for j in range(len(cache_ratio)):
                cost_online = online_adj_access * 2 + online_nfeat_access[j]
                cost_offline = presample_miss[i] * 2 + presample_hit[i] * 8 + cold_nfeat_access[i][j]
                speedup.append(cost_online / cost_offline)
            speedup_all.append(speedup)
        plot_speedup(speedup_all, f"{dataset}: graph in disk")

        # Graph in disk, overlap presample
        speedup_all = []
        for i in range(len(presample_ratio)):
            speedup = []
            for j in range(len(cache_ratio)):
                cost_online = online_adj_access * 2 + online_nfeat_access[j]
                cost_offline = presample_miss[i] * 2 + cold_nfeat_access[i][j]
                speedup.append(cost_online / cost_offline)
            speedup_all.append(speedup)
        plot_speedup(speedup_all, f"{dataset}: graph in disk, overlap presample")

        namespace = f.readline()
