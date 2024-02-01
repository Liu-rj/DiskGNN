import matplotlib.pyplot as plt
from collections import defaultdict
from draw_utils import *


def plot_disk_cache(filter_ds, filter_cm):
    headers, lines = read_csv("../logs/disk_cache.csv", has_header=None)
    print(f"[Note]headers:{headers}")

    seg_size_dict = defaultdict(list)
    for line in lines:
        if line[0].startswith("unused"):
            continue
        dataset, feat_cache_size = line[0], line[3]
        if dataset != filter_ds or feat_cache_size != filter_cm:
            continue
        seg_size, cache_num, io_before, io, disk = line[4:]
        seg_size_dict[seg_size].append((cache_num, io_before, io, disk))

    for k, v in seg_size_dict.items():
        sorted_v = sorted(v, key=lambda x: int(x[0]))
        seg_size_dict[k] = sorted_v

    sorted_keys = sorted(seg_size_dict.keys(), key=lambda x: int(x))

    plt_init(figsize=(8, 6), labelsize=14)
    max_y = 0
    for i, k in enumerate(sorted_keys):
        sorted_v = seg_size_dict[k]
        cm_list = [int(x[0]) for x in sorted_v]
        io_ratio_before = [float(x[1]) for x in sorted_v]
        io_ratio = [float(x[2]) for x in sorted_v]
        disk_ratio = [float(x[3]) for x in sorted_v]
        max_y = max(np.max(io_ratio_before), max_y)
        plt.plot(
            cm_list,
            io_ratio_before,
            label=f"IO Traffic w/o Reorder {k}",
            linestyle="--",
            color=color_list[i],
        )
        plt.plot(
            cm_list,
            io_ratio,
            label=f"IO Traffic w/ Reorder {k}",
            linestyle="-",
            color=color_list[i],
        )
        plt.plot(
            cm_list,
            disk_ratio,
            label=f"Disk Size {k}",
            linestyle=":",
            color=color_list[i],
        )
    plt.xlabel("Disk Cache Node Num")
    plt.ylabel("Ratio")
    plt.title(f"{filter_ds} Disk Cache v.s. IO and Storage")
    plt.yticks(np.arange(0, max_y + 0.5, 0.5))
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt_save_and_final(f"../figs/{filter_ds}-{filter_cm}-disk-cache.png")


plot_disk_cache("friendster", "10000000000")
plot_disk_cache("igb-full", "10000000000")
plot_disk_cache("igb-full", "30000000000")
