import matplotlib.pyplot as plt
from collections import defaultdict
from draw_utils import *


def plot_disk_cache_motivation_io(log_file, filter_ds, filter_cm, name):
    font_size = 22
    headers, lines = read_csv(log_file, has_header=None)
    print(f"[Note]headers:{headers}")

    seg_size_dict = defaultdict(list)
    for line in lines:
        if line[0].startswith("unused"):
            continue
        dataset, feat_cache_size = line[0], float(line[3])
        if dataset != filter_ds or feat_cache_size != filter_cm:
            continue
        seg_size, cache_num, io_before, io, disk = line[4:9]
        seg_size_dict[seg_size].append((cache_num, io_before, io, disk))

    for k, v in seg_size_dict.items():
        sorted_v = sorted(v, key=lambda x: float(x[0]))
        seg_size_dict[k] = sorted_v

    sorted_keys = sorted(seg_size_dict.keys(), key=lambda x: int(x))

    plt_init(figsize=(7.5, 6), labelsize=font_size)
    max_y = 0
    for i, k in enumerate(sorted_keys):
        sorted_v = seg_size_dict[k]
        cm_list = [float(x[0]) for x in sorted_v]
        io_ratio_before = [float(x[1]) for x in sorted_v]
        io_ratio = [float(x[2]) for x in sorted_v]
        max_y = max(np.max(io_ratio_before), max_y)
        plt.plot(
            cm_list,
            io_ratio_before,
            label=f"I/O Traffic w/o Reorder",
            color="blue",
            marker="o",
        )
        plt.plot(
            cm_list,
            io_ratio,
            label=f"I/O Traffic w/ Reorder",
            color="green",
            marker="x",
        )
    plt.xlabel("Number of Disk Cache Entries", fontsize=font_size)
    plt.ylabel("I/O Traffic Amplification", fontsize=font_size)
    # plt.title(f"{filter_ds} Disk Cache v.s. IO and Storage", fontsize=font_size)
    plt.yticks(np.arange(1, max_y + 0.5, 0.5), fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.grid(axis="y", linestyle="--")
    plt.legend(loc="best", fontsize=font_size)
    plt_save_and_final(f"../figs/{filter_ds}-{filter_cm:g}-{name}.pdf")


def plot_disk_cache_motivation_io_decompose(log_file, name):
    font_size = 22
    headers, lines = read_csv(log_file, has_header=None)
    print(f"[Note]headers:{headers}")

    pages_num_ratio = []
    for line in lines:
        pages_num_ratio.append(int(line[1]) / int(line[0]))
    pages_num_ratio = sorted(pages_num_ratio)

    plt_init(figsize=(7.5, 6), labelsize=font_size)
    plt.plot(range(len(pages_num_ratio)), pages_num_ratio, color="orange")
    plt.xlabel("Minibatch ID", fontsize=font_size)
    plt.ylabel("IO Traffic Reduction", fontsize=font_size)
    plt.xticks(np.arange(0, 601, 200), fontsize=font_size)
    plt.yticks(np.arange(0.1, 1.1, 0.15), fontsize=font_size)
    plt.grid(axis="y", linestyle="--")
    plt_save_and_final(f"../figs/{name}.pdf")


def plot_disk_cache(log_file, filter_ds, filter_cm, name, has_opt=False):
    headers, lines = read_csv(log_file, has_header=None)
    print(f"[Note]headers:{headers}")

    seg_size_dict = defaultdict(list)
    for line in lines:
        if line[0].startswith("unused"):
            continue
        dataset, feat_cache_size = line[0], float(line[3])
        if dataset != filter_ds or feat_cache_size != filter_cm:
            continue
        if has_opt:
            seg_size, cache_num, io_before, io, io_opt, disk = line[4:10]
            seg_size_dict[seg_size].append((cache_num, io_before, io, io_opt, disk))
        else:
            seg_size, cache_num, io_before, io, disk = line[4:9]
            seg_size_dict[seg_size].append((cache_num, io_before, io, disk))

    for k, v in seg_size_dict.items():
        sorted_v = sorted(v, key=lambda x: float(x[0]))
        seg_size_dict[k] = sorted_v

    sorted_keys = sorted(seg_size_dict.keys(), key=lambda x: int(x))

    plt_init(figsize=(8, 6), labelsize=14)
    max_y = 0
    for i, k in enumerate(sorted_keys):
        sorted_v = seg_size_dict[k]
        cm_list = [float(x[0]) for x in sorted_v]
        io_ratio_before = [float(x[1]) for x in sorted_v]
        io_ratio = [float(x[2]) for x in sorted_v]
        if has_opt:
            io_ratio_opt = [float(x[3]) for x in sorted_v]
            disk_ratio = [float(x[4]) for x in sorted_v]
        else:
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
        if has_opt:
            plt.plot(
                cm_list,
                io_ratio_opt,
                label=f"IO Traffic w/ Reorder Opt {k}",
                linestyle=":",
                color=color_list[i],
            )
        plt.plot(
            cm_list,
            disk_ratio,
            label=f"Disk Size {k}",
            linestyle="-.",
            color=color_list[i],
        )
    plt.xlabel("Disk Cache Node Num")
    plt.ylabel("Ratio")
    plt.title(f"{filter_ds} Disk Cache v.s. IO and Storage")
    plt.yticks(np.arange(0, max_y + 0.5, 0.5))
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt_save_and_final(f"../figs/{filter_ds}-{filter_cm:g}-{name}.png")


# plot_disk_cache("../logs/disk_cache.csv", "friendster", 1e10, "disk-cache")
# plot_disk_cache("../logs/disk_cache.csv", "igb-full", 3e10, "disk-cache")
# plot_disk_cache("../logs/disk_cache.csv", "igb-full", 1e10, "disk-cache")

# plot_disk_cache("../logs/disk_cache_2.csv", "friendster", 1e10, "disk-cache-2")
# plot_disk_cache("../logs/disk_cache_2.csv", "igb-full", 3e10, "disk-cache-2")
# plot_disk_cache("../logs/disk_cache_2.csv", "igb-full", 1e10, "disk-cache-2")

# plot_disk_cache("../logs/disk_cache_new.csv", "friendster", 1e10, "disk-cache", True)
# plot_disk_cache("../logs/disk_cache_new.csv", "igb-full", 3e10, "disk-cache", True)
# plot_disk_cache("../logs/disk_cache_new.csv", "igb-full", 1e10, "disk-cache", True)

plot_disk_cache_motivation_io(
    "../logs/global_disk_cache.csv", "friendster", 4e9, "disk-cache-motivation-io"
)

# plot_disk_cache_motivation_io_decompose(
#     "../logs/friendster-4e+09-global-disk-cache-io-decompose.csv",
#     "friendster-disk-cache-motivation-io-decompose",
# )
