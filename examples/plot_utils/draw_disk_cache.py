import matplotlib.pyplot as plt
from collections import defaultdict
from draw_utils import *
import math


dataset_num_nodes = {"friendster": 65608366, "igb-full": 269346174}


def plot_disk_cache_motivation_io(log_file, filter_ds, filter_cm, name):
    font_size = 16
    marker_size = 10
    line_width = 2
    color_list = ["blue", "green"]
    label_list = ["Global", "Segmented"]
    marker_list = ["x", "o"]
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

    sorted_keys = sorted(seg_size_dict.keys(), key=lambda x: int(x), reverse=True)

    plt_init(figsize=(5, 4), labelsize=font_size)
    ax = plt.gca()
    max_y = 0
    for i, k in enumerate(sorted_keys):
        sorted_v = seg_size_dict[k]
        cm_list = [float(x[0]) for x in sorted_v]
        io_ratio_before = [float(x[1]) for x in sorted_v]
        io_ratio = [float(x[2]) for x in sorted_v]
        max_y = max(np.max(io_ratio_before), max_y)
        if i == 0:
            plt.plot(
                cm_list,
                io_ratio_before,
                label="No-reorder",
                color="orange",
                marker="*",
                markersize=marker_size + 5,
                markerfacecolor="none",
                linewidth=line_width,
            )
        plt.plot(
            cm_list,
            io_ratio,
            label=label_list[i],
            color=color_list[i],
            marker=marker_list[i],
            markersize=marker_size,
            markerfacecolor="none",
            linewidth=line_width,
        )
    plt.xlabel("Number of Disk Cache Entries", fontsize=font_size)
    plt.ylabel("I/O Traffic Amplification", fontsize=font_size)
    # plt.title(f"{filter_ds} Disk Cache v.s. IO and Storage", fontsize=font_size)
    x_ticks = np.arange(0, cm_list[-1] + 2.5e6, 2.5e6)
    y_ticks = np.arange(1, max_y + 0.5, 0.5)
    plt.xticks(x_ticks, fontsize=font_size)
    plt.yticks(y_ticks, fontsize=font_size)
    ax.set_xticklabels([f"{int(x/1e6)}M" for x in x_ticks], fontsize=font_size)
    ax.set_yticklabels(y_ticks, fontsize=font_size)
    plt.grid(axis="y", linestyle="--")
    plt.legend(loc="best", fontsize=font_size - 2)
    plt_save_and_final(f"../figs/{filter_ds}-{filter_cm:g}-{name}.pdf")


def plot_disk_cache_motivation_io_decompose(log_file_list, name):
    font_size = 16
    marker_size = 10
    line_width = 2
    color_list = ["blue", "green"]
    label_list = ["Global", "Segmented"]

    plt_init(figsize=(5, 4), labelsize=font_size)
    ax = plt.gca()
    for i, log_file in enumerate(log_file_list):
        headers, lines = read_csv(log_file, has_header=None)
        print(f"[Note]headers:{headers}")

        pages_num_ratio = []
        for line in lines:
            pages_num_ratio.append(int(line[1]) / int(line[0]))
        pages_num_ratio = sorted(pages_num_ratio)

        plt.plot(
            range(len(pages_num_ratio)),
            pages_num_ratio,
            color=color_list[i],
            linewidth=line_width,
            label=label_list[i],
        )
    plt.xlabel("Minibatch ID", fontsize=font_size)
    plt.ylabel("Ratio of I/O over No-reorder", fontsize=font_size)
    x_ticks = np.arange(0, 601, 200)
    y_ticks = np.arange(0.1, 1.15, 0.15)
    plt.xticks(x_ticks, fontsize=font_size)
    plt.yticks(y_ticks, fontsize=font_size)
    ax.set_xticklabels(x_ticks, fontsize=font_size)
    ax.set_yticklabels([round(i, 2) for i in y_ticks], fontsize=font_size)
    plt.grid(axis="y", linestyle="--")
    plt.legend(loc="best", fontsize=font_size - 2)
    plt_save_and_final(f"../figs/{name}.pdf")


def plot_disk_cache(
    log_file,
    filter_ds,
    filter_cm,
    filter_seg,
    name,
    has_opt=False,
    draw_opt=False,
    draw_legend=True,
):
    num_nodes = dataset_num_nodes[filter_ds]
    font_size = 16
    marker_size = 10
    line_width = 2

    headers, lines = read_csv(log_file, has_header=None)
    print(f"[Note]headers:{headers}")

    seg_size_dict = defaultdict(list)
    for line in lines:
        if line[0].startswith("unused"):
            continue
        dataset, feat_cache_size, seg_size = line[0], float(line[3]), int(line[4])
        if (
            dataset != filter_ds
            or feat_cache_size != filter_cm
            or seg_size not in filter_seg
        ):
            continue
        if has_opt:
            seg_size, cache_num, io_before, io, io_opt, disk, original_ds = line[4:11]
            seg_size_dict[seg_size].append((cache_num, io_before, io, io_opt, disk))
        else:
            seg_size, cache_num, io_before, io, disk, original_ds = line[4:10]
            seg_size_dict[seg_size].append((cache_num, io_before, io, disk))

    for k, v in seg_size_dict.items():
        sorted_v = sorted(v, key=lambda x: float(x[0]))
        seg_size_dict[k] = sorted_v

    sorted_keys = sorted(seg_size_dict.keys(), key=lambda x: int(x))

    plt_init(figsize=(10, 4), labelsize=font_size, subplot_flag=True)

    plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=font_size,
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
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
        ax.plot(
            cm_list,
            io_ratio_before,
            label=f"{k}",
            linestyle="-",
            color=color_list[i],
            marker=marker_list[i],
            markersize=marker_size_list[i],
            markerfacecolor="none",
            linewidth=line_width,
        )
        ax.plot(
            cm_list,
            io_ratio,
            # label=f"{k}",
            linestyle="--",
            color=color_list[i],
            marker=marker_list[i],
            markersize=marker_size_list[i],
            markerfacecolor="none",
            linewidth=line_width,
        )
        if draw_opt:
            ax.plot(
                cm_list,
                io_ratio_opt,
                label=f"{k}",
                linestyle=":",
                color=color_list[i],
            )
    ax.plot(
        [0],
        [1],
        color="black",
        linestyle="-",
        label=f"w/o Reorder",
        linewidth=line_width,
    )
    ax.plot(
        [0],
        [1],
        color="black",
        linestyle="--",
        label=f"w/ Reorder",
        linewidth=line_width,
    )
    ax.plot(
        [0], [1], color="black", linestyle="-.", label=f"Blowup", linewidth=line_width
    )
    ax.set_xlabel("Number of Disk Cache Entries", fontsize=font_size)
    ax.set_ylabel("I/O Traffic Amplification", fontsize=font_size)
    yticks, xticks = np.arange(1, max_y + 1, 1, dtype=int), np.arange(
        0, cm_list[-1] + 5e6, 5e6
    )
    ax.set_xticks(xticks, xticks, fontsize=font_size)
    ax.set_xticklabels([f"{int(x/1e6)}M" for x in xticks], fontsize=font_size)
    ax.set_yticks(yticks, yticks, fontsize=font_size)
    ax.set_yticklabels(yticks, fontsize=font_size)
    ax.set_ylim(0.5, max_y + 0.5)
    ax.grid(axis="y", linestyle="--")
    if draw_legend:
        ax.legend(
            loc="upper center",
            fontsize=font_size - 4,
            bbox_to_anchor=(1.1, 1.15),
            ncol=6,
            edgecolor="black",
        )

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=font_size,
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    min_y, max_y = np.inf, 0
    for i, k in enumerate(sorted_keys):
        sorted_v = seg_size_dict[k]
        cm_list = [float(x[0]) for x in sorted_v]
        if has_opt:
            disk_ratio = [float(x[4]) for x in sorted_v]
        else:
            disk_ratio = [float(x[3]) for x in sorted_v]
        disk_size_blowup = [i * int(original_ds) / num_nodes for i in disk_ratio]
        min_y = min(min_y, np.min(disk_size_blowup))
        max_y = max(max_y, np.max(disk_size_blowup))
        ax.plot(
            cm_list,
            disk_size_blowup,
            linestyle="-.",
            color=color_list[i],
            marker=marker_list[i],
            markersize=marker_size_list[i],
            markerfacecolor="none",
            linewidth=line_width,
        )
    ax.set_xlabel("Number of Disk Cache Entries", fontsize=font_size)
    ax.set_ylabel("Disk Space Blowup", fontsize=font_size)
    yticks = np.arange(math.ceil(min_y), math.floor(max_y) + 1, 1, dtype=int)
    xticks = np.arange(0, cm_list[-1] + 5e6, 5e6)
    ax.set_xticks(xticks, xticks, fontsize=font_size)
    ax.set_xticklabels([f"{int(x/1e6)}M" for x in xticks], fontsize=font_size)
    ax.set_yticks(yticks, yticks, fontsize=font_size)
    ax.set_yticklabels(yticks, fontsize=font_size)
    ax.grid(axis="y", linestyle="--")
    plt_save_and_final(f"../figs/{filter_ds}-{filter_cm:g}-{name}.pdf")


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
    "../logs/global_disk_cache_motivate.csv",
    "friendster",
    4e9,
    "disk-cache-motivation-io",
)

plot_disk_cache_motivation_io_decompose(
    [
        "../logs/friendster-4e+09-global-disk-cache-io-decompose.csv",
        "../logs/friendster-4e+09-50-1e+07-disk-cache-io-decompose.csv",
    ],
    "friendster-disk-cache-motivation-io-decompose",
)

# plot_disk_cache(
#     "../logs/disk_cache_fs_ig.csv",
#     "friendster",
#     4e9,
#     [50, 100, 150],
#     "disk-cache",
#     True,
#     False,
#     True,
# )
# plot_disk_cache(
#     "../logs/disk_cache_fs_ig.csv",
#     "igb-full",
#     15e9,
#     [100, 200, 400],
#     "disk-cache",
#     True,
#     False,
#     True,
# )
