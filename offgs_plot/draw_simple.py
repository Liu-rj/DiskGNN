import numpy as np
import matplotlib.pyplot as plt
import csv
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True

def plt_init(figsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.clf()
    ax = plt.gca()
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=24,
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
    )


def plt_save_and_final(save_path):
    print(f"[Note]Save to {save_path}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def plt_bar(
    elements,
    labels,
    xlabels,
    ylabel=None,
    yticks=None,
    na_str="N/A",
    save_path=None,
    lengend=True,
    title=None,
):
    plt_init(figsize=(20, 3))
    # fix parameter
    font_size = 14
    
    hatch_list = {
        "DGL": "\\\\\\\\\\",
        "gSampler": '/////',
        "PyG": "/////",
        "GunRock": "/////",
        "SkyWalker": "\\\\\\\\\\",
        "cuGraph": "\\\\\\\\\\",
        "CPU": "||",
    }
    color_list = {
        "DGL": "w",
        "gSampler": "w",
        "PyG": "w",
        "GunRock": "w",
        "SkyWalker": "w",
        "cuGraph": "w",
        "CPU": "w",
    }
    edgecolors = {"dimgrey", "lightseagreen", "tomato", "slategray", "silver"}

    ax = plt.gca()
    num_series = len(labels)
    num_elements_per_series = len(xlabels)
    value_limit = yticks[-1]
    width = 1.2
    for i in range(num_series):
        plot_x = [
            (num_series + 3) * j + i * width for j in range(num_elements_per_series)
        ]
        # handle N/A
        plot_y = []
        plot_label = []
        for e in elements[i]:
            if isfloat(e) or e.isdigit():
                val_e = float(e)
                if val_e < value_limit:
                    plot_y.append(float(e))
                    if isfloat(e):
                        res = float(e)
                    else:
                        res = int(e)
                    plot_label.append(res)
                else:
                    plot_y.append(value_limit)
                    if isfloat(e):
                        res = float(e)
                    else:
                        res = int(e)
                    plot_label.append(res)
            else:
                plot_y.append(0.01)
                plot_label.append(na_str)

        container = ax.bar(
            plot_x,
            plot_y,
            width=width,
            edgecolor="k",
            hatch=hatch_list[labels[i]],
            color=color_list[labels[i]],
            label=labels[i],
            zorder=10,
        )
        # print("ok")
        ax.bar_label(container, plot_label, fontsize=13, zorder=20, fontweight="bold")

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)

    if yticks is not None:
        ax.set_yticks(yticks, yticks, fontsize=font_size)
        ax.set_ylim(0, value_limit)

    plot_xticks = [
        (num_series + 3) * j + (width / 2) * (num_series - 1)
        for j in range(num_elements_per_series)
    ]

    ax.set_xticks(plot_xticks, xlabels, fontsize=font_size)
    if lengend:
        ax.legend(
            fontsize=font_size,
            edgecolor="k",
            ncol=6,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.30),
        )
    # ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(title, fontsize=font_size, fontweight="bold")
    ax.set_ylabel("Time (\\textit{s})", fontsize=font_size, fontweight="bold")
    # plt.grid(axis="y", linestyle="-.", zorder=0)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.show()
    if save_path is not None:
        plt_save_and_final(save_path=save_path)


def isfloat(val):
    return all(
        [[any([i.isnumeric(), i in [".", "e"]]) for i in val], len(val.split(".")) == 2]
    )


def draw_figure(input_path, yticks, lengend):
    with open(input_path, "r") as file:
        reader = csv.reader(file)
        headers = next(reader)
        print(f"[Note]headers:{headers}")
        elements = []
        labels = []
        save_path = ""
        title = ""
        for row in reader:
            if len(row) == 1:
                if len(elements) > 0:
                    plt_bar(
                        elements,
                        labels,
                        headers,
                        save_path=save_path,
                        lengend=lengend,
                        title=title,
                        yticks=yticks,
                    )
                title = row[0]
                save_path = f"figures/{title}.pdf"
                elements.clear()
                labels.clear()
            else:
                labels.append(row[0])
                elements.append(row[1:])

            # print(f"[Note]len: {len(row)}")
        if len(elements) > 0:
            plt_bar(
                elements,
                labels,
                headers,
                save_path=save_path,
                lengend=lengend,
                title=title,
                yticks=yticks,
            )


if __name__ == "__main__":
    # draw_figure("./simple.csv")
    # draw_figure("./complex.csv")
    # draw_figure("./complex_t4.csv")
    # draw_figure("./new_data/deepwalk.csv", np.arange(0, 5.5, 1), True)
    # draw_figure("./new_data/node2vec.csv", np.arange(0, 11, 2), False)
    draw_figure("/home/ubuntu/offgs_plot/offgs_plot/graphsage.csv", np.arange(0, 5.5, 1), False)
