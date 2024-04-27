import numpy as np
import matplotlib.pyplot as plt
import csv


def plt_init(figsize=None, plot_num=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    # if plot_num is not None:
    plt.subplot(1, 2, plot_num)
    # plt.clf()
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
    patch,
    ylabel=None,
    yticks_list=None,
    na_str="N/A",
    save_path=None,
    lengend=True,
    titles=None,
):
    plt.figure(figsize=(20, 3))
    plt.clf()
    # fix parameter
    font_size = 14
    hatch_list = {
        "DGL": None,
        "gSampler": None,
        "PyG": "..",
        "GunRock": "--",
        "SkyWalker": "//",
        "cuGraph": "x",
        "CPU": "||",
    }
    color_list = {
        "DGL": "w",
        "gSampler": "k",
        "PyG": "w",
        "GunRock": "w",
        "SkyWalker": "w",
        "cuGraph": "w",
        "CPU": "w",
    }
    for it, (elements, labels, xlabels) in enumerate(patch):
        yticks = yticks_list[it]
        value_limit = yticks[-1] + 1
        plt.subplot(1, 2, it + 1)
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
        num_series = len(labels)
        num_elements_per_series = len(xlabels)
        width = 1.5
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
                        # if res == 45.36:
                        #     res = '45.36  '
                        # elif res == 125.12:
                        #     res = '  125.12'
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
            ax.bar_label(
                container, plot_label, fontsize=11, fontweight="bold", zorder=20
            )

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
        # ax.set_title(title, fontsize=font_size)
        ax.set_xlabel(titles[it], fontsize=font_size, fontweight="bold")
        ax.set_ylabel("Time (s)", fontsize=font_size, fontweight="bold")
        plt.grid(axis="y", linestyle="-.", zorder=0)

    if lengend:
        ax.legend(
            fontsize=font_size,
            edgecolor="k",
            ncols=6,
            loc="upper center",
            bbox_to_anchor=(-0.1, 1.3),
        )
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
        patch = []
        save_path = ""
        titles = []
        name = ""
        for row in reader:
            if len(row) == 1:
                if len(elements) > 0:
                    patch.append([elements, labels, headers])
                titles.append(row[0])
                name = name + row[0]
                elements = []
                labels = []
            else:
                labels.append(row[0])
                elements.append(row[1:])

        save_path = f"figures/{name}.pdf"
        patch.append([elements, labels, headers])
        print(patch)
        if len(elements) > 0:
            plt_bar(
                patch,
                save_path=save_path,
                lengend=lengend,
                titles=titles,
                yticks_list=yticks,
            )


if __name__ == "__main__":
    draw_figure("ladies_asgcn.csv", [np.arange(0, 51, 10), np.arange(0, 51, 10)], True)
    draw_figure("pass_shadow.csv", [np.arange(0, 101, 20), np.arange(0, 31, 5)], False)
