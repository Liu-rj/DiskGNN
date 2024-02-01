import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import csv

color_list = ["r", "g", "c", "m", "k"]
# marker_list = ["o", "x", "v", "*", "^"]

# zuo_green = "#5daa2d"
# zuo_oriange = "#f2883f"
# zuo_blue = "#268be6"
# zuo_red = "#fc0464"
# zuo_purple = "#7863d6"

# color_list = [zuo_blue, zuo_green, zuo_oriange, zuo_purple, zuo_red]
# color_list = "gbcmrw"
# marker_list = "oxvD*"
# marker_size_list = [10, 10, 10, 10, 15]
# linestyle_list = ["--", "--", "--", "--", "-"]


# def get_fmt(id):
#     return f"{marker_list[id]}{linestyle_list[id]}"


# def get_color(id):
#     return color_list[id]


def plt_init(figsize=None, labelsize=24, subplot_flag=False):
    if figsize is not None:
        plt.figure(figsize=figsize)

    plt.clf()
    if not subplot_flag:
        ax = plt.gca()
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=labelsize,
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


def read_csv(input_path=None, has_header=True):
    # print(f"[Note]read_csv from {input_path}")
    with open(input_path, "r") as file:
        reader = csv.reader(file)
        if has_header:
            headers = next(reader)
        else:
            headers = None
        # print(f"[Note]headers:{headers}")
        elements = [row for row in reader if len(row) > 0]

    return headers, elements
