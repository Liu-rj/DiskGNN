import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import font_manager
import csv

# font_dirs = ["/home/ubuntu/helvetica-255"]
# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

# for font_file in font_files:
#     font_manager.fontManager.addfont(font_file)

color_list = ["r", "g", "c", "m", "k"]
marker_list = ["*", "v", "o", "x", "^"]
marker_size_list = [15, 10, 10, 10, 10]

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
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["hatch.linewidth"] = 1
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["text.usetex"] = True
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
