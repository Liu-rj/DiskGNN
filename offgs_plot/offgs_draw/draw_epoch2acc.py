# draw the epoch to accuracy curve, data is in logs/data.csv
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True
font_size = 18


micro_green = "#5daa2d"
micro_oriange = "#f2883f"
micro_blue = "#268be6"
micro_red = "#fc0464"
micro_purple = "#7863d6"


def plt_save_and_final(save_path):
    print(f"[Note]Save to {save_path}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()


# scan through all files in the logs folder and store the files into a list
all_acc_files = []
for file in os.listdir("../offgs_data/acc"):
    all_acc_files.append(file)

print("Acc Files:", all_acc_files)

all_acc_files = sorted(all_acc_files)

plt.figure(figsize=(5, 4))

datasets = ["ogbn-papers100M", "mag240m", "igb-full"]
models = ["SAGE"]
for dataset in datasets:
    for model in models:
        target_acc_files = [None, None, None]
        for acc_file in all_acc_files:
            if dataset in acc_file and model in acc_file:
                print("Processing", acc_file)
                if "online" in acc_file:
                    target_acc_files[0] = acc_file
                elif "mariusgnn" in acc_file:
                    target_acc_files[1] = acc_file
                elif "offline" in acc_file:
                    target_acc_files[2] = acc_file
                else:
                    print("Error: Unknown file", acc_file)
        for acc_file in target_acc_files:
            if acc_file is None:
                continue
            data = np.loadtxt(f"../offgs_data/acc/{acc_file}", delimiter=",")
            epoch = data[:, 0]
            if "mariusgnn" in acc_file:
                acc_col = 1 if dataset == "mag240m" else 2
                accuracy = data[:, acc_col] / 100
                label = "MariusGNN"
                marker = "x"
                color = micro_blue
                epoch = epoch[1:]
                accuracy = accuracy[1:]
            else:
                epoch[6:] = epoch[6:] + 1
                acc_col = 3 if dataset == "mag240m" else 4
                accuracy = data[:, acc_col] / 100
                label = "DiskGNN" if "offline" in acc_file else "Ginex"
                marker = "o" if "offline" in acc_file else "v"
                color = micro_red if "offline" in acc_file else micro_green
                epoch = epoch[2:]
                accuracy = accuracy[2:]

            max_acc = np.max(accuracy)
            ax = plt.gca()
            ax.plot(
                epoch,
                accuracy,
                color=color,
                marker=marker,
                label=label,
                markersize=5,
                markerfacecolor="none",
            )
            ax.axhline(max_acc, color=color, linestyle="--", linewidth=1.5)
            if "offline" in acc_file or "mariusgnn" in acc_file:
                ax.text(-7, max_acc - 0.01, round(max_acc, 2))
        ax.set_xlabel("Number of Epochs", fontsize=font_size)
        ax.set_ylabel("Accuracy", fontsize=font_size)
        # plt.title("Epoch to Accuracy Curve")
        xticks = np.arange(0, 60, 10)
        yticks = np.round(np.arange(0.1, 1.2, 0.2), 2)
        ax.set_xticks(xticks, xticks, fontsize=font_size)
        ax.set_xticklabels(xticks, fontsize=font_size)
        ax.set_yticks(yticks, yticks, fontsize=font_size)
        ax.set_yticklabels(yticks, fontsize=font_size)
        if dataset == "ogbn-papers100M":
            ax.set_ylim(0.05, 0.7)
        else:
            ax.set_ylim(0.1, 0.7)
        ax.grid(axis="y", linestyle="--")
        ax.legend(
            loc="best",
            fontsize=font_size - 2,
            # bbox_to_anchor=(1.1, 1.15),
            ncol=1,
            edgecolor="black",
        )

        plt_save_and_final(f"../figures/{dataset}_{model}_epoch_to_accuracy.pdf")
