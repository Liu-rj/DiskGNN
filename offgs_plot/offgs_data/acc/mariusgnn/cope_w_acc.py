import argparse
import numpy as np

## add file name
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--filename",
    type=str,
    help="filename",
    default="/nvme2n1/renjie/marius/datasets/my_ogbn-papers100M_8192/marius_gs_mem_1000_model_sage_acc_new.log",
)
args = parser.parse_args()
import re


def parse_log_file(log_file_path):
    epochs = []
    validation_accuracies = []
    test_accuracies = []

    epoch_pattern = re.compile(r"Starting training epoch (\d+)")
    validation_pattern = re.compile(
        r"Evaluating validation set.*?Accuracy: ([\d.]+)%", re.DOTALL
    )
    test_pattern = re.compile(r"Evaluating test set.*?Accuracy: ([\d.]+)%", re.DOTALL)

    with open(log_file_path, "r") as file:
        log = file.read()

    epoch_matches = epoch_pattern.findall(log)
    validation_matches = validation_pattern.findall(log)
    test_matches = test_pattern.findall(log)

    for i in range(len(epoch_matches)):
        epoch = int(epoch_matches[i])
        validation_accuracy = float(validation_matches[i])
        test_accuracy = float(test_matches[i])
        epochs.append(epoch)
        validation_accuracies.append(validation_accuracy)
        test_accuracies.append(test_accuracy)
    max_val_num, max_val_index = max(
        validation_accuracies
    ), validation_accuracies.index(max(validation_accuracies))
    corresponding_test_accuracy = test_accuracies[max_val_index]
    ## print the max val acc and corresponding test acc and epoch
    print(
        "max val acc:",
        max_val_num,
        "corresponding test acc:",
        corresponding_test_accuracy,
        "epoch:",
        epochs[max_val_index],
    )
    epoch_accuracy_list = list(zip(epochs, validation_accuracies, test_accuracies))
    return epoch_accuracy_list


res = parse_log_file(args.filename)
print(res)
dataset = args.filename.split("/")[-1].split("_")[-1][:-4]
print(dataset)
np.savetxt(f"../mariusgnn_{dataset}_SAGE.csv", res, delimiter=",", fmt="%s")
# with open(args.filename) as f:
#     ## read every line and analyze
#     epoch_runtime = []
#     for line in f:

#         ## recognize this patern '[2024-04-11 06:29:00.858] [info] [trainer.cpp:62] Epoch Runtime: 205524ms' and extract the num
#         if line.find('Epoch Runtime') != -1:
#             ## find the index of the first number
#             start = line.find('Epoch Runtime') + len('Epoch Runtime: ')
#             ## find the index of the first non-number
#             end = start
#             while line[end].isdigit():
#                 end += 1
#             ## extract the number
#             num = line[start:end]
#             epoch_runtime.append(int(num))
#     ## calculate the average 保留两位小数
#     print(round(sum(epoch_runtime) / len(epoch_runtime)/1000,2))
#     epoch_runtime=epoch_runtime[1:]
#     ## print wo warm up
#     print(round(sum(epoch_runtime) / len(epoch_runtime)/1000,2))
