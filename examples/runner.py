import argparse
from train_single_thread import start


def get_predefined_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Training model device")
    parser.add_argument(
        "--batchsize", type=int, default=5000, help="batch size for training"
    )
    parser.add_argument("--dataset", default="ogbn-products", help="dataset")
    parser.add_argument(
        "--fanout", type=str, default="10,10,10", help="sampling fanout"
    )
    parser.add_argument("--model", type=str, default="SAGE", help="training model")
    parser.add_argument(
        "--dir",
        type=str,
        default="/nvme1n1/offgs_dataset",
        help="path to store subgraph",
    )
    parser.add_argument(
        "--cpu-cache-size", type=int, default=-1, help="cache size in bytes"
    )
    parser.add_argument(
        "--gpu-cache-size", type=int, default=-1, help="cache size in bytes"
    )
    parser.add_argument(
        "--feat-cache-size", type=int, default=-1, help="cache size in bytes"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=3, help="numbers of epoch in training"
    )
    ## argument whether use mega batch sampling
    parser.add_argument(
        "--mega_batch", action="store_true", help="whether use mega batch sampling"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="logs/train_single_thread_decompose.csv",
        help="log file",
    )
    args = parser.parse_args()
    print(args)
    return args


def process_args(args):
    if args.dataset == "ogbn-products":
        if args.feat_cache_size == 200000000:
            cpu_cache_size = [200000000, 100000000, 0]
            gpu_cache_size = [args.feat_cache_size - i for i in cpu_cache_size]
        elif args.feat_cache_size == 600000000:
            cpu_cache_size = [600000000, 300000000, 0]
            gpu_cache_size = [args.feat_cache_size - i for i in cpu_cache_size]
    elif args.dataset == "ogbn-papers100M":
        if args.feat_cache_size == 10000000000:
            cpu_cache_size = [10000000000, 5000000000, 2000000000]
            gpu_cache_size = [args.feat_cache_size - i for i in cpu_cache_size]
        elif args.feat_cache_size == 32000000000:
            cpu_cache_size = [32000000000, 24000000000]
            gpu_cache_size = [args.feat_cache_size - i for i in cpu_cache_size]
    elif args.dataset == "friendster":
        if args.feat_cache_size == 6400000000:
            cpu_cache_size = [6400000000, 3200000000, 0]
            gpu_cache_size = [args.feat_cache_size - i for i in cpu_cache_size]
        elif args.feat_cache_size == 19200000000:
            cpu_cache_size = [19200000000, 11200000000]
            gpu_cache_size = [args.feat_cache_size - i for i in cpu_cache_size]

    # generate args
    for i in range(len(cpu_cache_size)):
        yield {
            "cpu_cache_size": cpu_cache_size[i],
            "gpu_cache_size": gpu_cache_size[i],
        }


if __name__ == "__main__":
    args = get_predefined_args()

    for inputs in process_args(args):
        for key, value in inputs.items():
            setattr(args, key, value)
        print(args)
        start(args)
