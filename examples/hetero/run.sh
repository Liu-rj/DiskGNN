data_path=/nvme2n1/offgs_dataset

# ogbn-mag
python prepare_dataset.py --dataset ogbn-mag --store-path $data_path
python sampling.py --dataset ogbn-mag --store-path $data_path
python feat_packing.py --dataset ogbn-mag --fanout "25,10" --feat-cache-size 0.8e9 --store-path $data_path --ratio 1.0 --blowup -1
python train_multi_thread.py --dataset ogbn-mag --fanout "25,10" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 0 --cpu-cache-size 0.8e9 --dir $data_path --ratio 1.0 --blowup -1


# ogb-lsc-mag240m
# python sampling.py --dataset ogb-lsc-mag240m --store-path $data_path