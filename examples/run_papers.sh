data_path=/nvme2n1/offgs_dataset

# python prepare_dataset.py --dataset ogbn-papers100M-seeds --store-path $data_path

sudo sysctl -w vm.drop_caches=1
python sampling.py --dataset ogbn-papers100M-seeds --fanout "10,10,10" --store-path $data_path --ratio 1.0


sudo sysctl -w vm.drop_caches=1
python batched_packing.py --dataset ogbn-papers100M-seeds --fanout "10,10,10" --feat-cache-size 5e9 --store-path $data_path --ratio 1.0 --blowup -1

sudo sysctl -w vm.drop_caches=1
python batched_packing.py --dataset ogbn-papers100M-seeds --fanout "10,10,10" --feat-cache-size 10e9 --store-path $data_path --ratio 1.0 --blowup -1

sudo sysctl -w vm.drop_caches=1
python batched_packing.py --dataset ogbn-papers100M-seeds --fanout "10,10,10" --feat-cache-size 15e9 --store-path $data_path --ratio 1.0 --blowup -1


sudo sysctl -w vm.drop_caches=1
python train_multi_thread.py --dataset ogbn-papers100M-seeds --fanout "10,10,10" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 0 --cpu-cache-size 5e9 --dir $data_path --ratio 1.0 --blowup -1

sudo sysctl -w vm.drop_caches=1
python train_multi_thread.py --dataset ogbn-papers100M-seeds --fanout "10,10,10" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 0 --cpu-cache-size 10e9 --dir $data_path --ratio 1.0 --blowup -1

sudo sysctl -w vm.drop_caches=1
python train_multi_thread.py --dataset ogbn-papers100M-seeds --fanout "10,10,10" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 0 --cpu-cache-size 15e9 --dir $data_path --ratio 1.0 --blowup -1
