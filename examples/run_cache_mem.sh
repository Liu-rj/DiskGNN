# Ogbn-papers100M
sudo env PATH=$PATH python feat_packing.py --dataset ogbn-papers100M --fanout "10,15,20" --feat-cache-size 5e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python feat_packing.py --dataset ogbn-papers100M --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python feat_packing.py --dataset ogbn-papers100M --fanout "10,15,20" --feat-cache-size 25e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1

sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 8e9 --cpu-cache-size 17e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1


# MAG240M
sudo env PATH=$PATH python feat_packing.py --dataset mag240m --fanout "10,15,20" --feat-cache-size 10e9 --store-path /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python feat_packing.py --dataset mag240m --fanout "10,15,20" --feat-cache-size 30e9 --store-path /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python feat_packing.py --dataset mag240m --fanout "10,15,20" --feat-cache-size 50e9 --store-path /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1

sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 10e9 --cpu-cache-size 20e9 --dir /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 15e9 --cpu-cache-size 35e9 --dir /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1


# Friendster
sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 10e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1

sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1


# IGB-HOM
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 45e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 75e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1

sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 15e9 --cpu-cache-size 30e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 15e9 --cpu-cache-size 60e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1
