# Friendster
sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup 3.0
sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup 5.0

sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup 3.0
sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup 5.0


# IGB-HOM
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1 --blowup 3.0
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1 --blowup 5.0
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1 --blowup 7.0

sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup 3.0
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup 5.0
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup 7.0
