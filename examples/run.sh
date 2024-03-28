# Ogbn-papers100M
python prepare_dataset.py --dataset=ogbn-papers100M
# sudo /opt/conda/envs/SSD_GNN/bin/python sampling.py --dataset ogbn-papers100M --fanout="10,10,10" --store-path /nvme2n1/offgs_dataset --ratio 1.0
sudo env PATH=$PATH python sampling.py --dataset ogbn-papers100M --fanout "10,15,20" --store-path /nvme2n1/offgs_dataset --ratio 1.0
sudo env PATH=$PATH python feat_packing.py --dataset ogbn-papers100M --fanout "10,15,20" --feat-cache-size 5e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
# acc
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1 --device 1 --num-epoch 50 --debug
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1 --device 1 --num-epoch 50 --debug



# Friendster
python prepare_dataset.py --dataset=friendster
# sudo /opt/conda/envs/SSD_GNN/bin/python sampling.py --dataset=friendster --fanout="10,10,10"
sudo env PATH=$PATH python sampling.py --dataset friendster --fanout "10,15,20" --store-path /nvme2n1/offgs_dataset --ratio 0.1 --device 1
sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1



# IGB-FULL
python prepare_dataset.py --dataset igb-full --dataset_size full --path /efs/rjliu/dataset/igb_full --store-path /nvme1n1/offgs_dataset
sudo env PATH=$PATH python sampling.py --dataset igb-full --fanout "10,15,20" --store-path /nvme1n1/offgs_dataset --ratio 0.1
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1
# acc
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir /nvme1n1/offgs_dataset --ratio 0.2 --blowup -1 --device 1 --num-epoch 50 --debug
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1 --device 1 --num-epoch 100 --debug
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir /nvme1n1/offgs_dataset --ratio 0.02 --blowup -1 --device 1 --num-epoch 300 --debug --log_every 10


# MAG240M
python prepare_dataset.py --dataset mag240m --store-path /nvme1n1/offgs_dataset
sudo env PATH=$PATH python sampling.py --dataset mag240m --fanout "10,15,20" --store-path /nvme1n1/offgs_dataset --ratio 1.0
sudo env PATH=$PATH python feat_packing.py --dataset mag240m --fanout "10,15,20" --feat-cache-size 10e9 --store-path /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1
# acc
sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1 --device 1 --num-epoch 50 --debug
