# data_path=/nvme2n1/offgs_dataset
# data_path=/nvme3n1/offgs_dataset
data_path=/data/offgs_dataset

# Ogbn-papers100M
python prepare_dataset.py --dataset=ogbn-papers100M
sudo env PATH=$PATH python sampling.py --dataset ogbn-papers100M --fanout "10,15,20" --store-path $data_path --ratio 1.0
sudo env PATH=$PATH python feat_packing.py --dataset ogbn-papers100M --fanout "10,15,20" --feat-cache-size 5e9 --store-path $data_path --ratio 1.0 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 32 --dropout 0.2 --model GAT --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1
# acc
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1 --device 1 --num-epoch 50 --debug
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 32 --dropout 0.2 --model GAT --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1 --device 1 --num-epoch 50 --debug



# MAG240M
python prepare_dataset.py --dataset mag240m --store-path $data_path
sudo env PATH=$PATH python sampling.py --dataset mag240m --fanout "10,15,20" --store-path $data_path --ratio 1.0
sudo env PATH=$PATH python feat_packing.py --dataset mag240m --fanout "10,15,20" --feat-cache-size 10e9 --store-path $data_path --ratio 1.0 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,15,20" --hidden 32 --dropout 0.2 --model GAT --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1
# acc
sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1 --device 1 --num-epoch 50 --debug
sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,15,20" --hidden 32 --dropout 0.2 --model GAT --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1 --device 1 --num-epoch 50 --debug



# Friendster
python prepare_dataset.py --dataset=friendster
sudo env PATH=$PATH python sampling.py --dataset friendster --fanout "10,15,20" --store-path $data_path --ratio 0.1 --device 1
sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path $data_path --ratio 1.0 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir $data_path --ratio 1.0 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 32 --dropout 0 --model GAT --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir $data_path --ratio 1.0 --blowup -1



# IGB-FULL
python prepare_dataset.py --dataset igb-full --dataset_size full --path /efs/user/dataset/igb_full --store-path $data_path
sudo env PATH=$PATH python sampling.py --dataset igb-full --fanout "10,15,20" --store-path $data_path --ratio 0.1
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path $data_path --ratio 0.1 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir $data_path --ratio 0.1 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 32 --dropout 0 --model GAT --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir $data_path --ratio 0.1 --blowup -1
# acc
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir $data_path --ratio 0.1 --blowup -1 --device 1 --num-epoch 20 --debug
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 32 --dropout 0 --model GAT --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir $data_path --ratio 0.1 --blowup -1 --device 1 --num-epoch 20 --debug
