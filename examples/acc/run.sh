data_path=/nvme2n1/offgs_dataset

# papers100M
python train_online.py --dataset ogbn-papers100M --num-epoch 50 --hidden 256 --dropout 0.2 --fanout "10,15,20" --dir $data_path --model SAGE
sudo env PATH=$PATH python train_offline.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1 --num-epoch 50 --debug


python train_online.py --dataset ogbn-papers100M --num-epoch 50 --hidden 32 --dropout 0.2 --fanout "10,15,20" --dir $data_path --model GAT
sudo env PATH=$PATH python train_offline.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 32 --dropout 0.2 --model GAT --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1 --num-epoch 50 --debug


python train_online.py --dataset ogbn-papers100M --num-epoch 50 --hidden 256 --dropout 0.2 --fanout "10,15,20" --dir $data_path --model GCN
sudo env PATH=$PATH python train_offline.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model GCN --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1 --num-epoch 50 --debug


# mag240m
python train_online.py --dataset mag240m --num-epoch 50 --hidden 256 --dropout 0.2 --fanout "10,15,20" --dir $data_path --model SAGE
sudo env PATH=$PATH python train_offline.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1 --num-epoch 50 --debug


python train_online.py --dataset mag240m --num-epoch 50 --hidden 32 --dropout 0.2 --fanout "10,15,20" --dir $data_path  --model GAT
sudo env PATH=$PATH python train_offline.py --dataset mag240m --fanout "10,15,20" --hidden 32 --dropout 0.2 --model GAT --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1 --num-epoch 50 --debug


python train_online.py --dataset mag240m --num-epoch 50 --hidden 256 --dropout 0.2 --fanout "10,15,20" --dir $data_path --model GCN
sudo env PATH=$PATH python train_offline.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model GCN --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1 --num-epoch 50 --debug



# igb-full
python train_online.py --dataset igb-full --num-epoch 20 --hidden 256 --dropout 0 --fanout "10,15,20" --dir $data_path --model SAGE --ratio 0.1
sudo env PATH=$PATH python train_offline.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir $data_path --ratio 0.1 --blowup -1 --num-epoch 20 --debug


python train_online.py --dataset igb-full --num-epoch 20 --hidden 32 --dropout 0 --fanout "10,15,20" --dir $data_path --model GAT --ratio 0.1
sudo env PATH=$PATH python train_offline.py --dataset igb-full --fanout "10,15,20" --hidden 32 --dropout 0 --model GAT --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir $data_path --ratio 0.1 --blowup -1 --num-epoch 20 --debug
