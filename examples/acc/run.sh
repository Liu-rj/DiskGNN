# papers100M
# python train_online.py --dataset ogbn-papers100M --num-epoch 50 --hidden 256 --dropout 0.2 --fanout "10,15,20" --dir /nvme2n1/offgs_dataset --model SAGE
# sudo env PATH=$PATH python train_offline.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1 --num-epoch 50 --debug

# python train_online.py --dataset ogbn-papers100M --num-epoch 50 --hidden 32 --dropout 0.2 --fanout "10,15,20" --dir /nvme2n1/offgs_dataset --model GAT
# sudo env PATH=$PATH python train_offline.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 32 --dropout 0.2 --model GAT --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1 --num-epoch 50 --debug



# mag240m
# python train_online.py --dataset mag240m --num-epoch 50 --hidden 256 --dropout 0.2 --fanout "10,15,20" --dir /nvme2n1/offgs_dataset --model SAGE
# sudo env PATH=$PATH python train_offline.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1 --num-epoch 50 --debug


python train_online.py --dataset mag240m --num-epoch 50 --hidden 32 --dropout 0.2 --fanout "10,15,20" --dir /nvme2n1/offgs_dataset  --model GAT
sudo env PATH=$PATH python train_offline.py --dataset mag240m --fanout "10,15,20" --hidden 32 --dropout 0.2 --model GAT --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1 --num-epoch 50 --debug



# igb-full
# python train_online.py --dataset igb-full --num-epoch 20 --hidden 256 --dropout 0 --fanout "10,15,20" --dir /nvme1n1/offgs_dataset --model SAGE --ratio 0.1
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0 --ratio 0.2 --model SAGE
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0 --ratio 0.1 --model SAGE
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0 --ratio 0.05 --model SAGE
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0 --ratio 0.01 --model SAGE

# python train_online.py --dataset igb-full --num-epoch 20 --hidden 32 --dropout 0 --fanout "10,15,20" --dir /nvme1n1/offgs_dataset --model GAT --ratio 0.1
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 128 --dropout 0 --ratio 0.2 --model GAT
# sudo env PATH=$PATH python train_offline.py --dataset igb-full --fanout "10,15,20" --hidden 32 --dropout 0 --model GAT --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1 --num-epoch 20 --debug
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 128 --dropout 0 --ratio 0.05 --model GAT
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 128 --dropout 0 --ratio 0.01 --model GAT



# ogbn-products
# python train_online.py --dataset ogbn-products --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --dir /nvme1n1/offgs_dataset --ratio 1.0 --device 1 --num-epoch 100
# python train_online.py --dataset ogbn-products --fanout "10,15,20" --hidden 16 --dropout 0.2 --model GAT --dir /nvme1n1/offgs_dataset --ratio 1.0 --device 1 --num-epoch 100
