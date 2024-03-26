# # papers100M
# python train_online.py --dataset ogbn-papers100M --num-epoch 50 --hidden 256 --dropout 0.2 --dir /nvme2n1/offgs_dataset --model SAGE
python train_online.py --dataset ogbn-papers100M --num-epoch 50 --hidden 256 --dropout 0.2 --fanout "10,15,20" --dir /nvme2n1/offgs_dataset --model SAGE
# python train_offline.py --dataset ogbn-papers100M --num-epoch 100 --hidden 256 --dropout 0 --dir /nvme2n1/offgs_dataset --ratio 1.0 --model SAGE

python train_online.py --dataset ogbn-papers100M --num-epoch 100 --hidden 128 --dropout 0.2 --dir /nvme2n1/offgs_dataset --model GAT --device 1
python train_offline.py --dataset ogbn-papers100M --num-epoch 100 --hidden 128 --dropout 0.2 --dir /nvme2n1/offgs_dataset --ratio 1.0 --model GAT --device 1

# # mag240m
python train_online.py --dataset mag240m --num-epoch 50 --hidden 256 --dropout 0.2 --fanout "10,15,20" --dir /nvme1n1/offgs_dataset --model SAGE
# python train_offline.py --dataset mag240m --num-epoch 100 --model SAGE

python train_online.py --dataset mag240m --num-epoch 100 --hidden 128 --dropout 0.2 --model GAT --device 1
python train_offline.py --dataset mag240m --num-epoch 100 --hidden 128 --dropout 0.2 --ratio 1.0 --model GAT --device 1

# igb-full
python train_online.py --dataset igb-full --num-epoch 5 --hidden 256 --dropout 0 --fanout "10,15,20" --dir /nvme1n1/offgs_dataset --model SAGE
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0.2 --ratio 0.2 --model SAGE
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0.2 --ratio 0.1 --model SAGE
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0.2 --ratio 0.05 --model SAGE
# python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0.2 --ratio 0.01 --model SAGE

python train_online.py --dataset igb-full --num-epoch 5 --hidden 128 --dropout 0.2 --model GAT --device 1
python train_offline.py --dataset igb-full --num-epoch 5 --hidden 128 --dropout 0.2 --ratio 0.2 --model GAT --device 1
python train_offline.py --dataset igb-full --num-epoch 5 --hidden 128 --dropout 0.2 --ratio 0.1 --model GAT --device 1
python train_offline.py --dataset igb-full --num-epoch 5 --hidden 128 --dropout 0.2 --ratio 0.05 --model GAT --device 1
python train_offline.py --dataset igb-full --num-epoch 5 --hidden 128 --dropout 0.2 --ratio 0.01 --model GAT --device 1
