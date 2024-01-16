# # papers100M
# python train_online.py --dataset ogbn-papers100M --num-epoch 100 --hidden 512 --dropout 0.2
# python train_offline.py --dataset ogbn-papers100M --num-epoch 100 --hidden 512 --dropout 0.2

# # mag240m
# python train_online.py --dataset mag240m --num-epoch 100
# python train_offline.py --dataset mag240m --num-epoch 100

# igb-full
# python train_online.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0.2
python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0.2 --ratio 0.2
python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0.2 --ratio 0.1
python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0.2 --ratio 0.05
python train_offline.py --dataset igb-full --num-epoch 5 --hidden 512 --dropout 0.2 --ratio 0.01
