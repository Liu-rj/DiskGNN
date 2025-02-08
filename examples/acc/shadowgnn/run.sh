data_path=/nvme2n1/offgs_dataset

# papers100M
python train_online.py --dataset ogbn-papers100M --num-epoch 50 --hidden 128 --dropout 0.2 --fanout "10,10,10" --dir $data_path --model SAGE
sudo env PATH=$PATH python train_offline.py --dataset ogbn-papers100M --fanout "10,10,10" --hidden 128 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1 --num-epoch 50 --debug



# mag240m
python train_online.py --dataset mag240m --num-epoch 50 --hidden 128 --dropout 0.2 --fanout "10,10,10" --dir $data_path --model SAGE
sudo env PATH=$PATH python train_offline.py --dataset mag240m --fanout "10,10,10" --hidden 128 --dropout 0.2 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1 --num-epoch 50 --debug
