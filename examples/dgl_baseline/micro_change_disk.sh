data_path=/data/offgs_dataset

# Ogbn-papers100M
sudo env PATH=$PATH python dgl_on_disk.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --dir $data_path --ratio 1.0 --num-epoch 1


# MAG240M
sudo env PATH=$PATH python dgl_on_disk.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --dir $data_path --ratio 1.0 --num-epoch 1


# Friendster
sudo env PATH=$PATH python dgl_on_disk.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --dir $data_path --ratio 1.0 --num-epoch 1


# IGB-FULL
sudo env PATH=$PATH python dgl_on_disk.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --dir $data_path --ratio 0.1 --num-epoch 1