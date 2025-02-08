data_path=/nvme2n1/offgs_dataset

############################## ShadowGNN ########################################
# Ogbn-papers100M
sudo env PATH=$PATH python sampling.py --dataset ogbn-papers100M --fanout "10,10,10" --store-path $data_path --ratio 1.0
sudo env PATH=$PATH python feat_packing.py --dataset ogbn-papers100M --fanout "10,10,10" --feat-cache-size 5e9 --store-path $data_path --ratio 1.0 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,10,10" --hidden 128 --dropout 0.2 --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1

# MAG240M
sudo env PATH=$PATH python sampling.py --dataset mag240m --fanout "10,10,10" --store-path $data_path --ratio 1.0
sudo env PATH=$PATH python feat_packing.py --dataset mag240m --fanout "10,10,10" --feat-cache-size 10e9 --store-path $data_path --ratio 1.0 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,10,10" --hidden 128 --dropout 0.2 --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1
