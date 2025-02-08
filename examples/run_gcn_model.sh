data_path=/nvme2n1/offgs_dataset

############################## GCN ########################################
# Ogbn-papers100M
python prepare_dataset.py --dataset=ogbn-papers100M --store-path $data_path
sudo env PATH=$PATH python sampling.py --dataset ogbn-papers100M --fanout "10,15,20" --store-path $data_path --ratio 1.0
sudo env PATH=$PATH python feat_packing.py --dataset ogbn-papers100M --fanout "10,15,20" --feat-cache-size 5e9 --store-path $data_path --ratio 1.0 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model GCN --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1

# MAG240M
python prepare_dataset.py --dataset mag240m --store-path $data_path
sudo env PATH=$PATH python sampling.py --dataset mag240m --fanout "10,15,20" --store-path $data_path --ratio 1.0
sudo env PATH=$PATH python feat_packing.py --dataset mag240m --fanout "10,15,20" --feat-cache-size 10e9 --store-path $data_path --ratio 1.0 --blowup -1
# speed
sudo env PATH=$PATH python train_multi_thread.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model GCN --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1
