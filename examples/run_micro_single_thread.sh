data_path=/nvme2n1/offgs_dataset

# Ogbn-papers100M
sudo env PATH=$PATH python train_single_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir $data_path --ratio 1.0 --blowup -1



# MAG240M
sudo env PATH=$PATH python train_single_thread.py --dataset mag240m --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir $data_path --ratio 1.0 --blowup -1



# Friendster
sudo env PATH=$PATH python train_single_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir $data_path --ratio 1.0 --blowup -1



# IGB-FULL
sudo env PATH=$PATH python train_single_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme3n1/offgs_dataset --ratio 0.1 --blowup -1
