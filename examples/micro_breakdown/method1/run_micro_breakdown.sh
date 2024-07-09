# data_path=/nvme2n1/offgs_dataset

# Friendster
# sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 0 --store-path $data_path --ratio 1.0 --blowup -1
# sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 2e9 --store-path $data_path --ratio 1.0 --blowup -1
# sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path $data_path --ratio 1.0 --blowup -1
# sudo env PATH=$PATH python feat_packing.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path $data_path --ratio 1.0 --blowup 5.0

# sudo env PATH=$PATH python train_multi_thread_plain.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir $data_path --ratio 1.0 --blowup -1
# sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 0 --cpu-cache-size 0 --dir $data_path --ratio 1.0 --blowup -1
# sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 0 --dir $data_path --ratio 1.0 --blowup -1
# sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir $data_path --ratio 1.0 --blowup -1
# sudo env PATH=$PATH python train_multi_thread.py --dataset friendster --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir $data_path --ratio 1.0 --blowup 5.0


data_path=/nvme3n1/offgs_dataset

# IGB-FULL
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 0 --store-path $data_path --ratio 0.1 --blowup -1
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 5e9 --store-path $data_path --ratio 0.1 --blowup -1
# sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path $data_path --ratio 0.1 --blowup -1
# sudo env PATH=$PATH python feat_packing.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path $data_path --ratio 0.1 --blowup 5.0

sudo env PATH=$PATH python train_multi_thread_plain.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir $data_path --ratio 0.1 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 0 --cpu-cache-size 0 --dir $data_path --ratio 0.1 --blowup -1
sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 0 --dir $data_path --ratio 0.1 --blowup -1
# sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir $data_path --ratio 0.1 --blowup -1
# sudo env PATH=$PATH python train_multi_thread.py --dataset igb-full --fanout "10,15,20" --hidden 256 --dropout 0 --model SAGE --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir $data_path --ratio 0.1 --blowup 5.0
