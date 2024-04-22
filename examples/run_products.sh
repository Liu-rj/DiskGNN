## products
python prepare_dataset.py --dataset products --store-path /nvme1n1/offgs_dataset
sudo env PATH=$PATH python sampling.py --dataset ogbn-products --fanout "10,15,20" --store-path /nvme1n1/offgs_dataset --ratio 1.0
sudo env PATH=$PATH python feat_packing.py --dataset ogbn-products --fanout "10,15,20" --feat-cache-size 5e8 --store-path /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1
# acc
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-products --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e8 --cpu-cache-size 3e8 --dir /nvme1n1/offgs_dataset --ratio 1.0 --blowup -1 --device 1 --num-epoch 100 --debug