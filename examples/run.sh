# python prepare_dataset.py --dataset=ogbn-products
# sudo env PATH=$PATH python sampling.py --dataset=ogbn-products --fanout="10,10,10"

# python feat_packing.py --dataset=ogbn-products --feat-cache-size=100000000
# python train.py --dataset=ogbn-products --feat-cache-size=100000000

# python preprocess.py --dataset=ogbn-products --feat-cache-size=200000000
# python train.py --dataset=ogbn-products --feat-cache-size=200000000

# python preprocess.py --dataset=ogbn-products --feat-cache-size=1000000000
# python train.py --dataset=ogbn-products --feat-cache-size=1000000000

# python preprocess.py --dataset=ogbn-papers100M --feat-cache-size=6000000000
# python train.py --dataset=ogbn-papers100M --feat-cache-size=6000000000


# sudo cgcreate -t $USER:$USER -a $USER:$USER  -g memory:0.3gb
# echo 300000000 > /sys/fs/cgroup/memory/0.3gb/memory.limit_in_bytes


# cgexec -g memory:3.1gb python feat_packing.py --dataset=ogbn-products --feat-cache-size=100000000
# cgexec -g memory:3.1gb python train.py --dataset=ogbn-products --feat-cache-size=100000000

# cgexec -g memory:0.3gb python feat_packing.py --dataset=ogbn-products --feat-cache-size=200000000
# cgexec -g memory:0.3gb python train.py --dataset=ogbn-products --feat-cache-size=200000000

# cgexec -g memory:0.5gb python feat_packing.py --dataset=ogbn-products --feat-cache-size=400000000
# cgexec -g memory:0.5gb python train.py --dataset=ogbn-products --feat-cache-size=400000000

# cgexec -g memory:0.7gb python feat_packing.py --dataset=ogbn-products --feat-cache-size=600000000
# cgexec -g memory:0.7gb python train.py --dataset=ogbn-products --feat-cache-size=600000000

# cgexec -g memory:0.9gb python feat_packing.py --dataset=ogbn-products --feat-cache-size=800000000
# cgexec -g memory:0.9gb python train.py --dataset=ogbn-products --feat-cache-size=800000000

# cgexec -g memory:1.1gb python feat_packing.py --dataset=ogbn-products --feat-cache-size=1000000000
# cgexec -g memory:1.1gb python train.py --dataset=ogbn-products --feat-cache-size=1000000000




# python prepare_dataset.py --dataset=ogbn-products
# sudo env PATH=$PATH python sampling.py --dataset=ogbn-products --fanout="10,10,10"

# sudo env PATH=$PATH python feat_packing.py --dataset=ogbn-products --feat-cache-size=200000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=0 --cpu-cache-size=200000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=100000000 --cpu-cache-size=100000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=200000000 --cpu-cache-size=0
# sudo env PATH=$PATH python train.py --dataset=ogbn-products --feat-cache-size=200000000

# sudo env PATH=$PATH python feat_packing.py --dataset=ogbn-products --feat-cache-size=600000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=0 --cpu-cache-size=600000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=300000000 --cpu-cache-size=300000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=600000000 --cpu-cache-size=0
# sudo env PATH=$PATH python train.py --dataset=ogbn-products --feat-cache-size=600000000





# python prepare_dataset.py --dataset=ogbn-papers100M
# sudo env PATH=$PATH python sampling.py --dataset=ogbn-papers100M --fanout="10,10,10"

# sudo env PATH=$PATH python feat_packing.py --dataset=ogbn-papers100M --feat-cache-size=10000000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=0 --cpu-cache-size=10000000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=5000000000 --cpu-cache-size=5000000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=8000000000 --cpu-cache-size=2000000000
# sudo env PATH=$PATH python train.py --dataset=ogbn-papers100M --feat-cache-size=10000000000

# sudo env PATH=$PATH python feat_packing.py --dataset=ogbn-papers100M --feat-cache-size=32000000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=0 --cpu-cache-size=32000000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=8000000000 --cpu-cache-size=24000000000
# sudo env PATH=$PATH python train.py --dataset=ogbn-papers100M --feat-cache-size=32000000000





# python prepare_dataset.py --dataset=friendster
# sudo env PATH=$PATH python sampling.py --dataset=friendster --fanout="10,10,10"

# sudo env PATH=$PATH python feat_packing.py --dataset=friendster --feat-cache-size=6400000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=0 --cpu-cache-size=6400000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=3200000000 --cpu-cache-size=3200000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=6400000000 --cpu-cache-size=0
# sudo env PATH=$PATH python train.py --dataset=friendster --feat-cache-size=6400000000

# sudo env PATH=$PATH python feat_packing.py --dataset=friendster --feat-cache-size=19200000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=0 --cpu-cache-size=19200000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=8000000000 --cpu-cache-size=11200000000
# sudo env PATH=$PATH python train.py --dataset=friendster --feat-cache-size=19200000000


# datasets=(ogbn-products ogbn-papers100M friendster)
# batchsizes=(2048 4096 8192)
# cachesizes_2=(200000000 10000000000 6400000000)
# cachesizes_6=(600000000 32000000000 19200000000)

datasets=(ogbn-products)
batchsizes=(2048)
cachesizes_2=(200000000)
cachesizes_6=(600000000)

for i in ${!datasets[@]}
do
for batchsize in ${batchsizes[@]}
do
# sudo env PATH=$PATH python mega_batch_sampling.py --dataset ${datasets[$i]} --batchsize ${batchsize} --fanout "10,10,10" --store-path /nvme1n1/offgs_dataset

# sudo env PATH=$PATH python feat_packing.py --dataset ${datasets[$i]} --batchsize ${batchsize} --store-path /nvme1n1/offgs_dataset --feat-cache-size ${cachesizes_2[$i]}
sudo env PATH=$PATH python runner.py --dataset ${datasets[$i]} --batchsize ${batchsize} --dir /nvme1n1/offgs_dataset --feat-cache-size ${cachesizes_2[$i]} --mega_batch

# sudo env PATH=$PATH python feat_packing.py --dataset ${datasets[$i]} --batchsize ${batchsize} --store-path /nvme1n1/offgs_dataset --feat-cache-size ${cachesizes_6[$i]}
# sudo env PATH=$PATH python runner.py --dataset ${datasets[$i]} --batchsize ${batchsize} --dir /nvme1n1/offgs_dataset --feat-cache-size ${cachesizes_6[$i]} --mega_batch
done
done
