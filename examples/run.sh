# python prepare_dataset.py --dataset=ogbn-products
# sudo /opt/conda/envs/SSD_GNN/bin/python sampling.py --dataset=ogbn-products --fanout="10,10,10"

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




# python prepare_dataset.py --dataset=ogbn-products --store-path /nvme1n1/offgs_dataset
# sudo /opt/conda/envs/SSD_GNN/bin/python sampling.py --dataset=ogbn-products --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset

# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset=ogbn-products --feat-cache-size=200000000 --store-path /nvme1n1/offgs_dataset
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=0 --cpu-cache-size=200000000 --dir /nvme1n1/offgs_dataset
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=100000000 --cpu-cache-size=100000000 --dir /nvme1n1/offgs_dataset
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=200000000 --cpu-cache-size=0 --dir /nvme1n1/offgs_dataset
# sudo /opt/conda/envs/SSD_GNN/bin/python train.py --dataset=ogbn-products --feat-cache-size=200000000

# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset=ogbn-products --feat-cache-size=600000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=0 --cpu-cache-size=600000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=300000000 --cpu-cache-size=300000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=600000000 --cpu-cache-size=0
# sudo /opt/conda/envs/SSD_GNN/bin/python train.py --dataset=ogbn-products --feat-cache-size=600000000





# python prepare_dataset.py --dataset=ogbn-papers100M
# sudo /opt/conda/envs/SSD_GNN/bin/python sampling.py --dataset ogbn-papers100M --fanout="10,10,10" --store-path /nvme2n1/offgs_dataset --ratio 1.0

# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset ogbn-papers100M --feat-cache-size 10000000000 --store-path /nvme2n1/offgs_dataset --ratio 1.0
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=0 --cpu-cache-size=10000000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset ogbn-papers100M --gpu-cache-size 5000000000 --cpu-cache-size 5000000000 --dir /nvme2n1/offgs_dataset --ratio 1.0
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=8000000000 --cpu-cache-size=2000000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train.py --dataset=ogbn-papers100M --feat-cache-size=10000000000

# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset=ogbn-papers100M --feat-cache-size=32000000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=0 --cpu-cache-size=32000000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=8000000000 --cpu-cache-size=24000000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train.py --dataset=ogbn-papers100M --feat-cache-size=32000000000





# python prepare_dataset.py --dataset=friendster
# sudo /opt/conda/envs/SSD_GNN/bin/python sampling.py --dataset=friendster --fanout="10,10,10"

# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset=friendster --feat-cache-size=6400000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=friendster --gpu-cache-size=0 --cpu-cache-size=6400000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=friendster --gpu-cache-size=3200000000 --cpu-cache-size=3200000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=friendster --gpu-cache-size=6400000000 --cpu-cache-size=0
# sudo /opt/conda/envs/SSD_GNN/bin/python train.py --dataset=friendster --feat-cache-size=6400000000

# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset=friendster --feat-cache-size=19200000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=friendster --gpu-cache-size=0 --cpu-cache-size=19200000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset=friendster --gpu-cache-size=8000000000 --cpu-cache-size=11200000000
# sudo /opt/conda/envs/SSD_GNN/bin/python train.py --dataset=friendster --feat-cache-size=19200000000

sudo env PATH=$PATH python feat_packing.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 0 --segment-size -1
sudo env PATH=$PATH python feat_packing.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 4e6 --segment-size 200
sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=5e9 --cpu-cache-size=5e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 0 --segment-size -1
sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=5e9 --cpu-cache-size=5e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 4e6 --segment-size 200
# sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=3200000000 --cpu-cache-size=3200000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=6400000000 --cpu-cache-size=0
sudo env PATH=$PATH python train_multi_thread.py --dataset=friendster --gpu-cache-size=5e9 --cpu-cache-size=5e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 0 --segment-size -1
sudo env PATH=$PATH python train_multi_thread.py --dataset=friendster --gpu-cache-size=5e9 --cpu-cache-size=5e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 4e6 --segment-size 200
sudo env PATH=$PATH python train_multi_process.py --dataset=friendster --gpu-cache-size=5e9 --cpu-cache-size=5e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 0 --segment-size -1
sudo env PATH=$PATH python feat_deduplicate_disk_cache.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0
sudo env PATH=$PATH python feat_deduplicate_disk_cache_2.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0
# sudo env PATH=$PATH python feat_deduplicate_bucket_subg.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0
sudo env PATH=$PATH python test_duplicate.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 8e6 --segment-size 200



# IGB-FULL
# python prepare_dataset.py --dataset igb-full --dataset_size full --path /efs/rjliu/dataset/igb_full --store-path /nvme1n1/offgs_dataset



## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# sudo /opt/conda/envs/SSD_GNN/bin/python sampling.py --dataset igb-full --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset --ratio 0.2
# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset igb-full --feat-cache-size=7000000000 --store-path /nvme1n1/offgs_dataset --ratio 0.2
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset igb-full --gpu-cache-size=1000000000 --cpu-cache-size=6000000000 --dir /nvme1n1/offgs_dataset --ratio 0.2




# MAG240M
# python prepare_dataset.py --dataset mag240m --store-path /nvme1n1/offgs_dataset
# sudo /opt/conda/envs/SSD_GNN/bin/python sampling.py --dataset mag240m --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset --ratio 1 --use-artifitial-train
# sudo /opt/conda/envs/SSD_GNN/bin/python sampling.py --dataset mag240m --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset --ratio 2

# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset mag240m --feat-cache-size=30000000000 --store-path /nvme1n1/offgs_dataset --ratio 1
# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset mag240m --feat-cache-size=20000000000 --store-path /nvme1n1/offgs_dataset --ratio 1
# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset mag240m --feat-cache-size=10000000000 --store-path /nvme1n1/offgs_dataset --ratio 1
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset mag240m --gpu-cache-size=10000000000 --cpu-cache-size=20000000000 --dir /nvme1n1/offgs_dataset --ratio 1
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset mag240m --gpu-cache-size=10000000000 --cpu-cache-size=10000000000 --dir /nvme1n1/offgs_dataset --ratio 1
# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset mag240m --gpu-cache-size=10000000000 --cpu-cache-size=0 --dir /nvme1n1/offgs_dataset --ratio 1
# python runner_acc.py --dataset mag240m --num-epoch 100 --batchsize 1024 --dir /nvme1n1/offgs_dataset




# log_dir=logs/train_single_thread_decompose.csv
# datasets=(ogbn-products ogbn-papers100M friendster)
# batchsizes=(1024)
# cachesizes_2=(200000000 10000000000 6400000000)
# cachesizes_6=(600000000 32000000000 19200000000)

# for i in ${!datasets[@]}
# do
# for batchsize in ${batchsizes[@]}
# do
# # sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_sample.py --dataset ${datasets[$i]} --mega_batch_size ${batchsize} --batchsize 1024 --fanout "10,10,10" --store-path /nvme1n1/offgs_dataset

# # sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_packing.py --dataset ${datasets[$i]} --mega_batch_size ${batchsize} --batchsize 1024 --store-path /nvme1n1/offgs_dataset --feat-cache-size ${cachesizes_2[$i]}
# sudo /opt/conda/envs/SSD_GNN/bin/python runner.py --dataset ${datasets[$i]} --mega_batch_size ${batchsize} --batchsize 1024 --dir /nvme1n1/offgs_dataset --feat-cache-size ${cachesizes_2[$i]} --log ${log_dir}

# # sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_packing.py --dataset ${datasets[$i]} --mega_batch_size ${batchsize} --batchsize 1024 --store-path /nvme1n1/offgs_dataset --feat-cache-size ${cachesizes_6[$i]}
# sudo /opt/conda/envs/SSD_GNN/bin/python runner.py --dataset ${datasets[$i]} --mega_batch_size ${batchsize} --batchsize 1024 --dir /nvme1n1/offgs_dataset --feat-cache-size ${cachesizes_6[$i]} --log ${log_dir}
# done
# done


# log_dir=logs/merge_minibatch_train_single_thread_decompose.csv
# datasets=(ogbn-products ogbn-papers100M friendster)
# batchsizes=(4096 10240)
# cachesizes_2=(200000000 10000000000 6400000000)
# cachesizes_6=(600000000 32000000000 19200000000)

# datasets=(igb-full igb-full igb-full igb-full)
# mega_batch_size=(200000000 400000000 800000000 1600000000)
# feat_cache_size=(6800000000 6600000000 6200000000 5400000000)
# cpu_cache_size=(5800000000 5600000000 5200000000 4400000000)
# gpu_cache_size=(1000000000 1000000000 1000000000 1000000000)
# ratio=(0.2 0.2 0.2 0.2)

# # datasets=(ogbn-papers100M ogbn-papers100M ogbn-papers100M)
# # mega_batch_size=(100000000 1000000000 2000000000)
# # feat_cache_size=(9900000000 9000000000 8000000000)
# # cpu_cache_size=(4900000000 4000000000 3000000000)
# # gpu_cache_size=(5000000000 5000000000 5000000000)
# # ratio=(1.0 1.0 1.0)
# dir=/nvme1n1/offgs_dataset

# for i in ${!datasets[@]}
# do
# sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_sample.py --dataset ${datasets[$i]} --feat-cache-size ${feat_cache_size[$i]} --mega-batch-size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --store-path ${dir} --ratio ${ratio[$i]}
# sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_packing.py --dataset ${datasets[$i]} --feat-cache-size ${feat_cache_size[$i]} --mega_batch_size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --store-path ${dir} --ratio ${ratio[$i]}
# sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_train_single_thread.py --dataset ${datasets[$i]} --cpu-cache-size ${cpu_cache_size[$i]} --gpu-cache-size ${gpu_cache_size[$i]} --mega_batch_size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --dir ${dir} --ratio ${ratio[$i]}
# done


