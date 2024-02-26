# python prepare_dataset.py --dataset=ogbn-products --store-path /nvme1n1/offgs_dataset
# sudo env PATH=$PATH python sampling.py --dataset=ogbn-products --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset

# sudo env PATH=$PATH python feat_packing.py --dataset=ogbn-products --feat-cache-size=200000000 --store-path /nvme1n1/offgs_dataset
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=0 --cpu-cache-size=200000000 --dir /nvme1n1/offgs_dataset
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=100000000 --cpu-cache-size=100000000 --dir /nvme1n1/offgs_dataset
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-products --gpu-cache-size=200000000 --cpu-cache-size=0 --dir /nvme1n1/offgs_dataset
# sudo env PATH=$PATH python train.py --dataset=ogbn-products --feat-cache-size=200000000





# python prepare_dataset.py --dataset=ogbn-papers100M
# sudo env PATH=$PATH python sampling.py --dataset ogbn-papers100M --fanout="10,10,10" --store-path /nvme2n1/offgs_dataset --ratio 1.0

# sudo env PATH=$PATH python feat_packing.py --dataset ogbn-papers100M --feat-cache-size 10000000000 --store-path /nvme2n1/offgs_dataset --ratio 1.0
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=0 --cpu-cache-size=10000000000
# sudo env PATH=$PATH python train_single_thread.py --dataset ogbn-papers100M --gpu-cache-size 5000000000 --cpu-cache-size 5000000000 --dir /nvme2n1/offgs_dataset --ratio 1.0
# sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-papers100M --gpu-cache-size=8000000000 --cpu-cache-size=2000000000
# sudo env PATH=$PATH python train.py --dataset=ogbn-papers100M --feat-cache-size=10000000000





# python prepare_dataset.py --dataset=friendster --store-path /nvme1n1/offgs_dataset
# sudo env PATH=$PATH python sampling.py --dataset=friendster --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset --ratio 1.0

sudo env PATH=$PATH python feat_packing.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 0 --segment-size -1
sudo env PATH=$PATH python feat_packing.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 4e6 --segment-size 200
sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=5e9 --cpu-cache-size=5e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 0 --segment-size -1
sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=5e9 --cpu-cache-size=5e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 4e6 --segment-size 200
# sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=3200000000 --cpu-cache-size=3200000000
# sudo env PATH=$PATH python train_single_thread.py --dataset=friendster --gpu-cache-size=6400000000 --cpu-cache-size=0
# sudo env PATH=$PATH python train.py --dataset=friendster --feat-cache-size=6400000000
sudo env PATH=$PATH python feat_deduplicate_disk_cache.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0
sudo env PATH=$PATH python feat_deduplicate_disk_cache_2.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0
# sudo env PATH=$PATH python feat_deduplicate_bucket_subg.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0
sudo env PATH=$PATH python test_duplicate.py --dataset friendster --feat-cache-size 1e10 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --disk-cache-num 8e6 --segment-size 200



# IGB-FULL
# python prepare_dataset.py --dataset igb-full --dataset_size full --path /efs/rjliu/dataset/igb_full --store-path /nvme1n1/offgs_dataset
# sudo env PATH=$PATH python sampling.py --dataset igb-full --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset --ratio 0.2
# sudo env PATH=$PATH python sampling.py --dataset igb-full --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset --ratio 0.1
# sudo env PATH=$PATH python sampling.py --dataset igb-full --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset --ratio 0.05
# sudo env PATH=$PATH python sampling.py --dataset igb-full --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset --ratio 0.01

sudo env PATH=$PATH python feat_packing.py --dataset igb-full --feat-cache-size 1e10 --store-path /nvme1n1/offgs_dataset --ratio 0.2
# sudo env PATH=$PATH python feat_packing.py --dataset igb-full --feat-cache-size=20000000000 --store-path /nvme1n1/offgs_dataset --ratio 0.2
sudo env PATH=$PATH python feat_packing.py --dataset igb-full --feat-cache-size 3e10 --store-path /nvme1n1/offgs_dataset --ratio 0.2
# sudo env PATH=$PATH python feat_packing.py --dataset igb-full --feat-cache-size=45800000000 --store-path /nvme1n1/offgs_dataset --ratio 0.1
# sudo env PATH=$PATH python feat_packing.py --dataset igb-full --feat-cache-size=45800000000 --store-path /nvme1n1/offgs_dataset --ratio 0.05
# sudo env PATH=$PATH python feat_packing.py --dataset igb-full --feat-cache-size=45800000000 --store-path /nvme1n1/offgs_dataset --ratio 0.05
# sudo env PATH=$PATH python train_single_thread.py --dataset igb-full --gpu-cache-size=10000000000 --cpu-cache-size=20000000000 --dir /nvme1n1/offgs_dataset --ratio 0.2
# python runner_igbfull.py --dataset igb-full --num-epoch 10 --batchsize 1024 --dir /nvme1n1/offgs_dataset
sudo env PATH=$PATH python feat_deduplicate_disk_cache.py --dataset igb-full --feat-cache-size 3e10 --store-path /nvme1n1/offgs_dataset --ratio 0.2
sudo env PATH=$PATH python feat_deduplicate_disk_cache.py --dataset igb-full --feat-cache-size 1e10 --store-path /nvme1n1/offgs_dataset --ratio 0.2
sudo env PATH=$PATH python feat_deduplicate_disk_cache_2.py --dataset igb-full --feat-cache-size 3e10 --store-path /nvme1n1/offgs_dataset --ratio 0.2
sudo env PATH=$PATH python feat_deduplicate_disk_cache_2.py --dataset igb-full --feat-cache-size 1e10 --store-path /nvme1n1/offgs_dataset --ratio 0.2
# sudo env PATH=$PATH python feat_deduplicate_disk_cache.py --dataset igb-full --feat-cache-size 0 --store-path /nvme1n1/offgs_dataset --ratio 0.2
# sudo env PATH=$PATH python feat_deduplicate_bucket_subg.py --dataset igb-full --feat-cache-size 30000000000 --store-path /nvme1n1/offgs_dataset --ratio 0.2
sudo env PATH=$PATH python test_duplicate.py --dataset igb-full --feat-cache-size 3e10 --store-path /nvme1n1/offgs_dataset --ratio 0.2 --disk-cache-num 8e6 --segment-size 1000
sudo env PATH=$PATH python test_duplicate.py --dataset igb-full --feat-cache-size 1e10 --store-path /nvme1n1/offgs_dataset --ratio 0.2 --disk-cache-num 1e7 --segment-size 200


# MAG240M
# python prepare_dataset.py --dataset mag240m --store-path /nvme1n1/offgs_dataset
# sudo env PATH=$PATH python sampling.py --dataset mag240m --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset --ratio 1.0
# sudo env PATH=$PATH python sampling.py --dataset mag240m --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset --ratio 2.0

# sudo env PATH=$PATH python feat_packing.py --dataset mag240m --feat-cache-size=30000000000 --store-path /nvme1n1/offgs_dataset --ratio 1
# sudo env PATH=$PATH python feat_packing.py --dataset mag240m --feat-cache-size=20000000000 --store-path /nvme1n1/offgs_dataset --ratio 1
# sudo env PATH=$PATH python feat_packing.py --dataset mag240m --feat-cache-size=10000000000 --store-path /nvme1n1/offgs_dataset --ratio 1
# sudo env PATH=$PATH python train_single_thread.py --dataset mag240m --gpu-cache-size=10000000000 --cpu-cache-size=20000000000 --dir /nvme1n1/offgs_dataset --ratio 1
# sudo env PATH=$PATH python train_single_thread.py --dataset mag240m --gpu-cache-size=10000000000 --cpu-cache-size=10000000000 --dir /nvme1n1/offgs_dataset --ratio 1
# sudo env PATH=$PATH python train_single_thread.py --dataset mag240m --gpu-cache-size=10000000000 --cpu-cache-size=0 --dir /nvme1n1/offgs_dataset --ratio 1
# python runner_acc.py --dataset mag240m --num-epoch 100 --batchsize 1024 --dir /nvme1n1/offgs_dataset



# log_dir=logs/merge_minibatch_train_single_thread_decompose.csv
# datasets=(ogbn-products ogbn-papers100M friendster)
# batchsizes=(4096 10240)
# cachesizes_2=(200000000 10000000000 6400000000)
# cachesizes_6=(600000000 32000000000 19200000000)

# datasets=(igb-full igb-full igb-full igb-full igb-full igb-full)
# mega_batch_size=(100000000 500000000 1000000000 2000000000 5000000000 10000000000)
# feat_cache_size=(29900000000 29500000000 29000000000 28000000000 25000000000 20000000000)
# cpu_cache_size=(19900000000 19500000000 19000000000 18000000000 15000000000 10000000000)
# gpu_cache_size=(10000000000 10000000000 10000000000 10000000000 10000000000 10000000000)
# ratio=(0.2 0.2 0.2 0.2 0.2 0.2)
# sample_dirs=(/nvme1n1/offgs_dataset /nvme1n1/offgs_dataset /nvme1n1/offgs_dataset /nvme1n1/offgs_dataset /nvme2n1/offgs_dataset /nvme2n1/offgs_dataset)
# dir=/nvme1n1/offgs_dataset

# datasets=(ogbn-papers100M ogbn-papers100M ogbn-papers100M)
# datasets=(friendster friendster friendster)
# mega_batch_size=(100000000 1000000000 2000000000)
# feat_cache_size=(9900000000 9000000000 8000000000)
# cpu_cache_size=(4900000000 4000000000 3000000000)
# gpu_cache_size=(5000000000 5000000000 5000000000)
# ratio=(1.0 1.0 1.0)
# sample_dirs=(/nvme1n1/offgs_dataset /nvme1n1/offgs_dataset /nvme1n1/offgs_dataset)
# dir=/nvme1n1/offgs_dataset

# for i in ${!datasets[@]}
# do
# # sudo env PATH=$PATH python merge_minibatch_sample.py --dataset ${datasets[$i]} --feat-cache-size ${feat_cache_size[$i]} --mega-batch-size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --store-path ${dir} --ratio ${ratio[$i]}
# # sudo env PATH=$PATH python merge_minibatch_packing.py --dataset ${datasets[$i]} --feat-cache-size ${feat_cache_size[$i]} --mega_batch_size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --store-path ${dir} --ratio ${ratio[$i]}
# sudo env PATH=$PATH python merge_minibatch_train_single_thread.py --dataset ${datasets[$i]} --cpu-cache-size ${cpu_cache_size[$i]} --gpu-cache-size ${gpu_cache_size[$i]} --mega_batch_size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --dir ${dir} --ratio ${ratio[$i]}
# done

# # sudo env PATH=$PATH python merge_minibatch_train_single_thread.py --dataset igb-full --cpu-cache-size 15000000000 --gpu-cache-size 10000000000 --mega_batch_size 5000000000 --batchsize 1024 --fanout "10,10,10" --dir /nvme1n1/offgs_dataset --ratio 0.2 --sample-dir /nvme2n1/offgs_dataset
# sudo env PATH=$PATH python merge_minibatch_train_single_thread.py --dataset ogbn-papers100M --cpu-cache-size 4900000000 --gpu-cache-size 5000000000 --mega_batch_size 100000000 --batchsize 1024 --fanout "10,10,10" --dir /nvme2n1/offgs_dataset --ratio 1.0 --debug
