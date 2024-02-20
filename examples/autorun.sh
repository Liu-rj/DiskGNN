
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 2048
# # sudo /opt/conda/envs/pytorch/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 4096
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 6144
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 8192
# sudo /opt/conda/envs/pytorch/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 10240
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 20480
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 40960

# sudo /opt/conda/envs/SSD_GNN/bin/python  /home/ubuntu/OfflineSampling/examples/prepare_dataset.py --dataset igb-full --dataset_size full --path /efs/rjliu/dataset/igb_full --store-path /nvme1n1/offgs_dataset
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/prepare_dataset.py --dataset mag240m --store-path /nvme1n1/offgs_dataset


# sudo /opt/conda/envs/SSD_GNN/bin/python sampling.py --dataset friendster --fanout="10,10,10" --store-path /nvme1n1/offgs_dataset 
# sudo /opt/conda/envs/SSD_GNN/bin/python feat_packing.py --dataset friendster --feat-cache-size=3000000000 --store-path /nvme1n1/offgs_dataset 



























# datasets=(friendster friendster friendster friendster)
# mega_batch_size=(200000000 400000000 800000000 1600000000)
# feat_cache_size=(2800000000 2600000000 2200000000 1400000000)
# cpu_cache_size=(2300000000 2100000000 1700000000 900000000)
# gpu_cache_size=(500000000 500000000 500000000 500000000)
# ratio=(1.0 1.0 1.0 1.0)


# dir=/nvme1n1/offgs_dataset

# for i in ${!datasets[@]}
# do
# dataset=${datasets[$i]}
# mega_batch=${mega_batch_size[$i]}
# feat_cache=${feat_cache_size[$i]}
# cpu_cache=${cpu_cache_size[$i]}
# gpu_cache=${gpu_cache_size[$i]}
# batch_ratio=${ratio[$i]}
# sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_sample.py --dataset ${datasets[$i]} --feat-cache-size ${feat_cache_size[$i]} --mega-batch-size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --store-path ${dir} --ratio ${ratio[$i]}
# sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_packing.py --dataset ${datasets[$i]} --feat-cache-size ${feat_cache_size[$i]} --mega_batch_size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --store-path ${dir} --ratio ${ratio[$i]}
# sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_train_single_thread.py --dataset ${datasets[$i]} --cpu-cache-size ${cpu_cache_size[$i]} --gpu-cache-size ${gpu_cache_size[$i]} --mega_batch_size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --dir ${dir} --ratio ${ratio[$i]}
# sudo rm -rf "${dir}/${dataset}-1024-10,10,10-${batch_ratio}/cache-size-${feat_cache}-mega-batch-size-${mega_batch}"

# done

# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset friendster --gpu-cache-size=500000000 --cpu-cache-size=2500000000 --dir /nvme1n1/offgs_dataset 

datasets=(friendster friendster friendster friendster)
mega_batch_size=(200000000 400000000 800000000 1600000000)
feat_cache_size=(2800000000 2600000000 2200000000 1400000000)
cpu_cache_size=(2300000000 2100000000 1700000000 900000000)
gpu_cache_size=(500000000 500000000 500000000 500000000)
ratio=(1.0 1.0 1.0 1.0)

dir=/nvme1n1/offgs_dataset

for i in ${!datasets[@]}
do
dataset=${datasets[$i]}
mega_batch=${mega_batch_size[$i]}
feat_cache=${feat_cache_size[$i]}
cpu_cache=${cpu_cache_size[$i]}
gpu_cache=${gpu_cache_size[$i]}
batch_ratio=${ratio[$i]}

sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_sample.py --dataset ${datasets[$i]} --feat-cache-size ${feat_cache_size[$i]} --mega-batch-size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --store-path ${dir} --ratio ${ratio[$i]}
sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_packing.py --dataset ${datasets[$i]} --feat-cache-size ${feat_cache_size[$i]} --mega_batch_size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --store-path ${dir} --ratio ${ratio[$i]}
sudo /opt/conda/envs/SSD_GNN/bin/python merge_minibatch_train_single_thread.py --dataset ${datasets[$i]} --cpu-cache-size ${cpu_cache_size[$i]} --gpu-cache-size ${gpu_cache_size[$i]} --mega_batch_size ${mega_batch_size[$i]} --batchsize 1024 --fanout "10,10,10" --dir ${dir} --ratio ${ratio[$i]}
sudo rm -rf "${dir}/${dataset}-1024-10,10,10-${batch_ratio}/cache-size-${feat_cache}-mega-batch-size-${mega_batch}"

done

# sudo /opt/conda/envs/SSD_GNN/bin/python train_single_thread.py --dataset friendster --gpu-cache-size=500000000 --cpu-cache-size=2500000000 --dir /nvme1n1/offgs_dataset 
