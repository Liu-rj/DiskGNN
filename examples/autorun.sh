
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 2048
# # sudo /opt/conda/envs/pytorch/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 4096
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 6144
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 8192
# sudo /opt/conda/envs/pytorch/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 10240
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 20480
# sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/merge_minibatch_sample.py --mega_batch_size 40960

sudo /opt/conda/envs/SSD_GNN/bin/python  /home/ubuntu/OfflineSampling/examples/prepare_dataset.py --dataset igb-full --dataset_size full --path /efs/rjliu/dataset/igb_full --store-path /nvme1n1/offgs_dataset
sudo /opt/conda/envs/SSD_GNN/bin/python /home/ubuntu/OfflineSampling/examples/prepare_dataset.py --dataset mag240m --store-path /nvme1n1/offgs_dataset