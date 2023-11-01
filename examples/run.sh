# python prepare_dataset.py --dataset=ogbn-products
# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python sampling.py --dataset=ogbn-products --fanout="10,10,10"

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
# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python sampling.py --dataset=ogbn-products --fanout="10,10,10"

# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python feat_packing.py --dataset=ogbn-products --feat-cache-size=200000000
sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python train_single_thread.py --dataset=ogbn-products --feat-cache-size=200000000

# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python feat_packing.py --dataset=ogbn-products --feat-cache-size=600000000
sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python train_single_thread.py --dataset=ogbn-products --feat-cache-size=600000000





# python prepare_dataset.py --dataset=ogbn-papers100M
# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python sampling.py --dataset=ogbn-papers100M --fanout="10,10,10"

# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python feat_packing.py --dataset=ogbn-papers100M --feat-cache-size=10000000000
sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python train_single_thread.py --dataset=ogbn-papers100M --feat-cache-size=10000000000

# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python feat_packing.py --dataset=ogbn-papers100M --feat-cache-size=32000000000
sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python train_single_thread.py --dataset=ogbn-papers100M --feat-cache-size=32000000000





# python prepare_dataset.py --dataset=friendster
# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python sampling.py --dataset=friendster --fanout="10,10,10"

# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python feat_packing.py --dataset=friendster --feat-cache-size=6400000000
sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python train_single_thread.py --dataset=friendster --feat-cache-size=6400000000

# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python feat_packing.py --dataset=friendster --feat-cache-size=19200000000
sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python train_single_thread.py --dataset=friendster --feat-cache-size=19200000000
