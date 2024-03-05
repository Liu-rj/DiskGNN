# OfflineSampling for Out-of-core GNN Training

## Installation

* From the root directory of this project

```
mkdir build; cd build; cmake ..; make -j8
```

* From the root directory of this project

```
cd python; python setup.py install
```

## Usage

* Prepare dataset

```shell
python prepare_dataset.py --dataset=ogbn-products
```

* Offline sampling

```shell
sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python sampling.py --dataset=ogbn-products --fanout="10,10,10"
```

* Cold feature packing

```shell
sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python feat_packing.py --dataset=ogbn-products --feat-cache-size=200000000
```

* Online training

```shell
sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages /opt/conda/envs/npc/bin/python train_single_thread.py --dataset=ogbn-products --feat-cache-size=200000000
```

### for accelerate preprocessing
previous steps are the same

* Cold feature packing

```shell
python /home/ubuntu/OfflineSampling/examples/feat_map_reduce_packing.py
```

* Online training

```shell
python /home/ubuntu/OfflineSampling/examples/train_single_mapreduce.py
```