# OfflineSampling for Out-of-core GNN Training

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
