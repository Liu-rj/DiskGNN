# DiskGNN for Out-of-core GNN Training

## Installation

* Create a new env with python 3.9.18 and MKL 2023.2.0:

```shell
conda create -n pytorch2 python=3.9.18 mkl=2023.2.0
```

* Install Pytorch 2.0.1 with CUDA 11.7:

```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

* Install DGL with CUDA 11.7:

```shell
conda install dgl=1.1.2 -c dglteam/label/cu117
```

* Install PyG with CUDA 11.7:

```shell
conda install pyg=2.5.0 -c pyg

# additional packages
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

* Install [liburing](https://github.com/axboe/liburing).

* Build and install (from the root directory of this repo):

```shell
# build
mkdir build; cd build; cmake ..; make -j8
# install
cd ../python; python setup.py install
```

## Usage

### Main result

illustrate one dataset and graph model using graphsage and papers100M

- cd in the example pth

```
cd example
```

* Prepare dataset

```shell
python prepare_dataset.py --dataset=ogbn-products
```

* Offline sampling

```shell
sudo env PATH=$PATH python sampling.py --dataset ogbn-papers100M --fanout "10,15,20" --store-path /nvme2n1/offgs_dataset --ratio 1.0
```

* Cold feature packing/ data layout preprocessing

```shell
sudo env PATH=$PATH python feat_packing.py --dataset ogbn-papers100M --fanout "10,15,20" --feat-cache-size 5e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
```

* Online training efficiency test

```shell
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
```

- Online training accuracy test

```shell
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1 --device 1 --num-epoch 50 --debug
```

**the full experiment script is in examples/run.sh**

### Batched Packing

sample and dataset prepare these previous steps are the same

you should select a dataset and then do the following compare

* Cold feature packing using batched packing

```shell
python /home/ubuntu/OfflineSampling/examples/feat_map_reduce_packing.py
```

* Cold feature packing using individual packing

```
python /home/ubuntu/OfflineSampling/examples/feat_packing.py
```

* Online training to demonstrate the batched packing does not bring overhead to online training

```shell
python /home/ubuntu/OfflineSampling/examples/train_single_mapreduce.py
```

### Test blowup

```
bash run_blowup.sh
```

### Marius Baseline

install following the marius github

take one dataset for example

#### preprocessing dataset and profile time

```
python marius_preprocess_mag.py
```

#### online training of marius

```
marius_train datasets/my_mag240m_8192/marius_gs_acc.yaml
```

you can change the config in the .yaml file