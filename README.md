# OfflineSampling for Out-of-core GNN Training

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

* Prepare dataset

```shell
python prepare_dataset.py --dataset=ogbn-products
```

* Offline sampling

```shell
sudo env PATH=$PATH python sampling.py --dataset=ogbn-products --fanout="10,10,10"
```

* Cold feature packing

```shell
sudo env PATH=$PATH python feat_packing.py --dataset=ogbn-products --feat-cache-size=2e8
```

* Online training

```shell
sudo env PATH=$PATH python train_single_thread.py --dataset=ogbn-products --feat-cache-size=2e8
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