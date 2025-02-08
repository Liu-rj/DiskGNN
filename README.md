# DiskGNN: Bridging I/O Efficiency and Model Accuracy for Out-of-Core GNN Training

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

The following example runs DiskGNN on the GraphSAGE model and Ogbn-papers100M dataset.

- cd into the example directory (from the root directory of this repo):

```shell
cd example
```

* Prepare dataset:

```shell
python prepare_dataset.py --dataset ogbn-papers100M
```

* Offline sampling:

```shell
sudo env PATH=$PATH python sampling.py --dataset ogbn-papers100M --fanout "10,15,20" --store-path /nvme2n1/offgs_dataset --ratio 1.0
```

* Cold feature packing with smart search method:

```shell
sudo env PATH=$PATH python feat_packing.py --dataset ogbn-papers100M --fanout "10,15,20" --feat-cache-size 5e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
```

* Online training efficiency evaluation:

```shell
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
```

- Online training accuracy evaluation:

```shell
sudo env PATH=$PATH python train_multi_thread.py --dataset ogbn-papers100M --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 2e9 --cpu-cache-size 3e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1 --device 1 --num-epoch 50 --debug
```
