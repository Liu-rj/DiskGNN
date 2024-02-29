# OfflineSampling for Out-of-core GNN Training

## Installation

* Install [liburing](https://github.com/axboe/liburing).

* Build and install (from the root directory of this repo)

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
