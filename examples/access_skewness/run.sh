data_path=/data/offgs_dataset

python skewness.py --dataset friendster --fanout "10,15,20" --store-path $data_path --ratio 1.0

python skewness.py --dataset ogbn-papers100M --fanout "10,15,20" --store-path $data_path --ratio 1.0

python skewness.py --dataset mag240m --fanout "10,15,20" --store-path $data_path --ratio 1.0

python skewness.py --dataset igb-full --fanout "10,15,20" --store-path $data_path --ratio 0.1
