# cache mem
python cachemem_blowup.py --dataset friendster --fanout "10,15,20" --store-path /nvme2n1/offgs_dataset --ratio 1.0
python cachemem_blowup.py --dataset igb-full --fanout "10,15,20" --store-path /nvme3n1/offgs_dataset --ratio 0.1

# fanout
python fanout_blowup.py --dataset friendster --feat-cache-size 4e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0
python fanout_blowup.py --dataset igb-full --feat-cache-size 15e9 --store-path /nvme3n1/offgs_dataset --ratio 0.1

# batch size
python batchsize_blowup.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0
python batchsize_blowup.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme3n1/offgs_dataset --ratio 0.1
