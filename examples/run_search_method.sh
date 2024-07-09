# Friendster
python search_method.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup 3.0 --search-step 100
python search_method.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0 --blowup 5.0 --serach-step 100

# IGB-HOM
python search_method.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1 --blowup 3.0 --search-step 1000
python search_method.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1 --blowup 5.0 --search-step 1000
python search_method.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1 --blowup 7.0 --search-step 1000
