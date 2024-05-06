# disk cache motivation
# python feat_deduplicate_disk_cache.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path /nvme1n1/offgs_dataset --ratio 1.0

# disk cache effectiveness
python feat_deduplicate_disk_cache.py --dataset friendster --fanout "10,15,20" --feat-cache-size 4e9 --store-path /nvme2n1/offgs_dataset --ratio 1.0
python feat_deduplicate_disk_cache.py --dataset igb-full --fanout "10,15,20" --feat-cache-size 15e9 --store-path /nvme1n1/offgs_dataset --ratio 0.1
