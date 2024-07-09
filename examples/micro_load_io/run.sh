# Friendster
python run_single_thread.py --dataset friendster --fanout "10,15,20" --gpu-cache-size 2e9 --cpu-cache-size 2e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
python run_single_thread.py --dataset friendster --fanout "10,15,20" --gpu-cache-size 4e9 --cpu-cache-size 6e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1
python run_single_thread.py --dataset friendster --fanout "10,15,20" --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme2n1/offgs_dataset --ratio 1.0 --blowup -1


# IGB-HOM
python run_single_thread.py --dataset igb-full --fanout "10,15,20" --gpu-cache-size 5e9 --cpu-cache-size 10e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1
python run_single_thread.py --dataset igb-full --fanout "10,15,20" --gpu-cache-size 15e9 --cpu-cache-size 30e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1
python run_single_thread.py --dataset igb-full --fanout "10,15,20" --gpu-cache-size 15e9 --cpu-cache-size 60e9 --dir /nvme1n1/offgs_dataset --ratio 0.1 --blowup -1
