# python graphsage.py --num-batch=2000 --dataset=ogbn-products --model=SAGE
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=10 --dataset=ogbn-products --model=SAGE
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=50 --dataset=ogbn-products --model=SAGE
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=100 --dataset=ogbn-products --model=SAGE
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=200 --dataset=ogbn-products --model=SAGE
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=500 --dataset=ogbn-products --model=SAGE


# python graphsage.py --num-batch=2000 --dataset=ogbn-products --model=GAT
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=10 --dataset=ogbn-products --model=GAT
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=50 --dataset=ogbn-products --model=GAT
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=100 --dataset=ogbn-products --model=GAT
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=200 --dataset=ogbn-products --model=GAT
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=500 --dataset=ogbn-products --model=GAT



# python graphsage_offlinesample.py --num-batch=2000 --num-sample=10 --dataset=reddit --model=SAGE
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=50 --dataset=reddit --model=SAGE
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=100 --dataset=reddit --model=SAGE
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=200 --dataset=reddit --model=SAGE
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=500 --dataset=reddit --model=SAGE

# python graphsage.py --num-batch=2000 --dataset=reddit --model=GAT
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=10 --dataset=reddit --model=GAT
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=50 --dataset=reddit --model=GAT
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=100 --dataset=reddit --model=GAT
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=200 --dataset=reddit --model=GAT
# python graphsage_offlinesample.py --num-batch=2000 --num-sample=500 --dataset=reddit --model=GAT

# python graphsage.py --num-epoch=20 --dataset=ogbn-papers100M --use-uva=True
# python graphsage_offlinesample.py --num-epoch=20 --num-sample=0.05 --dataset=ogbn-papers100M --use-uva=True

# for i in {1..5}
# do
#     python graphsage.py --num-epoch=50 --dataset=products
#     python graphsage_offlinesample.py --num-epoch=50 --num-sample=0.05 --dataset=products
#     python graphsage_offlinesample.py --num-epoch=50 --num-sample=0.25 --dataset=products
#     python graphsage_offlinesample.py --num-epoch=50 --num-sample=0.5 --dataset=products
#     python graphsage_offlinesample.py --num-epoch=50 --num-sample=1 --dataset=products
#     python graphsage_offlinesample.py --num-epoch=50 --num-sample=2 --dataset=products
#     python graphsage_offlinesample.py --num-epoch=50 --num-sample=3 --dataset=products
# done

# for i in {1..5}
# do
#     python graphsage.py --num-epoch=50 --dataset=reddit
#     python graphsage_offlinesample.py --num-epoch=50 --num-sample=0.25 --dataset=reddit
#     python graphsage_offlinesample.py --num-epoch=50 --num-sample=0.5 --dataset=reddit
#     python graphsage_offlinesample.py --num-epoch=50 --num-sample=1 --dataset=reddit
#     python graphsage_offlinesample.py --num-epoch=50 --num-sample=2 --dataset=reddit
# done

# for i in {1..5}
# do
#     python ladies.py --num-epoch=50 --dataset=products
#     python ladies_offlinesample.py --num-epoch=50 --num-sample=0.25 --dataset=products
#     python ladies_offlinesample.py --num-epoch=50 --num-sample=0.5 --dataset=products
#     python ladies_offlinesample.py --num-epoch=50 --num-sample=1 --dataset=products
#     python ladies_offlinesample.py --num-epoch=50 --num-sample=2 --dataset=products
#     python ladies_offlinesample.py --num-epoch=50 --num-sample=3 --dataset=products
# done

# for i in {1..5}
# do
#     python ladies.py --num-epoch=50 --dataset=reddit
#     python ladies_offlinesample.py --num-epoch=50 --num-sample=0.25 --dataset=reddit
#     python ladies_offlinesample.py --num-epoch=50 --num-sample=0.5 --dataset=reddit
#     python ladies_offlinesample.py --num-epoch=50 --num-sample=1 --dataset=reddit
#     python ladies_offlinesample.py --num-epoch=50 --num-sample=2 --dataset=reddit
#     python ladies_offlinesample.py --num-epoch=50 --num-sample=3 --dataset=reddit
# done


# python graphsage_offlinesample.py --num-sample=200 --dataset=ogbn-papers100M --use-uva=True --fanout=10,10
# python graphsage_offlinesample.py --num-sample=200 --dataset=ogbn-papers100M --use-uva=True --fanout=10,10,10
# python graphsage_offlinesample.py --num-sample=200 --dataset=friendster --use-uva=True --fanout=10,10
# python graphsage_offlinesample.py --num-sample=200 --dataset=friendster --use-uva=True --fanout=10,10,10
# python graphsage_offlinesample.py --num-sample=200 --dataset=igb_tiny --use-uva=True --fanout=10,10 --path=/efs/rjliu/dataset/igb_tiny --dataset_size=tiny --in_memory=0
# python graphsage_offlinesample.py --num-sample=200 --dataset=igb_full --use-uva=True --fanout=10,10 --path=/efs/rjliu/dataset/igb_full --dataset_size=full --in_memory=0
# python graphsage_offlinesample.py --num-sample=200 --dataset=igb_full --use-uva=True --fanout=10,10,10 --path=/efs/rjliu/dataset/igb_full --dataset_size=full --in_memory=0
python graphsage_offlinesample.py --num-sample=200 --dataset=igb_large --use-uva=True --fanout=10,10 --path=/efs/rjliu/dataset/igb_large --dataset_size=large --in_memory=0
python graphsage_offlinesample.py --num-sample=200 --dataset=igb_large --use-uva=True --fanout=10,10,10 --path=/efs/rjliu/dataset/igb_large --dataset_size=large --in_memory=0
