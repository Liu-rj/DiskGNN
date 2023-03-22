for i in {1..5}
do
    python graphsage.py --num-epoch=50 --dataset=products
    python graphsage_offlinesample.py --num-epoch=50 --num-sample=0.25 --dataset=products
    python graphsage_offlinesample.py --num-epoch=50 --num-sample=0.5 --dataset=products
    python graphsage_offlinesample.py --num-epoch=50 --num-sample=1 --dataset=products
    python graphsage_offlinesample.py --num-epoch=50 --num-sample=2 --dataset=products
    python graphsage_offlinesample.py --num-epoch=50 --num-sample=3 --dataset=products
done

for i in {1..5}
do
    python graphsage.py --num-epoch=50 --dataset=reddit
    python graphsage_offlinesample.py --num-epoch=50 --num-sample=0.25 --dataset=reddit
    python graphsage_offlinesample.py --num-epoch=50 --num-sample=0.5 --dataset=reddit
    python graphsage_offlinesample.py --num-epoch=50 --num-sample=1 --dataset=reddit
    python graphsage_offlinesample.py --num-epoch=50 --num-sample=2 --dataset=reddit
done

for i in {1..5}
do
    python ladies.py --num-epoch=50 --dataset=products
    python ladies_offlinesample.py --num-epoch=50 --num-sample=0.25 --dataset=products
    python ladies_offlinesample.py --num-epoch=50 --num-sample=0.5 --dataset=products
    python ladies_offlinesample.py --num-epoch=50 --num-sample=1 --dataset=products
    python ladies_offlinesample.py --num-epoch=50 --num-sample=2 --dataset=products
    python ladies_offlinesample.py --num-epoch=50 --num-sample=3 --dataset=products
done

for i in {1..5}
do
    python ladies.py --num-epoch=50 --dataset=reddit
    python ladies_offlinesample.py --num-epoch=50 --num-sample=0.25 --dataset=reddit
    python ladies_offlinesample.py --num-epoch=50 --num-sample=0.5 --dataset=reddit
    python ladies_offlinesample.py --num-epoch=50 --num-sample=1 --dataset=reddit
    python ladies_offlinesample.py --num-epoch=50 --num-sample=2 --dataset=reddit
    python ladies_offlinesample.py --num-epoch=50 --num-sample=3 --dataset=reddit
done