# ogbl-ppa
# python seal_ogbl.py \
#     --dataset ogbl-ppa \
#     --use_feature \
#     --use_edge_weight \
#     --eval_steps 5 \
#     --epochs 20 \
#     --train_percent 5 

# ogbl-collab
# python seal_ogbl.py \
#     --dataset ogbl-collab \
#     --train_percent 15 \
#     --hidden_channels 256 \
#     --use_valedges_as_input

python seal_ogbl_offline.py \
    --dataset ogbl-collab \
    --train_percent 15 \
    --hidden_channels 256 \
    --use_valedges_as_input \
    --num_parts 6

python seal_ogbl_offline.py \
    --dataset ogbl-collab \
    --train_percent 15 \
    --hidden_channels 256 \
    --use_valedges_as_input \
    --num_parts 1

# ogbl-ddi
# python seal_ogbl.py \
#     --dataset ogbl-ddi \
#     --ratio_per_hop 0.2 \
#     --use_edge_weight \
#     --eval_steps 1 \
#     --epochs 10 \
#     --train_percent 5

# ogbl-citation2
# python seal_ogbl.py \
#     --dataset ogbl-citation2 \
#     --use_feature \
#     --use_edge_weight \
#     --eval_steps 1 \
#     --epochs 10 \
#     --train_percent 2 \
#     --val_percent 1 \
#     --test_percent 1