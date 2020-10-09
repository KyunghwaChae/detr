# python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --no_aux_loss \
#     --data_path datasets/data/coco --dataset_file coco --batch_size 1 \
#     --eval --hidden_dim 512 --resume output/coco/fpn_20q/checkpoint0099.pth \
#     --fpn --output_dir output/coco/fpn_20q --num_queries 20

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --no_aux_loss \
    --data_path datasets/data/coco --dataset_file coco --batch_size 1 \
    --eval --resume output/coco/base/model_final_new.pth \
    --output_dir output/coco/base
