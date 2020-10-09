# python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
#     --lr_drop 100 --epochs 150 --data_path datasets/data/openlogo --dataset_file openlogo \
#     --batch_size 1 --pyramid 3 2 \
#     --finetune https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --lr_drop 100 --epochs 150 --data_path datasets/data/coco --dataset_file coco \
    --batch_size 1 --fpn --num_queries 20 --output_dir output/coco/fpn_20q/checkpoint0099.pth

