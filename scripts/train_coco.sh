python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --no_aux_loss \
    --data_path datasets/data/coco --dataset_file coco --batch_size 2 \ 
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --eval
