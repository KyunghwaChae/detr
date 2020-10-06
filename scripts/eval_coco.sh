python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --no_aux_loss \
    --data_path datasets/data/coco --dataset_file coco --batch_size 1 \
    --eval --hidden_dim 512 --resume output/coco/fpn_20q/checkpoint0099.pth \
    --pyramid 3 2 --output_dir output/coco/fpn_20q --num_queries 20
