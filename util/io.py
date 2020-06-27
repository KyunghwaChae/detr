import os
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from util.misc import is_main_process
from torchvision.transforms import ToTensor
from datasets.transforms import resize


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_wandb(name, model, args, **kwargs):

    wandb.init(project=name, dir=args.output_dir, config=args)
    wandb.watch(model, log="all")
    for k, v in kwargs.items():
        wandb.config.update({k: v})


def log_wandb(train_stats, test_stats):

    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR@1", "AR@10", "AR@100", "AR@100s", "AR@100m", "AR@100l"]

    for k, v in zip(keys, test_stats["coco_eval_bbox"]):
        test_stats[k] = v
    test_stats.pop("coco_eval_bbox")

    log_train = {f'train_{k}': v for k, v in train_stats.items()}
    log_test = {f'test_{k}': v for k, v in test_stats.items()}


    wandb.log(log_train, commit=False)
    wandb.log(log_test)


def save_checkpoint(args, model, optimizer, lr_scheduler, epoch):

    checkpoint_path = os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth')
    save_on_master({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }, checkpoint_path)


def resume(args, model, optimizer, lr_scheduler):

    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1


def load_frozen(args, model):

    checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    model.detr.load_state_dict(checkpoint['model'])


def create_wandb_img(classes, img_path, target, preds):

    prob = F.softmax(preds["pred_logits"], -1)
    scores, labels = prob[..., :-1].max(-1)
    img = Image.open(img_path)

    img = ToTensor()(resize(img, size=(800, 1333), target=None)[0])

    boxes_data = []
    for sc, cl, (cx, cy, width, height) in zip(scores.tolist(), labels.tolist(), preds["pred_boxes"].tolist()):
        boxes_data.append({"position": {"middle": (cx, cy), "width": width, "height": height},
                           "box_caption": f"{classes[cl]}: {sc:0.2f}",
                           "class_id": cl, "scores": {"score": sc}})

    gt_data = []
    for cl, (cx, cy, width, height) in zip(target["labels"].tolist(), target["boxes"].tolist()):

        gt_data.append({"position": {"middle": (cx, cy), "width": width, "height": height},
                        "box_caption": f"{classes[cl]}",
                        "class_id": cl, "scores": {"score": 1.0}})

    boxes = {"predictions": {"box_data": boxes_data, "class_labels": classes}}
    boxes["ground_truth"] = {"box_data": gt_data, "class_labels": classes}

    img = wandb.Image(img, boxes=boxes, caption="Image: " + str(target["image_id"].item()))

    return img
