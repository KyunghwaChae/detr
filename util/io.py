import os
import random
import math
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from util.misc import is_main_process
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from datasets.transforms import resize
from util.box_ops import rescale_bboxes


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

    ignore = ["unscaled", "_0", "_1", "_2", "_3", "_4", "lr", "cardinality"]

    log_train = {f'train_{k}': v for k, v in train_stats.items() if not any(substr in k for substr in ignore)}
    log_test = {f'test_{k}': v for k, v in test_stats.items() if not any(substr in k for substr in ignore)}


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


def finetune(args, model):

    if args.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.finetune, map_location='cpu')

    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]
    model.load_state_dict(checkpoint['model'], strict=False)



def load_frozen(args, model):

    checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    model.detr.load_state_dict(checkpoint['model'])


def create_wandb_img(classes, img_path, target, preds, att_map, shape, dec_att):

    prob = F.softmax(preds["pred_logits"], -1)
    scores, labels = prob[..., :-1].max(-1)
    img = Image.open(img_path)

    # size for logging purposes
    tensor_img = ToTensor()(resize(img, size=(1500, 1333), target=None)[0])

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

    wimg = wandb.Image(tensor_img, boxes=boxes, caption="Image: " + str(target["image_id"].item()))

    # resize to feedforward size
    tensor_img = ToTensor()(resize(img, size=800, target=None, max_size=1333)[0])

    # Taken from https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb

    # select 4 highest scores
    keep = torch.sort(scores, 0, descending=True)[1][:4]
    boxes = preds["pred_boxes"][keep]

    fact = 2 ** round(math.log2(tensor_img.shape[-1] / att_map[-1].shape[-1]))

    # how much was the original image upsampled before feeding it to the model
    scale_y = img.height / tensor_img.shape[-2]
    scale_x = img.width / tensor_img.shape[-1]

    colors = ['lime', 'deepskyblue', 'orange', 'red']

    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, -1]),
        fig.add_subplot(gs[1, -1]),
    ]

    # visualize encoder self attention
    coords = []
    num_q = dec_att[0].shape[-2]
    for idx_o, idx_q, ax, col in zip(boxes, keep, axs, colors):

        x = int((idx_o[0] * tensor_img.shape[-1]).item())
        y = int((idx_o[1] * tensor_img.shape[-2]).item())
        coords.append((x, y))

        idx = ((x // fact), y // fact)

        level = "3" if idx_q // num_q == 4 else "2"

        ax.imshow(att_map[idx_q // num_q][..., idx[1], idx[0]], cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'Self-attention level {level}: {col} ({classes[labels[idx_q].item()]})')

    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(img)

    for (x, y), col in zip(coords, colors):
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        fcenter_ax.add_patch(plt.Circle((x * scale_x, y * scale_y), fact // 4, color=col))
        fcenter_ax.axis('off')

    self_att = wandb.Image(fig, caption="Image: " + str(target["image_id"].item()))

    bboxes_scaled = rescale_bboxes(preds["pred_boxes"][keep].cpu(), (img.width, img.height))

    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, -1]),
        fig.add_subplot(gs[1, -1]),
    ]
    h, w = shape
    for idx, ax, col in zip(keep, axs, colors):

        level = "3" if idx // num_q == 4 else "2"

        ax.imshow(dec_att[idx // num_q][0, idx % num_q].view(h, w))
        ax.axis('off')
        ax.set_title(f'Attention level {level}: {col} ({classes[labels[idx].item()]})')

    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(img)

    for col, (xmin, ymin, xmax, ymax) in zip(colors, bboxes_scaled):
        fcenter_ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=col, linewidth=2))

    att_map = wandb.Image(plt, caption="Image: " + str(target["image_id"].item()))
    plt.close()












    return wimg, self_att, att_map
