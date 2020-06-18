# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from kornia import denormalize
import cv2
import numpy as np

from util.box_ops import box_cxcywh_to_xyxy


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0):
    dfs = [pd.read_json(Path(p) / 'log.txt', lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(pd.np.stack(df.test_coco_eval.dropna().values)[:, 1]).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs

def plot_image(image, bbox):

    img = denormalize(image, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    bbox = box_cxcywh_to_xyxy(bbox)

    img_w, img_h = img.shape[1], img.shape[0]
    bbox.mul_(torch.tensor([img_w, img_h, img_w, img_h], device=bbox.device))
    x1, y1, x2, y2 = bbox

    img = cv2.rectangle(np.ascontiguousarray(img), (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.imshow("1", img[..., ::-1])
    cv2.waitKey(0)
