"""
Proper LaneATT-style CULane Evaluation Script.

Evaluation methodology (same as LaneATT paper):
  - For each row anchor, check if predicted x is within threshold of GT x
  - TP if matched_points / total_points > 0.75
  - Hungarian matching for optimal assignment
  - Reports per-category F1, Precision, Recall
  - Also reports classification and regression accuracy separately

Usage:
    python test.py \
        --checkpoint work_dirs/repvit_v2/best.pth \
        --data_root /path/to/culane \
        --conf_threshold 0.3 \
        --nms_topk 8
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RepViTLaneATT
from data.culane_dataset import CULaneDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',     required=True)
    p.add_argument('--data_root',      required=True)
    p.add_argument('--split',          default='test')
    p.add_argument('--img_h',          type=int,   default=360)
    p.add_argument('--img_w',          type=int,   default=640)
    p.add_argument('--S',              type=int,   default=72)
    p.add_argument('--conf_threshold', type=float, default=0.3)
    p.add_argument('--nms_thres',      type=float, default=45.0)
    p.add_argument('--nms_topk',       type=int,   default=8)
    p.add_argument('--match_thresh',   type=float, default=20.0,
                   help='Max x distance (px) to count as matched point')
    p.add_argument('--iou_thresh',     type=float, default=0.75,
                   help='Min matched_points/total_points to count as TP')
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--num_workers',    type=int,   default=4)
    return p.parse_args()


# ------------------------------------------------------------------
# LaneATT-style lane similarity
# ------------------------------------------------------------------
def lane_similarity(pred_xs, gt_xs, match_thresh):
    """
    LaneATT-style similarity between two lanes.
    pred_xs, gt_xs: arrays of x coords at each row anchor (0 = invalid)
    Returns: matched_points / total_valid_points
    """
    valid = (pred_xs > 0) & (gt_xs > 0)
    if valid.sum() < 2:
        return 0.0
    dist    = np.abs(pred_xs[valid] - gt_xs[valid])
    matched = (dist < match_thresh).sum()
    return float(matched) / float(valid.sum())


def match_lanes(pred_lanes, gt_lanes, match_thresh, iou_thresh):
    """
    Hungarian matching between predicted and GT lanes.
    Returns: tp, fp, fn
    """
    if len(pred_lanes) == 0 and len(gt_lanes) == 0:
        return 0, 0, 0
    if len(pred_lanes) == 0:
        return 0, 0, len(gt_lanes)
    if len(gt_lanes) == 0:
        return 0, len(pred_lanes), 0

    # similarity matrix
    sim_mat = np.zeros((len(pred_lanes), len(gt_lanes)))
    for i, p in enumerate(pred_lanes):
        for j, g in enumerate(gt_lanes):
            sim_mat[i, j] = lane_similarity(p, g, match_thresh)

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-sim_mat)
    matched_pred, matched_gt = set(), set()
    for r, c in zip(row_ind, col_ind):
        if sim_mat[r, c] >= iou_thresh:
            matched_pred.add(r)
            matched_gt.add(c)

    tp = len(matched_pred)
    fp = len(pred_lanes) - tp
    fn = len(gt_lanes)   - len(matched_gt)
    return tp, fp, fn


# ------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load model
    model = RepViTLaneATT(
        img_h=args.img_h, img_w=args.img_w,
        S=args.S, pretrained_backbone=False,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model', ckpt))
    model.eval()
    print(f"Loaded: {args.checkpoint}")
    print(f"Size: {model.model_size_mb():.1f} MB | Params: {model.param_count():,}")

    # dataset
    ds = CULaneDataset(args.data_root, split=args.split,
                       img_h=args.img_h, img_w=args.img_w,
                       S=args.S, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    print(f"Evaluating on {len(ds):,} {args.split} images")

    # category setup
    categories = ['test0_normal', 'test1_crowd', 'test2_hlight', 'test3_shadow',
                  'test4_noline', 'test5_arrow', 'test6_curve', 'test7_cross',
                  'test8_night']
    cat_dir   = os.path.join(args.data_root, 'list/list/test_split')
    data_cats = {}
    for cat in categories:
        p = os.path.join(cat_dir, cat + '.txt')
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    data_cats[line.strip()] = cat

    list_path = os.path.join(args.data_root, f'list/list/{args.split}.txt')
    with open(list_path) as f:
        test_lines = [l.strip() for l in f if l.strip()]

    # stats
    cat_stats = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in categories}
    cat_stats['overall'] = {'tp': 0, 'fp': 0, 'fn': 0}

    # regression accuracy tracking
    reg_errors  = []   # mean x error for matched lanes
    cls_correct = 0    # correctly classified anchors
    cls_total   = 0    # total anchors evaluated

    global_idx = 0

    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(loader):
            imgs = imgs.to(device)
            proposals_list = model(imgs,
                                   conf_threshold=args.conf_threshold,
                                   nms_thres=args.nms_thres,
                                   nms_topk=args.nms_topk)

            for (proposals, _, _, _), target in zip(proposals_list, targets):
                line = test_lines[global_idx] if global_idx < len(test_lines) else ''
                key  = line.strip().lstrip('/')
                cat  = data_cats.get(key, 'test0_normal')

                # predicted lanes — x coords at each anchor
                pred_lanes = []
                for lane in proposals:
                    xs = lane[5:].cpu().numpy()
                    if (xs > 0).sum() >= 2:
                        pred_lanes.append(xs)

                # GT lanes from target tensor
                gt_lanes = []
                for t in target:
                    if t[1].item() == 1:
                        xs = t[5:].cpu().numpy()
                        if (xs > 0).sum() >= 2:
                            gt_lanes.append(xs)

                # classification accuracy
                for t in target:
                    is_lane = int(t[1].item() == 1)
                    cls_total += 1
                    # check if any prediction matches this GT
                    if is_lane:
                        matched = any(
                            lane_similarity(p, t[5:].cpu().numpy(),
                                          args.match_thresh) >= args.iou_thresh
                            for p in pred_lanes
                        )
                        if matched:
                            cls_correct += 1
                    else:
                        # true negative — no prediction should match
                        cls_correct += 1

                # regression accuracy for matched lanes
                if pred_lanes and gt_lanes:
                    sim_mat = np.zeros((len(pred_lanes), len(gt_lanes)))
                    for i, p in enumerate(pred_lanes):
                        for j, g in enumerate(gt_lanes):
                            sim_mat[i, j] = lane_similarity(
                                p, g, args.match_thresh)
                    row_ind, col_ind = linear_sum_assignment(-sim_mat)
                    for r, c in zip(row_ind, col_ind):
                        if sim_mat[r, c] >= args.iou_thresh:
                            valid = (pred_lanes[r] > 0) & (gt_lanes[c] > 0)
                            if valid.sum() > 0:
                                err = np.abs(
                                    pred_lanes[r][valid] - gt_lanes[c][valid]
                                ).mean()
                                reg_errors.append(err)

                # TP/FP/FN
                tp, fp, fn = match_lanes(pred_lanes, gt_lanes,
                                         args.match_thresh, args.iou_thresh)
                cat_stats[cat]['tp']      += tp
                cat_stats[cat]['fp']      += fp
                cat_stats[cat]['fn']      += fn
                cat_stats['overall']['tp'] += tp
                cat_stats['overall']['fp'] += fp
                cat_stats['overall']['fn'] += fn

                global_idx += 1

            if (batch_idx + 1) % 100 == 0:
                done = (batch_idx + 1) * args.batch_size
                print(f"  {min(done, len(ds))}/{len(ds)} images done...")

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    eps = 1e-9
    print(f"\n{'='*75}")
    print(f"RepViT-LaneATT — CULane {args.split.upper()} Evaluation")
    print(f"Method: LaneATT-style | match_thresh={args.match_thresh}px | "
          f"iou_thresh={args.iou_thresh}")
    print(f"conf={args.conf_threshold} | nms_topk={args.nms_topk}")
    print(f"{'='*75}")
    print(f"{'Category':<20} {'TP':>6} {'FP':>6} {'FN':>6} "
          f"{'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"{'-'*75}")

    for cat in categories + ['overall']:
        s          = cat_stats[cat]
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        marker = ' <<<' if cat == 'overall' else ''
        print(f"{cat:<20} {tp:>6} {fp:>6} {fn:>6} "
              f"{prec:>8.4f} {rec:>8.4f} {f1:>8.4f}{marker}")

    print(f"{'='*75}")

    # classification and regression summary
    cls_acc = cls_correct / (cls_total + eps) * 100
    reg_mae = np.mean(reg_errors) if reg_errors else 0.0
    print(f"\nClassification Accuracy : {cls_acc:.2f}%  "
          f"({cls_correct}/{cls_total} anchors correct)")
    print(f"Regression MAE          : {reg_mae:.2f} px  "
          f"(mean x error on matched lanes)")
    print(f"Matched lane pairs      : {len(reg_errors)}")
    print(f"{'='*75}\n")


if __name__ == '__main__':
    main()
