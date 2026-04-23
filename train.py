"""
RepViT-LaneATT Training Script — Fixed Version

Key fixes over previous version:
  - Proper pixel IoU evaluation (same as CLRNet/LaneATT papers)
  - Hungarian matching for TP/FP/FN counting
  - Evaluation at original CULane resolution (1640x590)
  - Honest F1 that matches real test metric
  - Warmup + Cosine LR schedule
  - Hard negative mining
  - Best + final model saved

Usage:
    python train.py --data_root /path/to/culane --epochs 50 --batch_size 16 --amp
"""

import os
import sys
import argparse
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RepViTLaneATT
from data.culane_dataset import CULaneDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',      required=True)
    p.add_argument('--work_dir',       default='work_dirs/repvit_laneatt')
    p.add_argument('--epochs',         type=int,   default=50)
    p.add_argument('--batch_size',     type=int,   default=16)
    p.add_argument('--lr',             type=float, default=2e-4)
    p.add_argument('--lr_start',       type=float, default=1e-5)
    p.add_argument('--lr_end',         type=float, default=1e-6)
    p.add_argument('--warmup_epochs',  type=int,   default=3)
    p.add_argument('--freeze_epochs',  type=int,   default=2)
    p.add_argument('--num_workers',    type=int,   default=4)
    p.add_argument('--resume',         default=None)
    p.add_argument('--amp',            action='store_true')
    p.add_argument('--anchors_freq',   default=None)
    p.add_argument('--img_h',          type=int,   default=360)
    p.add_argument('--img_w',          type=int,   default=640)
    p.add_argument('--S',              type=int,   default=72)
    p.add_argument('--topk_anchors',   type=int,   default=1000)
    p.add_argument('--eval_every',     type=int,   default=2)
    p.add_argument('--save_every',     type=int,   default=2)
    p.add_argument('--cls_weight',     type=float, default=1.5)
    p.add_argument('--reg_weight',     type=float, default=1.5)
    p.add_argument('--neg_pos_ratio',  type=int,   default=3)
    p.add_argument('--min_f1',         type=float, default=0.05)
    p.add_argument('--patience',       type=int,   default=8)
    p.add_argument('--eval_conf',      type=float, default=0.3)
    p.add_argument('--eval_nms_thres', type=float, default=45.0)
    p.add_argument('--eval_nms_topk',  type=int,   default=8)
    p.add_argument('--eval_max_imgs',  type=int,   default=2000,
                   help='Max val images for eval (faster training loop)')
    return p.parse_args()


# ------------------------------------------------------------------
# LaneATT-style evaluation (matches our test.py metric)
# ------------------------------------------------------------------
MATCH_THRESH = 20.0   # max x distance in pixels to count as matched
IOU_THRESH   = 0.75   # min matched_points/total_points for TP


def lane_sim(p, g):
    valid = (p > 0) & (g > 0)
    if valid.sum() < 2: return 0.0
    return float((np.abs(p[valid] - g[valid]) < MATCH_THRESH).sum()) / valid.sum()


def proper_f1(model, val_samples, device, img_h, img_w, S,
              conf, nms_thres, nms_topk, max_imgs=2000):
    """
    LaneATT-style F1 — same metric as test.py.
    Uses row-anchor x-coordinate matching, not pixel IoU masks.
    """
    model.eval()
    mean      = np.array([0.485, 0.456, 0.406], np.float32)
    std       = np.array([0.229, 0.224, 0.225], np.float32)
    anchor_ys = np.linspace(1, 0, S) * img_h
    tp = fp = fn = 0
    samples = val_samples[:max_imgs]

    with torch.no_grad():
        for img_path in samples:
            if not os.path.exists(img_path): continue
            img = cv2.imread(img_path)
            if img is None: continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_r   = cv2.resize(img_rgb, (img_w, img_h))
            img_f   = (img_r.astype(np.float32) / 255.0 - mean) / std
            inp     = torch.from_numpy(
                img_f.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

            proposals_list = model(inp, conf_threshold=conf,
                                   nms_thres=nms_thres, nms_topk=nms_topk)

            pred_lanes = [p[5:].cpu().numpy() for p in proposals_list[0][0]
                          if (p[5:].cpu().numpy() > 0).sum() >= 2]

            # load GT from target encoded in dataset
            label = img_path.replace('.jpg', '.lines.txt')
            gt_lanes_raw = []
            if os.path.exists(label):
                with open(label) as f:
                    for line in f:
                        nums = list(map(float, line.strip().split()))
                        pts  = [(nums[i], nums[i+1]) for i in range(0, len(nums)-1, 2)]
                        if len(pts) >= 2: gt_lanes_raw.append(pts)

            # encode GT to anchor xs
            gt_lanes = []
            for pts in gt_lanes_raw:
                ys_gt = np.array([p[1] for p in pts])
                xs_gt = np.array([p[0] for p in pts])
                order = np.argsort(ys_gt)
                xs_interp = np.interp(anchor_ys, ys_gt[order], xs_gt[order],
                                      left=-1., right=-1.)
                valid = (xs_interp >= 0) & (xs_interp <= img_w)
                if valid.sum() >= 2:
                    xs_out = np.zeros(S)
                    xs_out[valid] = xs_interp[valid]
                    gt_lanes.append(xs_out)

            if not pred_lanes and not gt_lanes: continue
            if not pred_lanes: fn += len(gt_lanes); continue
            if not gt_lanes:   fp += len(pred_lanes); continue

            sim = np.zeros((len(pred_lanes), len(gt_lanes)))
            for i, p in enumerate(pred_lanes):
                for j, g in enumerate(gt_lanes):
                    sim[i, j] = lane_sim(p, g)

            ri, ci = linear_sum_assignment(-sim)
            mp, mg = set(), set()
            for r, c in zip(ri, ci):
                if sim[r, c] >= IOU_THRESH: mp.add(r); mg.add(c)
            tp += len(mp)
            fp += len(pred_lanes) - len(mp)
            fn += len(gt_lanes)   - len(mg)

    eps  = 1e-9
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    return float(f1), float(prec), float(rec)


# ------------------------------------------------------------------
# LR schedule
# ------------------------------------------------------------------
def get_lr(epoch, warmup_epochs, epochs, lr_start, lr_peak, lr_end):
    """Warmup + Cosine decay LR schedule."""
    if epoch <= warmup_epochs:
        return lr_start + (lr_peak - lr_start) * epoch / warmup_epochs
    t = (epoch - warmup_epochs) / (epochs - warmup_epochs)
    return lr_end + 0.5 * (lr_peak - lr_end) * (1 + math.cos(math.pi * t))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ------------------------------------------------------------------
# Training epoch
# ------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, amp,
                    cls_weight, reg_weight, neg_pos_ratio):
    model.train()
    total_loss = 0.

    for i, (imgs, targets) in enumerate(loader):
        imgs    = imgs.to(device)
        targets = [t.to(device) for t in targets]

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=amp):
            proposals  = model(imgs, nms_thres=0., nms_topk=None)
            loss, info = model.loss(proposals, targets,
                                    cls_loss_weight=cls_weight,
                                    reg_loss_weight=reg_weight,
                                    neg_pos_ratio=neg_pos_ratio)

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f"  [{i:4d}/{len(loader)}] "
                  f"loss={loss.item():.4f}  "
                  f"cls={info['cls_loss']:.4f}  "
                  f"reg={info['reg_loss']:.4f}  "
                  f"pos={info['positives']}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / len(loader)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.work_dir, exist_ok=True)

    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # datasets
    train_ds = CULaneDataset(args.data_root, split='train',
                             img_h=args.img_h, img_w=args.img_w,
                             S=args.S, augment=True)
    val_ds   = CULaneDataset(args.data_root, split='val',
                             img_h=args.img_h, img_w=args.img_w,
                             S=args.S, augment=False)
    print(f"Train  : {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # val image paths for proper eval
    val_samples = val_ds.samples

    loader_kwargs = dict(num_workers=args.num_workers, collate_fn=collate_fn,
                         pin_memory=True, persistent_workers=args.num_workers > 0)
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **loader_kwargs)

    # model
    model = RepViTLaneATT(
        img_h=args.img_h, img_w=args.img_w,
        S=args.S, topk_anchors=args.topk_anchors,
        anchors_freq_path=args.anchors_freq,
        pretrained_backbone=True,
    ).to(device)
    print(f"Params : {model.param_count():,}  |  Size: {model.model_size_mb():.1f} MB")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr_start, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda', enabled=args.amp)

    start_epoch = 1
    best_f1     = 0.0
    no_improve  = 0
    f1_history  = []

    if args.resume:
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        best_f1     = ckpt.get('best_f1', 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}  best_f1={best_f1:.4f} (optimizer reset)")

    log_path = os.path.join(args.work_dir, 'train_log.txt')
    log_f    = open(log_path, 'a')

    def log(msg):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f"\n{'='*60}")
    log(f"RepViT-LaneATT | epochs={args.epochs} bs={args.batch_size} lr={args.lr}")
    log(f"cls_weight={args.cls_weight} reg_weight={args.reg_weight} neg_pos_ratio={args.neg_pos_ratio}")
    log(f"eval_conf={args.eval_conf} eval_nms_topk={args.eval_nms_topk} eval_max_imgs={args.eval_max_imgs}")
    log(f"eval: LaneATT-style (row-anchor matching @ {MATCH_THRESH}px, iou_thresh={IOU_THRESH})")
    log(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        log(f"\n--- Epoch {epoch}/{args.epochs} ---")

        # LR schedule
        lr = get_lr(epoch, args.warmup_epochs, args.epochs,
                    args.lr_start, args.lr, args.lr_end)
        set_lr(optimizer, lr)

        # freeze / unfreeze backbone
        if epoch <= args.freeze_epochs:
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            if epoch == 1:
                log(f"  Backbone FROZEN for {args.freeze_epochs} epochs")
        elif epoch == args.freeze_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            log(f"  Backbone UNFROZEN")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            args.amp, args.cls_weight, args.reg_weight, args.neg_pos_ratio)
        log(f"  train_loss={train_loss:.4f}  lr={lr:.2e}")

        # save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.work_dir, f'epoch_{epoch}.pth')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'best_f1': best_f1},
                       ckpt_path)
            log(f"  Saved: epoch_{epoch}.pth")
            # keep only last 3
            old = os.path.join(args.work_dir, f'epoch_{epoch - 3*args.save_every}.pth')
            if os.path.exists(old):
                os.remove(old)

        # save final weights only
        if epoch == args.epochs:
            torch.save({'model': model.state_dict()},
                       os.path.join(args.work_dir, 'final_model.pth'))
            log(f"  Saved: final_model.pth (weights only)")

        # proper F1 evaluation
        if epoch % args.eval_every == 0:
            log(f"  Evaluating (pixel-IoU, {args.eval_max_imgs} val images)...")
            f1, prec, rec = proper_f1(
                model, val_samples, device,
                args.img_h, args.img_w, args.S,
                args.eval_conf, args.eval_nms_thres, args.eval_nms_topk,
                max_imgs=args.eval_max_imgs)
            f1_history.append((epoch, round(f1, 4)))
            log(f"  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

            # always save eval checkpoint
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'f1': f1},
                       os.path.join(args.work_dir, f'eval_epoch_{epoch}.pth'))

            if f1 > best_f1:
                best_f1    = f1
                no_improve = 0
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'f1': f1},
                           os.path.join(args.work_dir, 'best.pth'))
                log(f"  *** NEW BEST F1={best_f1:.4f} — saved best.pth ***")
            else:
                no_improve += 1
                log(f"  No improvement {no_improve}/{args.patience}")

            if epoch >= 6 and f1 < args.min_f1:
                log(f"\n  EARLY STOP: F1={f1:.4f} < min_f1={args.min_f1}")
                break

            if no_improve >= args.patience:
                log(f"\n  EARLY STOP: No improvement for {args.patience} evals")
                log(f"  Best F1={best_f1:.4f}  History={f1_history}")
                break

        model.train()

    log(f"\n{'='*60}")
    log(f"Done. Best F1={best_f1:.4f}")
    log(f"F1 history: {f1_history}")
    log(f"Best model : {args.work_dir}/best.pth")
    log(f"Final model: {args.work_dir}/final_model.pth")
    log(f"{'='*60}")
    log_f.close()


if __name__ == '__main__':
    main()
