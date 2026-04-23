"""
Video inference script for RepViT-LaneATT.

Visualization follows LaneATT paper methodology:
  - Uses start_y and length from anchor proposal to determine
    valid lane extent (no manual horizon ratio needed)
  - Only draws lane from its predicted start row downward
  - Ego lane selection using bottom-x proximity to center
  - Smoothing with moving average for clean polylines
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RepViTLaneATT

COLORS = [(0,255,0),(0,0,255),(255,0,0),(255,255,0),(0,255,255),(255,0,255)]
MEAN   = np.array([0.485, 0.456, 0.406], np.float32)
STD    = np.array([0.229, 0.224, 0.225], np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',     required=True)
    p.add_argument('--input',          required=True)
    p.add_argument('--output',         default='output_lanes.mp4')
    p.add_argument('--img_h',          type=int,   default=360)
    p.add_argument('--img_w',          type=int,   default=640)
    p.add_argument('--S',              type=int,   default=72)
    p.add_argument('--conf_threshold', type=float, default=0.3)
    p.add_argument('--nms_thres',      type=float, default=45.0)
    p.add_argument('--nms_topk',       type=int,   default=8)
    p.add_argument('--fps',            type=int,   default=30)
    p.add_argument('--max_frames',     type=int,   default=None)
    p.add_argument('--ego_only',       action='store_true',
                   help='Keep only the 2 ego lanes nearest image center')
    return p.parse_args()


def preprocess(img, img_h, img_w):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_w, img_h))
    img = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


def lane_to_pts(lane, anchor_ys, img_h, img_w):
    """
    Convert lane proposal to (x,y) points using LaneATT methodology.
    Uses start_y and length from the anchor to determine valid extent.
    lane format: [cls0, cls1, start_y, start_x, length, x_0..x_71]
    """
    start_y = lane[2].item()   # normalised start y (0=bottom, 1=top)
    length  = lane[4].item()   # number of valid row anchors
    xs      = lane[5:].cpu().numpy()

    # start_row: which anchor index the lane begins at
    n_strips   = len(anchor_ys) - 1
    start_row  = round(start_y * n_strips)
    end_row    = min(int(start_row + length), len(anchor_ys))

    pts = []
    for i in range(start_row, end_row):
        x = xs[i]
        y = anchor_ys[i]
        # only keep valid in-bounds x
        if x > 0 and 0 <= x <= img_w:
            pts.append((float(x), float(y)))

    if len(pts) < 2:
        return []

    # smooth with moving average
    xs_arr = np.array([p[0] for p in pts])
    ys_arr = np.array([p[1] for p in pts])
    if len(xs_arr) >= 5:
        k      = np.ones(5) / 5
        xs_arr = np.convolve(xs_arr, k, mode='same')
        xs_arr[0]  = pts[0][0]
        xs_arr[-1] = pts[-1][0]

    return [(float(x), float(y)) for x, y in zip(xs_arr, ys_arr)]


def pick_ego_lanes(lanes_pts, img_w):
    """
    Pick 2 ego lanes — left and right of center.
    Strategy: find lane closest to left of center and closest to right of center.
    Uses bottom x position (most reliable point) for sorting.
    """
    if len(lanes_pts) == 0:
        return []
    if len(lanes_pts) <= 2:
        return lanes_pts

    center = img_w / 2
    # get bottom x of each lane (last point = bottom of image)
    bottom_xs = [pts[-1][0] for pts in lanes_pts]

    # split into left (x < center) and right (x > center)
    left_lanes  = [(i, x) for i, x in enumerate(bottom_xs) if x <= center]
    right_lanes = [(i, x) for i, x in enumerate(bottom_xs) if x > center]

    selected = []
    # pick closest left lane to center
    if left_lanes:
        best_left = max(left_lanes, key=lambda t: t[1])  # largest x < center
        selected.append(lanes_pts[best_left[0]])
    # pick closest right lane to center
    if right_lanes:
        best_right = min(right_lanes, key=lambda t: t[1])  # smallest x > center
        selected.append(lanes_pts[best_right[0]])

    return selected


def draw_lanes(frame, proposals, anchor_ys, img_h, img_w, ego_only=False):
    oh, ow = frame.shape[:2]
    sx, sy = ow / img_w, oh / img_h

    # collect lanes using LaneATT start_y/length methodology
    all_lanes = []
    for lane in proposals[0][0]:
        pts = lane_to_pts(lane, anchor_ys, img_h, img_w)
        if len(pts) >= 2:
            all_lanes.append(pts)

    lanes_to_draw = pick_ego_lanes(all_lanes, img_w) if ego_only else all_lanes
    colors = [(0, 255, 0), (0, 0, 255)] if ego_only else COLORS

    for i, pts in enumerate(lanes_to_draw):
        color  = colors[i % len(colors)]
        scaled = [(int(x * sx), int(y * sy)) for x, y in pts]
        for k in range(len(scaled) - 1):
            cv2.line(frame, scaled[k], scaled[k+1], color, 3)

    return frame, len(lanes_to_draw)


def get_frames(input_path):
    if os.path.isdir(input_path):
        exts  = ('.jpg', '.jpeg', '.png')
        files = sorted([os.path.join(input_path, f)
                        for f in os.listdir(input_path)
                        if f.lower().endswith(exts)])
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                yield img
    else:
        cap = cv2.VideoCapture(input_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = RepViTLaneATT(
        img_h=args.img_h, img_w=args.img_w,
        S=args.S, pretrained_backbone=False,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model', ckpt))
    model.eval()
    print(f"Loaded: {args.checkpoint}")
    print(f"Size: {model.model_size_mb():.1f} MB | Params: {model.param_count():,}")

    anchor_ys = np.linspace(1, 0, args.S) * args.img_h
    fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
    writer    = None
    count     = 0

    for frame in get_frames(args.input):
        if args.max_frames and count >= args.max_frames:
            break

        if writer is None:
            oh, ow = frame.shape[:2]
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (ow, oh))
            print(f"Output: {args.output} ({ow}x{oh} @ {args.fps}fps)")

        inp = preprocess(frame, args.img_h, args.img_w).to(device)
        with torch.no_grad():
            proposals = model(inp,
                              conf_threshold=args.conf_threshold,
                              nms_thres=args.nms_thres,
                              nms_topk=args.nms_topk)

        frame, n = draw_lanes(
            frame, proposals, anchor_ys, args.img_h, args.img_w,
            ego_only=args.ego_only,
        )
        label = 'Ego lanes' if args.ego_only else 'Lanes'
        cv2.putText(frame, f'{label}: {n} | Frame: {count+1}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        writer.write(frame)
        count += 1

        if count % 100 == 0:
            print(f"  {count} frames done...")

    if writer:
        writer.release()
    size = os.path.getsize(args.output) / 1024 / 1024 if os.path.exists(args.output) else 0
    print(f"\nDone! {count} frames | {args.output} ({size:.1f} MB)")


if __name__ == '__main__':
    main()
