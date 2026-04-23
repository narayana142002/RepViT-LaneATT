"""
CULane Dataset Loader.

CULane .lines.txt format (per lane, one lane per line):
  x1 y1 x2 y2 x3 y3 ...   (pixel coordinates, not normalised)

List file format (train_gt.txt):
  /driver_X/video.MP4/frame.jpg /laneseg/... exist1 exist2 exist3 exist4
"""

import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset


class CULaneDataset(Dataset):
    def __init__(
        self,
        data_root,
        split='train',
        img_h=360,
        img_w=640,
        S=72,
        max_lanes=4,
        augment=True,
    ):
        self.data_root = data_root
        self.img_h     = img_h
        self.img_w     = img_w
        self.S         = S
        self.max_lanes = max_lanes
        self.augment   = augment and (split == 'train')

        list_files = {
            'train': 'list/train_gt.txt',
            'val':   'list/val.txt',
            'test':  'list/test.txt',
        }
        self.samples   = self._load_list(os.path.join(data_root, list_files[split]))
        self.anchor_ys = np.linspace(1, 0, num=S, dtype=np.float32) * img_h  # row positions in pixels

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_list(self, path):
        samples = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                img_path = self.data_root + parts[0]
                samples.append(img_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]

        # --- load image ---
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((590, 1640, 3), dtype=np.uint8)  # CULane native size
        orig_h, orig_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- load lanes ---
        label_path = img_path.replace('.jpg', '.lines.txt')
        lanes      = self._load_lanes(label_path, orig_w, orig_h)

        # --- augment (before resize so coords are in original space) ---
        if self.augment:
            img, lanes = self._augment(img, lanes, orig_w, orig_h)

        # --- resize image ---
        img = cv2.resize(img, (self.img_w, self.img_h))

        # --- scale lane coords to resized image ---
        sx = self.img_w / orig_w
        sy = self.img_h / orig_h
        lanes = [[(x * sx, y * sy) for x, y in lane] for lane in lanes]

        # --- encode lanes ---
        target = self._encode_lanes(lanes)

        # --- normalise image ---
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, target

    def _load_lanes(self, path, orig_w, orig_h):
        """Load lanes from .lines.txt — each line is one lane: x1 y1 x2 y2 ..."""
        lanes = []
        if not os.path.exists(path):
            return lanes
        with open(path) as f:
            for line in f:
                nums = line.strip().split()
                if len(nums) < 4:
                    continue
                pts = list(map(float, nums))
                # pairs: (x, y)
                coords = [(pts[i], pts[i+1]) for i in range(0, len(pts) - 1, 2)]
                # filter out-of-bounds points
                coords = [(x, y) for x, y in coords
                          if 0 <= x <= orig_w and 0 <= y <= orig_h]
                if len(coords) >= 2:
                    lanes.append(coords)
        return lanes[:self.max_lanes]

    def _encode_lanes(self, lanes):
        """
        Encode lanes to target tensor (max_lanes, 2+2+1+S).
        Format per lane:
          [neg_score, pos_score, start_y, start_x, length, x_0..x_{S-1}]
          start_y, start_x normalised to [0,1]
          x_i in pixels (0 = invalid row)
        """
        target = torch.zeros(self.max_lanes, 2 + 2 + 1 + self.S)

        for i, lane in enumerate(lanes):
            if i >= self.max_lanes:
                break

            xs = np.array([p[0] for p in lane], dtype=np.float32)
            ys = np.array([p[1] for p in lane], dtype=np.float32)

            # sort by y ascending (top of image = small y)
            order = np.argsort(ys)
            xs, ys = xs[order], ys[order]

            # interpolate x at each anchor row position
            # anchor_ys goes from img_h (bottom) to 0 (top)
            # interp needs sorted xp → sort ys ascending
            xs_interp = np.interp(self.anchor_ys, ys, xs,
                                  left=-1., right=-1.)

            valid = (xs_interp >= 0) & (xs_interp <= self.img_w)
            if valid.sum() < 2:
                continue

            # Anchor parameterization starts from the lane entry point at the image
            # boundary, so use the bottom-most visible anchor row.
            valid_rows = np.where(valid)[0]
            start_row  = valid_rows[0]  # anchor_ys is bottom→top

            start_y = 1.0 - (self.anchor_ys[start_row] / self.img_h)
            start_x = float(xs_interp[start_row]) / self.img_w
            length  = float(valid.sum())

            xs_out = torch.zeros(self.S)
            xs_out[valid] = torch.from_numpy(xs_interp[valid].astype(np.float32))

            target[i, 1] = 1.
            target[i, 2] = float(start_y)
            target[i, 3] = float(start_x)
            target[i, 4] = float(length)
            target[i, 5:] = xs_out

        return target

    def _augment(self, img, lanes, orig_w, orig_h):
        # horizontal flip
        if random.random() < 0.5:
            img   = cv2.flip(img, 1)
            lanes = [[(orig_w - x, y) for x, y in lane] for lane in lanes]

        # brightness / contrast jitter
        if random.random() < 0.5:
            alpha = random.uniform(0.6, 1.4)  # contrast
            beta  = random.randint(-30, 30)    # brightness
            img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # shadow simulation
        if random.random() < 0.4:
            x1, x2 = random.randint(0, orig_w), random.randint(0, orig_w)
            shadow  = np.ones_like(img, dtype=np.float32)
            pts     = np.array([[x1, 0], [x2, orig_h], [0, orig_h], [0, 0]], np.int32)
            cv2.fillPoly(shadow, [pts], (random.uniform(0.4, 0.7),) * 3)
            img = np.clip(img.astype(np.float32) * shadow, 0, 255).astype(np.uint8)

        # perspective transform
        if random.random() < 0.3:
            margin = orig_w * 0.05
            src = np.float32([[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]])
            dst = np.float32([
                [random.uniform(0, margin),        random.uniform(0, margin)],
                [random.uniform(orig_w-margin, orig_w), random.uniform(0, margin)],
                [random.uniform(orig_w-margin, orig_w), random.uniform(orig_h-margin, orig_h)],
                [random.uniform(0, margin),        random.uniform(orig_h-margin, orig_h)],
            ])
            M_p  = cv2.getPerspectiveTransform(src, dst)
            img  = cv2.warpPerspective(img, M_p, (orig_w, orig_h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT)
            new_lanes = []
            for lane in lanes:
                new_pts = []
                for x, y in lane:
                    pt  = np.array([[[x, y]]], dtype=np.float32)
                    pt  = cv2.perspectiveTransform(pt, M_p)[0][0]
                    if 0 <= pt[0] < orig_w and 0 <= pt[1] < orig_h:
                        new_pts.append((float(pt[0]), float(pt[1])))
                if len(new_pts) >= 2:
                    new_lanes.append(new_pts)
            lanes = new_lanes

        # affine: rotate + scale + translate
        angle = random.uniform(-6, 6)
        scale = random.uniform(0.85, 1.15)
        tx    = random.uniform(-25, 25)
        ty    = random.uniform(-10, 10)
        cx, cy = orig_w / 2, orig_h / 2
        M     = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img   = cv2.warpAffine(img, M, (orig_w, orig_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        new_lanes = []
        for lane in lanes:
            new_pts = []
            for x, y in lane:
                nx = M[0,0]*x + M[0,1]*y + M[0,2]
                ny = M[1,0]*x + M[1,1]*y + M[1,2]
                if 0 <= nx < orig_w and 0 <= ny < orig_h:
                    new_pts.append((nx, ny))
            if len(new_pts) >= 2:
                new_lanes.append(new_pts)
        return img, new_lanes


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)
