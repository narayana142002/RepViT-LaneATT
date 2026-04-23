"""
LaneATT Detection Head — clean rewrite.

Key properties:
  - NO grid_sample  → fully INT8 quantizable, ONNX exportable
  - NO custom C++ NMS → pure PyTorch NMS
  - Integer index lookup for anchor feature pooling
  - NO BMM attention → fully deployable on custom boards
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .focal_loss import FocalLoss
from .matching   import match_proposals_with_targets


class LaneATTHead(nn.Module):
    def __init__(
        self,
        img_h=360,
        img_w=640,
        S=72,
        fmap_stride=8,
        in_channels=64,
        anchor_feat_channels=64,
        topk_anchors=1000,
        anchors_freq_path=None,
    ):
        super().__init__()

        self.img_h               = img_h
        self.img_w               = img_w
        self.S                   = S
        self.n_strips            = S - 1
        self.n_offsets           = S
        self.fmap_h              = img_h // fmap_stride
        self.fmap_w              = img_w // fmap_stride
        self.anchor_feat_channels = anchor_feat_channels

        self.anchor_ys     = torch.linspace(1, 0, steps=S,           dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)

        self.left_angles   = [72., 60., 49., 39., 30., 22.]
        self.right_angles  = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100.,
                              90., 80., 72., 60., 49., 39., 30., 15.]

        anchors, anchors_cut = self._generate_anchors(lateral_n=72, bottom_n=128)

        if anchors_freq_path is not None:
            mask        = torch.load(anchors_freq_path).cpu()
            ind         = torch.argsort(mask, descending=True)[:topk_anchors]
            anchors     = anchors[ind]
            anchors_cut = anchors_cut[ind]

        # non-persistent → not saved in state_dict (keeps file size small)
        self.register_buffer('anchors',     anchors,     persistent=False)
        self.register_buffer('anchors_cut', anchors_cut, persistent=False)

        # pre-compute integer indices for feature pooling (no grid_sample)
        cut_zs, cut_ys, cut_xs, invalid_mask = self._compute_cut_indices()
        self.register_buffer('cut_zs',       cut_zs,       persistent=False)
        self.register_buffer('cut_ys',       cut_ys,       persistent=False)
        self.register_buffer('cut_xs',       cut_xs,       persistent=False)
        self.register_buffer('invalid_mask', invalid_mask, persistent=False)

        feat_dim  = anchor_feat_channels * self.fmap_h

        self.conv1     = nn.Conv2d(in_channels, anchor_feat_channels, 1)
        self.cls_layer = nn.Linear(feat_dim, 2)
        self.reg_layer = nn.Linear(feat_dim, self.n_offsets + 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, conf_threshold=None, nms_thres=0., nms_topk=3000):
        B = x.shape[0]
        x = self.conv1(x)                                        # (B, C, fh, fw)

        # anchor feature pooling — integer index lookup, no grid_sample
        anchor_feats = self._cut_anchor_features(x)              # (B, N, C*fh)

        N    = len(self.anchors)
        feat = anchor_feats.view(B * N, -1)                      # (B*N, C*fh)

        cls_logits = self.cls_layer(feat).view(B, N, 2)
        reg_out    = self.reg_layer(feat).view(B, N, self.n_offsets + 1)

        # build proposals: anchor + predicted offsets
        proposals = self.anchors.unsqueeze(0).expand(B, -1, -1).clone()
        proposals[:, :, :2]  = cls_logits
        proposals[:, :, 4:] += reg_out

        if conf_threshold is None and (nms_topk is None or nms_topk <= 0 or nms_thres <= 0):
            return [
                (proposal, self.anchors, None, None)
                for proposal in proposals
            ]

        return self._nms(proposals, nms_thres, nms_topk, conf_threshold)

    # ------------------------------------------------------------------
    # Loss  (inspired by official LaneATT loss)
    # ------------------------------------------------------------------
    def loss(self, proposals_list, targets, cls_loss_weight=1.0, reg_loss_weight=2.0,
             neg_pos_ratio=4):
        focal     = FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1 = nn.SmoothL1Loss()

        cls_loss  = proposals_list[0][0].new_zeros(1).squeeze()  # tensor 0.0 on correct device
        reg_loss  = cls_loss.clone()
        total_pos = 0

        for (proposals, anchors, _, _), target in zip(proposals_list, targets):
            # keep only existing lanes
            target = target[target[:, 1] == 1]

            if len(target) == 0:
                # all proposals are negatives
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_loss   = cls_loss + focal(proposals[:, :2], cls_target).sum()
                continue

            with torch.no_grad():
                pos_mask, neg_mask, tgt_idx = match_proposals_with_targets(
                    self, anchors, target)

            positives = proposals[pos_mask]
            negatives = proposals[neg_mask]
            n_pos     = len(positives)
            total_pos += n_pos

            if n_pos == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_loss   = cls_loss + focal(proposals[:, :2], cls_target).sum()
                continue

            # Keep only the hardest negatives so classification does not get
            # overwhelmed by easy background anchors.
            if len(negatives) > 0 and neg_pos_ratio is not None and neg_pos_ratio > 0:
                max_neg = max(n_pos * neg_pos_ratio, neg_pos_ratio)
                if len(negatives) > max_neg:
                    neg_scores = F.softmax(negatives[:, :2], dim=1)[:, 1]
                    hard_idx   = torch.topk(neg_scores, k=max_neg, largest=True).indices
                    negatives  = negatives[hard_idx]

            # classification loss
            all_props  = torch.cat([positives, negatives], dim=0)
            cls_target = proposals.new_zeros(n_pos + len(negatives)).long()
            cls_target[:n_pos] = 1
            cls_loss = cls_loss + focal(all_props[:, :2], cls_target).sum() / n_pos

            # regression loss
            reg_pred   = positives[:, 4:]                        # (n_pos, S+1)
            with torch.no_grad():
                matched_gt  = target[tgt_idx[pos_mask]]          # (n_pos, 77)
                reg_target  = matched_gt[:, 4:].clone()          # (n_pos, S+1)

                # align start positions
                pos_starts  = (positives[:, 2] * self.n_strips).round().long().clamp(0, self.n_strips)
                tgt_starts  = (matched_gt[:, 2] * self.n_strips).round().long().clamp(0, self.n_strips)
                reg_target[:, 0] = reg_target[:, 0] - (pos_starts - tgt_starts).float()

                # mask out offsets outside lane extent
                all_idx = torch.arange(n_pos, device=positives.device)
                ends    = (pos_starts + reg_target[:, 0] - 1).round().long().clamp(0, self.n_strips)
                inv     = torch.zeros(n_pos, self.n_offsets + 2, dtype=torch.int, device=positives.device)
                inv[all_idx, (pos_starts + 1).clamp(0, self.n_offsets + 1)] += 1
                inv[all_idx, (ends + 2).clamp(0, self.n_offsets + 1)]       -= 1
                inv     = inv.cumsum(dim=1)[:, :self.n_offsets + 1] == 0
                inv[:, 0] = False
                reg_target[inv] = reg_pred[inv].detach()

            reg_loss = reg_loss + smooth_l1(reg_pred, reg_target)

        n          = len(targets)
        total_loss = cls_loss_weight * cls_loss / n + reg_loss_weight * reg_loss / n

        return total_loss, {
            'cls_loss':  (cls_loss / n).item(),
            'reg_loss':  (reg_loss / n).item(),
            'positives': total_pos,
        }

    # ------------------------------------------------------------------
    # Pure PyTorch NMS
    # ------------------------------------------------------------------
    def _nms(self, batch_proposals, nms_thres, nms_topk, conf_threshold):
        out = []
        for proposals in batch_proposals:
            with torch.no_grad():
                scores = F.softmax(proposals[:, :2], dim=1)[:, 1]
                if conf_threshold is not None:
                    keep        = scores > conf_threshold
                    proposals   = proposals[keep]
                    scores      = scores[keep]
                if len(proposals) == 0:
                    out.append((proposals, self.anchors[:0], None, None))
                    continue
                keep_idx = self._pure_nms(proposals, scores, nms_thres, nms_topk)
            out.append((proposals[keep_idx], self.anchors[keep_idx], None, keep_idx))
        return out

    def _pure_nms(self, proposals, scores, nms_thres, nms_topk):
        order = scores.argsort(descending=True)[:nms_topk]
        keep  = []
        while len(order) > 0:
            i = order[0].item()
            keep.append(i)
            if len(order) == 1:
                break
            cur   = proposals[i,         5:].unsqueeze(0)   # (1, S)
            rest  = proposals[order[1:],  5:]                # (M, S)
            valid = (cur > 0) & (rest > 0)
            n_valid = valid.sum(dim=1).clamp(min=1).float()
            dist  = (torch.abs(cur - rest) * valid.float()).sum(dim=1) / n_valid
            order = order[1:][dist > nms_thres]
        return torch.tensor(keep, dtype=torch.long, device=proposals.device)

    # ------------------------------------------------------------------
    # Anchor generation
    # ------------------------------------------------------------------
    def _generate_anchors(self, lateral_n, bottom_n):
        la, lc = self._side_anchors(self.left_angles,   x=0., nb=lateral_n)
        ra, rc = self._side_anchors(self.right_angles,  x=1., nb=lateral_n)
        ba, bc = self._side_anchors(self.bottom_angles, y=1., nb=bottom_n)
        return torch.cat([la, ba, ra]), torch.cat([lc, bc, rc])

    def _side_anchors(self, angles, nb, x=None, y=None):
        starts = [(sx, y) for sx in np.linspace(1., 0., num=nb)] if y is not None \
            else [(x, sy) for sy in np.linspace(1., 0., num=nb)]
        n           = nb * len(angles)
        anchors     = torch.zeros(n, 2 + 2 + 1 + self.n_offsets)
        anchors_cut = torch.zeros(n, 2 + 2 + 1 + self.fmap_h)
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k               = i * len(angles) + j
                anchors[k]      = self._one_anchor(start, angle, cut=False)
                anchors_cut[k]  = self._one_anchor(start, angle, cut=True)
        return anchors, anchors_cut

    def _one_anchor(self, start, angle, cut=False):
        ys  = self.anchor_cut_ys if cut else self.anchor_ys
        sz  = self.fmap_h        if cut else self.n_offsets
        a   = torch.zeros(2 + 2 + 1 + sz)
        rad = angle * math.pi / 180.
        sx, sy = start
        a[2]   = 1 - sy
        a[3]   = sx
        a[5:]  = (sx + (1 - ys - 1 + sy) / math.tan(rad)) * self.img_w
        return a

    # ------------------------------------------------------------------
    # Feature pooling helpers
    # ------------------------------------------------------------------
    def _compute_cut_indices(self):
        """Pre-compute integer pixel indices for anchor feature pooling."""
        n_anchors = len(self.anchors_cut)
        fh        = self.fmap_h
        fw        = self.fmap_w
        C         = self.anchor_feat_channels

        # x indices: (n_anchors, fh) — one x per row per anchor
        scale = self.img_w / fw
        xs    = (self.anchors_cut[:, 5:] / scale).round().long()   # (N, fh)
        xs    = torch.flip(xs, dims=[1])                            # flip: top→bottom

        invalid = (xs < 0) | (xs >= fw)                            # (N, fh)
        xs_clamped = xs.clamp(0, fw - 1)                           # (N, fh)

        # expand for C feature channels: (N*C, fh)
        # cut_zs: channel index, cut_ys: row index, cut_xs: col index
        cut_ys = torch.arange(fh, dtype=torch.long).unsqueeze(0).expand(n_anchors * C, -1)  # (N*C, fh)
        cut_zs = torch.arange(C, dtype=torch.long).view(1, C, 1).expand(n_anchors, -1, fh)
        cut_zs = cut_zs.reshape(n_anchors * C, fh)                                            # (N*C, fh)
        cut_xs = xs_clamped.unsqueeze(1).expand(-1, C, -1).reshape(n_anchors * C, fh)        # (N*C, fh)

        invalid_exp = invalid.unsqueeze(1).expand(-1, C, -1).reshape(n_anchors * C, fh)     # (N*C, fh)

        return cut_zs, cut_ys, cut_xs, invalid_exp

    def _cut_anchor_features(self, features):
        """
        Pool features along anchor lines using integer index lookup.
        features: (B, C, fh, fw)
        returns:  (B, N, C*fh)
        """
        B         = features.shape[0]
        N         = len(self.anchors)
        C         = self.anchor_feat_channels
        fh        = self.fmap_h

        out = torch.zeros(B, N * C, fh, device=features.device, dtype=features.dtype)

        for b in range(B):
            feat = features[b]                                   # (C, fh, fw)
            # index: feat[cut_zs[i], cut_ys[i,j], cut_xs[i,j]]
            vals = feat[self.cut_zs,
                        self.cut_ys,
                        self.cut_xs]                             # (N*C, fh)
            vals[self.invalid_mask] = 0.
            out[b] = vals

        return out.view(B, N, C * fh)
