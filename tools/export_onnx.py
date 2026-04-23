"""
Export RepViT-LaneATT to ONNX for deployment.

Usage:
    python tools/export_onnx.py --checkpoint work_dirs/repvit_laneatt/best.pth
"""

import argparse
import torch
from model import RepViTLaneATT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output',     default='repvit_laneatt.onnx')
    p.add_argument('--img_h',      type=int, default=360)
    p.add_argument('--img_w',      type=int, default=640)
    p.add_argument('--S',          type=int, default=72)
    p.add_argument('--topk_anchors', type=int, default=1000)
    p.add_argument('--fuse',       action='store_true', help='Fuse RepViT branches first')
    return p.parse_args()


class ExportWrapper(torch.nn.Module):
    """Wraps model to return raw proposals tensor (no NMS) for ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck     = model.neck
        self.head     = model.head

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        feat        = self.neck(c3, c4, c5)
        # return raw logits before NMS — fixed output shape for ONNX
        feat = self.head.conv1(feat)
        anchor_feats = self.head._cut_anchor_features(feat)
        B = x.shape[0]
        anchor_feats = anchor_feats.view(B * len(self.head.anchors),
                                         self.head.anchor_feat_channels * self.head.fmap_h)
        scores   = self.head.attention_layer(anchor_feats)
        attention = torch.softmax(scores, dim=1).view(B, len(self.head.anchors), -1)
        attn_mat  = torch.eye(len(self.head.anchors), device=x.device).unsqueeze(0).repeat(B,1,1)
        off_diag  = (attn_mat == 0).nonzero(as_tuple=False)
        attn_mat[off_diag[:,0], off_diag[:,1], off_diag[:,2]] = attention.flatten()
        anchor_feats = anchor_feats.view(B, len(self.head.anchors), -1)
        attended     = torch.bmm(attn_mat.transpose(1,2), anchor_feats)
        anchor_feats = torch.cat([attended, anchor_feats], dim=2)
        anchor_feats = anchor_feats.view(B * len(self.head.anchors), -1)
        cls_logits   = self.head.cls_layer(anchor_feats).view(B, len(self.head.anchors), 2)
        reg          = self.head.reg_layer(anchor_feats).view(B, len(self.head.anchors), self.head.n_offsets + 1)
        proposals    = torch.zeros(B, len(self.head.anchors), 5 + self.head.n_offsets, device=x.device)
        proposals   += self.head.anchors
        proposals[:,:,:2]  = cls_logits
        proposals[:,:,4:] += reg
        return proposals   # (B, N_anchors, 77)


def main():
    args   = parse_args()
    device = torch.device('cpu')

    model = RepViTLaneATT(
        img_h=args.img_h, img_w=args.img_w,
        S=args.S, topk_anchors=args.topk_anchors,
        pretrained_backbone=False,
    )
    ckpt  = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt))
    model.eval()

    if args.fuse:
        model.fuse_repvit()
        print("RepViT branches fused.")

    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, args.img_h, args.img_w)

    torch.onnx.export(
        wrapper, dummy, args.output,
        input_names=['image'],
        output_names=['proposals'],
        dynamic_axes={'image': {0: 'batch'}, 'proposals': {0: 'batch'}},
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"Exported: {args.output}")

    import os
    mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"ONNX size: {mb:.1f} MB")


if __name__ == '__main__':
    main()
