"""
RepViT-LaneATT: Full lane detection model.

  RepViT-M1.0 backbone  (pretrained ImageNet 80%)
       +
  Lightweight FPN neck  (C3+C4+C5 → 64ch, stride-8)
       +
  LaneATT head          (anchor pooling, global attention, cls+reg)

Total: ~8.1M params, ~31 MB FP32, ~8 MB INT8
Constraints satisfied: <60MB, no grid_sample, expected F1 ~77-79%
"""

import torch
import torch.nn as nn

from .repvit_backbone import RepViTBackbone
from .fpn_neck        import FPNNeck
from .laneatt_head    import LaneATTHead


class RepViTLaneATT(nn.Module):
    def __init__(
        self,
        img_h=360,
        img_w=640,
        S=72,
        topk_anchors=1000,
        anchors_freq_path=None,
        pretrained_backbone=True,
    ):
        super().__init__()

        self.backbone = RepViTBackbone(pretrained=pretrained_backbone)

        self.neck = FPNNeck(
            in_channels=(112, 224, 448),
            out_channels=64,
        )

        self.head = LaneATTHead(
            img_h=img_h,
            img_w=img_w,
            S=S,
            fmap_stride=8,
            in_channels=64,
            anchor_feat_channels=64,
            topk_anchors=topk_anchors,
            anchors_freq_path=anchors_freq_path,
        )

    def forward(self, x, conf_threshold=None, nms_thres=0., nms_topk=3000):
        c3, c4, c5 = self.backbone(x)
        feat       = self.neck(c3, c4, c5)
        proposals  = self.head(feat, conf_threshold, nms_thres, nms_topk)
        return proposals

    def loss(self, proposals_list, targets, cls_loss_weight=1.0,
             reg_loss_weight=2.0, neg_pos_ratio=4):
        return self.head.loss(
            proposals_list,
            targets,
            cls_loss_weight=cls_loss_weight,
            reg_loss_weight=reg_loss_weight,
            neg_pos_ratio=neg_pos_ratio,
        )

    # ------------------------------------------------------------------
    # Quantization helpers
    # ------------------------------------------------------------------
    def fuse_repvit(self):
        """Merge RepViT reparameterization branches before quantization."""
        for m in self.backbone.modules():
            if hasattr(m, 'fuse'):
                m.fuse()
        print("RepViT branches fused.")

    def prepare_qat(self, backend='fbgemm'):
        """Prepare model for Quantization-Aware Training."""
        import torch.ao.quantization as tq
        self.fuse_repvit()
        self.train()
        self.qconfig = tq.get_default_qat_qconfig(backend)
        # skip first conv and last linear — standard QAT practice
        self.backbone.patch_embed[0].qconfig = None
        self.head.cls_layer.qconfig          = None
        self.head.reg_layer.qconfig          = None
        tq.prepare_qat(self, inplace=True)
        print(f"QAT prepared with backend={backend}")

    def convert_to_int8(self):
        """Convert QAT-trained model to INT8."""
        import torch.ao.quantization as tq
        self.eval()
        tq.convert(self, inplace=True)
        print("Converted to INT8.")

    def model_size_mb(self):
        import os, tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(self.state_dict(), f.name)
            mb = os.path.getsize(f.name) / 1024 / 1024
            os.unlink(f.name)
        return mb

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
