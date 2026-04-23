"""
RepViT-M1.0 Backbone with multi-scale feature output (C3, C4, C5).
Pretrained on ImageNet-1K (80.0% Top-1).

Feature maps for input 360x640:
  C3: (B, 112, 45, 80)  stride=8
  C4: (B, 224, 23, 40)  stride=16
  C5: (B, 448, 11, 20)  stride=32
"""

import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, groups=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, groups=groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(w.size(1) * c.groups, w.size(0), w.shape[2:],
                      stride=c.stride, padding=c.padding, groups=c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    def __init__(self, m, drop=0.0):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert m.groups == m.in_channels
            identity = torch.nn.functional.pad(
                torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1), [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        return self


class RepVGGDW(nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.conv  = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.bn    = nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + x)

    @torch.no_grad()
    def fuse(self):
        conv  = self.conv.fuse()
        conv1 = self.conv1
        conv1_w = nn.functional.pad(conv1.weight, [1, 1, 1, 1])
        identity = nn.functional.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1, 1, 1, 1])
        conv.weight.data.copy_(conv.weight + conv1_w + identity)
        conv.bias.data.copy_(conv.bias + conv1.bias)
        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(conv.weight * w[:, None, None, None])
        conv.bias.data.copy_(bn.bias + (conv.bias - bn.running_mean) * bn.weight /
                             (bn.running_var + bn.eps) ** 0.5)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        assert stride in [1, 2]
        assert hidden_dim == 2 * inp
        self.identity = (stride == 1 and inp == oup)

        act = nn.GELU

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, 1, 1, 0),
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(oup, 2 * oup, 1, 1, 0), act(),
                Conv2d_BN(2 * oup, oup, 1, 1, 0, ),
            ))
        else:
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(inp, hidden_dim, 1, 1, 0), act(),
                Conv2d_BN(hidden_dim, oup, 1, 1, 0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


# RepViT-M1.0 config: [kernel, expand, channels, SE, HS, stride]
REPVIT_M1_0_CFG = [
    [3, 2,  56, 1, 0, 1],
    [3, 2,  56, 0, 0, 1],
    [3, 2,  56, 0, 0, 1],
    [3, 2, 112, 0, 0, 2],  # -> C3 (stride 8)
    [3, 2, 112, 1, 0, 1],
    [3, 2, 112, 0, 0, 1],
    [3, 2, 112, 0, 0, 1],
    [3, 2, 224, 0, 1, 2],  # -> C4 (stride 16)
    [3, 2, 224, 1, 1, 1],
    [3, 2, 224, 0, 1, 1],
    [3, 2, 224, 1, 1, 1],
    [3, 2, 224, 0, 1, 1],
    [3, 2, 224, 1, 1, 1],
    [3, 2, 224, 0, 1, 1],
    [3, 2, 224, 1, 1, 1],
    [3, 2, 224, 0, 1, 1],
    [3, 2, 224, 1, 1, 1],
    [3, 2, 224, 0, 1, 1],
    [3, 2, 224, 1, 1, 1],
    [3, 2, 224, 0, 1, 1],
    [3, 2, 224, 1, 1, 1],
    [3, 2, 224, 0, 1, 1],
    [3, 2, 224, 0, 1, 1],
    [3, 2, 448, 0, 1, 2],  # -> C5 (stride 32)
    [3, 2, 448, 1, 1, 1],
    [3, 2, 448, 0, 1, 1],
]


class RepViTBackbone(nn.Module):
    """
    RepViT-M1.0 backbone returning multi-scale features C3, C4, C5.
    C3: stride=8,  channels=112
    C4: stride=16, channels=224
    C5: stride=32, channels=448
    """

    # indices in self.stages where stride-2 blocks produce C3, C4, C5
    C3_IDX = 3   # block index that produces stride-8 output
    C4_IDX = 7   # block index that produces stride-16 output
    C5_IDX = 23  # block index that produces stride-32 output

    def __init__(self, pretrained=True):
        super().__init__()

        # patch_embed: two stride-2 convs = overall stride 4
        input_channel = REPVIT_M1_0_CFG[0][2]
        self.patch_embed = nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1),
        )

        self.blocks = nn.ModuleList()
        for k, t, c, se, hs, s in REPVIT_M1_0_CFG:
            out_ch = _make_divisible(c, 8)
            exp_ch = _make_divisible(input_channel * t, 8)
            self.blocks.append(RepViTBlock(input_channel, exp_ch, out_ch, k, s, se, hs))
            input_channel = out_ch

        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        import urllib.request, os
        url  = "https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_0_distill_300e.pth"
        path = os.path.expanduser("~/.cache/repvit_m1_0_distill_300e.pth")
        if not os.path.exists(path):
            print(f"Downloading RepViT-M1.0 pretrained weights...")
            urllib.request.urlretrieve(url, path)
        state = torch.load(path, map_location="cpu")
        state = state.get("model", state)
        # keep only backbone keys (drop classifier)
        state = {k: v for k, v in state.items()
                 if not k.startswith("classifier")}
        # remap: features.0 -> patch_embed, features.N -> blocks.N-1
        new_state = {}
        for k, v in state.items():
            if k.startswith("features.0."):
                new_state[k.replace("features.0.", "patch_embed.")] = v
            elif k.startswith("features."):
                parts = k.split(".")
                idx = int(parts[1]) - 1
                new_state["blocks." + ".".join([str(idx)] + parts[2:])] = v
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        print(f"RepViT-M1.0 loaded | missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x):
        x = self.patch_embed(x)
        c3 = c4 = c5 = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.C3_IDX:
                c3 = x
            elif i == self.C4_IDX:
                c4 = x
            elif i == self.C5_IDX:
                c5 = x
        return c3, c4, c5  # (112,45,80), (224,23,40), (448,11,20)
