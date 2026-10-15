import torch
from torch import nn
from torch.nn import functional as F


class IFFPCS(nn.Module):
    def __init__(self, mode="all", gate=True, permute=False):
        super().__init__()
        self.mode, self.use_gate, self.permute = mode, gate, permute
        if mode != "rgb":
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
            self.encoder.requires_grad_(False)
            self.projections = nn.ModuleList([nn.Conv2d(384, 64, 1) for _ in range(3)])
            channels = 192 if mode == "all" else 64
            for k, projection in enumerate(self.projections):
                projection.requires_grad_(mode == "all" or k == (2 if mode == "deep" else 1))
        else:
            channels = 3
        self.gate = nn.Sequential(nn.Conv2d(3, 16, 1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        self.gate.requires_grad_(gate)
        self.decoder = nn.Sequential(nn.Conv2d(channels, 96, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(96, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))

    def forward(self, a, b):
        size = a.shape[-2:]
        if self.mode == "rgb":
            difference = (b - a).abs()
            distance = difference.mean(1, keepdim=True).expand(-1, 3, -1, -1)
        else:
            self.encoder.eval()
            pad = (0, (-size[1]) % 14, 0, (-size[0]) % 14)
            mean = a.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = a.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            with torch.no_grad():
                fa = self.encoder.get_intermediate_layers(F.pad((a - mean) / std, pad), n=[2, 5, 11], reshape=True)
                fb = self.encoder.get_intermediate_layers(F.pad((b - mean) / std, pad), n=[2, 5, 11], reshape=True)
            maps = [1 - F.cosine_similarity(x, y, dim=1, eps=1e-6)[:, None] for x, y in zip(fa, fb)]
            distance = F.interpolate(torch.cat(maps, 1), size=size, mode="bilinear", align_corners=False)
            selected = [0, 1, 2] if self.mode == "all" else [2 if self.mode == "deep" else 1]
            differences = [self.projections[k](fb[k] - fa[k]) for k in selected]
            difference = F.interpolate(torch.cat(differences, 1), size=maps[0].shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(distance).sigmoid() if self.use_gate else torch.zeros_like(distance[:, :1])
        if self.permute:
            g = g.flatten(2).roll(g.shape[-2] * g.shape[-1] // 3, -1).reshape_as(g)
        attenuation = F.interpolate(1 - g, size=difference.shape[-2:], mode="bilinear", align_corners=False)
        logits = F.interpolate(self.decoder(difference * attenuation), size=size, mode="bilinear", align_corners=False)
        return logits, g, distance


def boundary(x):
    kernels = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
    dx = F.conv2d(x, kernels, padding=1)
    dy = F.conv2d(x, kernels.transpose(-1, -2), padding=1)
    return (dx.abs() + dy.abs()).clamp(0, 1)


def objective(logits, g, distance, y, proxy, args):
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * y).sum((1, 2, 3)) + 1) / (probability.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    loss = F.binary_cross_entropy_with_logits(logits, y) + dice.mean()
    if not args.no_boundary:
        loss = loss + 0.2 * F.l1_loss(boundary(probability), boundary(y))
    if not args.no_pseudo and args.gate and proxy.any():
        loss = loss - args.pseudo_weight * g[proxy].clamp_min(1e-6).log().mean()
    if not args.no_preserve and args.gate:
        for i in range(len(y)):
            changed = y[i] > 0.5
            if changed.any():
                keep = changed & (distance[i, 2:3] > torch.quantile(distance[i, 2:3][changed], 0.5))
                if keep.any():
                    loss = loss - args.preserve_weight * (1 - g[i][keep]).clamp_min(1e-6).log().mean() / len(y)
    return loss
