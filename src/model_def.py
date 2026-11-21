import math
import torch
import torch.nn as nn
import torch.nn.functional as F
NCLS = {"instrument": 8, "anatomy": 4}

class MultiLevelHead(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 4, 384, 1, bias=False), nn.BatchNorm2d(384), nn.GELU(),
            nn.Conv2d(384, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class LastLayerHead(nn.Module):
    """Phase 43 anatomy decoder: final-layer tokens only."""

    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, classes, 1),
        )
        self.channels = channels

    def forward(self, x):
        return self.net(x[:, -self.channels:])


class DinoScene(nn.Module):
    """DINOv2 encoder shared by one or more task heads (blocks 3/6/9/12 fused)."""

    def __init__(self, tasks, name="facebook/dinov2-base", head_types=None):
        super().__init__()
        from transformers import Dinov2Model
        self.encoder = Dinov2Model.from_pretrained(name)
        c = self.encoder.config.hidden_size
        depth = self.encoder.config.num_hidden_layers
        self.levels = [depth // 4, depth // 2, 3 * depth // 4, depth]
        head_types = head_types or {}
        self.heads = nn.ModuleDict({
            t: (LastLayerHead(c, NCLS[t]) if head_types.get(t) == "last" else MultiLevelHead(c, NCLS[t]))
            for t in tasks})

    def features(self, x):
        out = self.encoder(pixel_values=x, output_hidden_states=True)
        maps = []
        for i in self.levels:
            t = out.hidden_states[i][:, 1:, :]
            side = math.isqrt(t.shape[1])
            maps.append(t.transpose(1, 2).reshape(t.shape[0], t.shape[2], side, side))
        return torch.cat(maps, 1)

    def forward(self, x, tasks):
        f = self.features(x)
        return {t: F.interpolate(self.heads[t](f), size=x.shape[-2:], mode="bilinear", align_corners=False)
                for t in tasks}


