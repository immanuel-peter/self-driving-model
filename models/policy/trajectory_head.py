from typing import Optional, Dict
import torch
import torch.nn as nn

class EasyBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class TrajectoryPolicy(nn.Module):
    def __init__(self, horizon: int = 8, context_dim: int = 0, backbone_dim: int = 512):
        super().__init__()
        self.horizon = horizon
        self.backbone = EasyBackbone(in_channels=3, out_dim=backbone_dim)

        head_in_dim = backbone_dim + (context_dim if context_dim > 0 else 0)
        hidden = 512
        self.head_wp = nn.Sequential(
            nn.Linear(head_in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, horizon * 2),
        )
        self.head_spd = nn.Sequential(
            nn.Linear(head_in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, horizon),
        )

    def forward(self, image: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feat = self.backbone(image)
        if context is not None:
            x = torch.cat([feat, context], dim=1)
        else:
            x = feat
        wp = self.head_wp(x).view(-1, self.horizon, 2)
        spd = self.head_spd(x).view(-1, self.horizon)
        return {"waypoints": wp, "speed": spd}