import torch.nn as nn
import torchvision.models as models

class BDDSegmentationExpert(nn.Module):
    def __init__(self, num_classes=19, pretrained_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        resnet = models.resnet18(pretrained=pretrained_backbone)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.decoder(features)  # [B, C, H, W]
        return logits
