import torch.nn as nn
import torchvision.models as models

class BDDDetectionExpert(nn.Module):
    def __init__(self, num_classes=10, pretrained_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        resnet = models.resnet18(pretrained=pretrained_backbone)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes + 4, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return {
            "class_logits": output[:, :self.num_classes, :, :],
            "bbox_deltas": output[:, self.num_classes:, :, :]
        }

    def predict(self, x):
        output = self.forward(x)
        return {
            "class_probs": output["class_logits"].softmax(dim=1),
            "bbox_deltas": output["bbox_deltas"].sigmoid()
        }
