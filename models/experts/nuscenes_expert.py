import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TNet(nn.Module):
    """Transformation network for PointNet"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: [B, N, k] -> [B, k, N]
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        x = x.view(-1, 1024)  # [B, 1024]
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # [B, k*k]
        
        iden = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k).repeat(x.size(0), 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNet(nn.Module):
    """PointNet architecture for point cloud processing"""
    def __init__(self, output_dim=1024, use_tnet=True):
        super().__init__()
        self.use_tnet = use_tnet
        
        if use_tnet:
            self.input_transform = TNet(k=3)
            self.feature_transform = TNet(k=64)
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: [B, N, 3] -> [B, 3, N]
        x = x.transpose(2, 1)
        B, D, N = x.size()
        
        if self.use_tnet:
            trans_input = self.input_transform(x.transpose(2, 1))  # [B, 3, 3]
            x = torch.bmm(trans_input, x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        if self.use_tnet:
            trans_feat = self.feature_transform(x.transpose(2, 1))  # [B, 64, 64]
            x = torch.bmm(trans_feat, x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        x = x.view(-1, 1024)  # [B, 1024]
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class NuScenesExpert(nn.Module):
    def __init__(self,
                 image_backbone=None,
                 lidar_backbone=None,
                 fusion: str = 'concat',
                 num_queries: int = 100,
                 use_lidar: bool = False,
                 use_tnet: bool = False,
                 bbox_dim: int = 7):
        super().__init__()

        if image_backbone is None:
            resnet18 = models.resnet18(pretrained=True)
            self.image_backbone = nn.Sequential(*list(resnet18.children())[:-1])
            self.image_projection = nn.Linear(512, 256)
        else:
            self.image_backbone = image_backbone
            self.image_projection = nn.Identity()

        # LIDAR encoder (PointNet)
        self.use_lidar = use_lidar
        if self.use_lidar:
            if lidar_backbone is None:
                self.lidar_backbone = PointNet(output_dim=256, use_tnet=use_tnet)
            else:
                self.lidar_backbone = lidar_backbone
        else:
            self.lidar_backbone = None

        # Fusion and multi-query detection head
        self.fusion_type = fusion  # 'concat' or 'sum'
        # If image-only or fusion=sum, feature dim is 256; if concat with lidar, 512
        if self.use_lidar and self.fusion_type == 'concat':
            fusion_dim = 512
        else:
            fusion_dim = 256
        self.num_queries = num_queries
        self.bbox_dim = bbox_dim

        self.query_embed = nn.Embedding(num_queries, fusion_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.class_head = nn.Linear(128, 10)
        self.bbox_head = nn.Linear(128, self.bbox_dim)

    def forward(self, batch):
        image = batch['image']         # [B, 3, H, W]
        lidar = batch.get('lidar', None)         # [B, P, 3]

        # Image branch
        img_feat = self.image_backbone(image)       # [B, 512, 1, 1]
        img_feat = img_feat.view(img_feat.size(0), -1)  # [B, 512]
        img_feat = self.image_projection(img_feat)      # [B, 256]

        # Lidar branch (optional): directly use the padded tensor; PointNet expects [B, N, 3]
        if self.use_lidar and self.lidar_backbone is not None and lidar is not None:
            lidar_feat = self.lidar_backbone(lidar)    # [B, 256]
        else:
            lidar_feat = None

        # 3) Fuse scene features
        if self.use_lidar and lidar_feat is not None:
            if self.fusion_type == 'concat':
                fused_feat = torch.cat([img_feat, lidar_feat], dim=-1)  # [B, 512]
            else:
                fused_feat = img_feat + lidar_feat                      # [B, 256]
        else:
            fused_feat = img_feat  # image-only

        # 4) Process multiple queries for multi-object detection
        B = fused_feat.size(0)
        
        # fused_feat: [B, fusion_dim] -> [B, num_queries, fusion_dim]
        fused_feat = fused_feat.unsqueeze(1).expand(B, self.num_queries, -1)
        
        # query_embed.weight: [num_queries, fusion_dim] -> [B, num_queries, fusion_dim]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        x = self.decoder(fused_feat + queries)  # [B, num_queries, 128]
        
        cls_logits = self.class_head(x)  # [B, num_queries, 10]
        bbox_preds = self.bbox_head(x)   # [B, num_queries, bbox_dim]
        
        return {
            'class_logits': cls_logits,
            'bbox_preds':   bbox_preds
        }
