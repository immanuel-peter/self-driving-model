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
        
        # Initialize as identity transformation
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
            x = torch.bmm(trans_input, x)  # Apply transformation
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        if self.use_tnet:
            trans_feat = self.feature_transform(x.transpose(2, 1))  # [B, 64, 64]
            x = torch.bmm(trans_feat, x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling for permutation invariance
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        x = x.view(-1, 1024)  # [B, 1024]
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class NuScenesExpert(nn.Module):
    def __init__(self, image_backbone=None, lidar_backbone=None, fusion='concat', num_queries=100):
        super().__init__()

        # Image encoder (ResNet-18)
        if image_backbone is None:
            resnet18 = models.resnet18(pretrained=True)
            # Remove the final classification layer
            self.image_backbone = nn.Sequential(*list(resnet18.children())[:-1])
            self.image_projection = nn.Linear(512, 256)  # ResNet-18 outputs 512 features
        else:
            self.image_backbone = image_backbone
            self.image_projection = nn.Identity()

        # LIDAR encoder (PointNet)
        if lidar_backbone is None:
            self.lidar_backbone = PointNet(output_dim=256, use_tnet=True)
        else:
            self.lidar_backbone = lidar_backbone

        # Fusion and multi-query detection head
        self.fusion_type = fusion  # 'concat' or 'sum'
        fusion_dim = 512 if fusion == 'concat' else 256
        self.num_queries = num_queries
        
        # --- KEY CHANGE: Learnable query embeddings ---
        # These queries will specialize in finding different types of objects
        self.query_embed = nn.Embedding(num_queries, fusion_dim)
        
        # Decoder to process each query with scene context
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Detection heads (now operate on each query)
        self.class_head = nn.Linear(128, 10)  # 10 classes
        self.bbox_head = nn.Linear(128, 7)    # 7 box params

    def forward(self, batch):
        image = batch['image']         # [B, 3, H, W]
        lidar = batch['lidar']         # [B, P, 3]   ← padded tensor

        # 1) Image branch
        img_feat = self.image_backbone(image)       # [B, 512, 1, 1]
        img_feat = img_feat.view(img_feat.size(0), -1)  # [B, 512]
        img_feat = self.image_projection(img_feat)      # [B, 256]

        # 2) Lidar branch: directly use the padded tensor
        #    PointNet expects [B, N, 3]
        lidar_feat = self.lidar_backbone(lidar)    # [B, 256]

        # 3) Fuse scene features
        if self.fusion_type == 'concat':
            fused_feat = torch.cat([img_feat, lidar_feat], dim=-1)  # [B, 512]
        else:
            fused_feat = img_feat + lidar_feat                      # [B, 256]

        # 4) Process multiple queries for multi-object detection
        B = fused_feat.size(0)
        
        # Expand scene features to match number of queries
        # fused_feat: [B, fusion_dim] -> [B, num_queries, fusion_dim]
        fused_feat = fused_feat.unsqueeze(1).expand(B, self.num_queries, -1)
        
        # Add learnable query embeddings to give each query a unique "personality"
        # query_embed.weight: [num_queries, fusion_dim] -> [B, num_queries, fusion_dim]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Combine scene context with query embeddings
        x = self.decoder(fused_feat + queries)  # [B, num_queries, 128]
        
        # Generate predictions for each query
        cls_logits = self.class_head(x)  # [B, num_queries, 10]
        bbox_preds = self.bbox_head(x)   # [B, num_queries, 7]
        
        return {
            'class_logits': cls_logits,
            'bbox_preds':   bbox_preds
        }
