import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou

class HungarianMatcher(nn.Module):
    """
    Matcher that uses:
      - 2D GIoU for 4-dim boxes
      - BEV GIoU (axis-aligned) for 7-dim 3D boxes
      - else falls back to L1 distance
    """
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou
        assert cost_class!=0 or cost_bbox!=0 or cost_giou!=0

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs['pred_logits']: [B, Q, C]
        outputs['pred_boxes'] : [B, Q, D]   (D==4 or D==7)
        targets[b]['boxes']   : [Ni, D]
        targets[b]['labels']  : [Ni]
        """
        pred_logits = outputs['pred_logits']
        pred_boxes  = outputs['pred_boxes']
        B, Q, C     = pred_logits.shape
        D           = pred_boxes.shape[2]

        indices = []
        for b in range(B):
            # 1) classification cost
            prob       = pred_logits[b].softmax(-1)       # [Q, C]
            tgt_labels = targets[b]['labels']             # [Ni]
            cost_class = -prob[:, tgt_labels]             # [Q, Ni]

            # 2) L1 bbox cost
            pb = pred_boxes[b]                            # [Q, D]
            tb = targets[b]['boxes']                      # [Ni, D]
            cost_bbox = torch.cdist(pb, tb, p=1)          # [Q, Ni]

            # 3) GIoU cost
            if self.cost_giou > 0:
                if D == 4:
                    # standard 2D GIoU on [cx,cy,w,h]
                    ob = box_convert(pb, 'cxcywh', 'xyxy')
                    tb2 = box_convert(tb, 'cxcywh', 'xyxy')
                    cost_giou = -generalized_box_iou(ob, tb2)
                elif D == 7:
                    # Approximate BEV GIoU by dropping z and yaw,
                    # and treating w,l as axis-aligned widths/lengths.
                    # [cx, cy, cz, w, l, h, yaw] -> [x1,y1,x2,y2]
                    x_c, y_c, _, w, l, _, _ = torch.split(pb, 1, dim=1)
                    x1 = x_c - w/2; y1 = y_c - l/2
                    x2 = x_c + w/2; y2 = y_c + l/2
                    pred_bev = torch.cat([x1,y1,x2,y2], dim=1)  # [Q,4]

                    x_c_t, y_c_t, _, w_t, l_t, _, _ = torch.split(tb, 1, dim=1)
                    x1t = x_c_t - w_t/2; y1t = y_c_t - l_t/2
                    x2t = x_c_t + w_t/2; y2t = y_c_t + l_t/2
                    tgt_bev = torch.cat([x1t,y1t,x2t,y2t], dim=1)  # [Ni,4]

                    cost_giou = -generalized_box_iou(pred_bev, tgt_bev)  # [Q, Ni]
                else:
                    cost_giou = torch.zeros_like(cost_bbox)
            else:
                cost_giou = torch.zeros_like(cost_bbox)

            # 4) combine
            C = self.cost_bbox  * cost_bbox \
              + self.cost_class * cost_class \
              + self.cost_giou  * cost_giou
            C = C.cpu()

            # 5) Hungarian assignment
            row_idx, col_idx = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(row_idx, dtype=torch.int64, device=pb.device),
                torch.as_tensor(col_idx, dtype=torch.int64, device=pb.device)
            ))

        return indices
