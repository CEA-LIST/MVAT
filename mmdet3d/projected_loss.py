import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

@weighted_loss
def multi_view_projection_loss(pred_3d_bboxes, 
                               gt_2d_bboxes, 
                               lidar2img, 
                               img_shapes, 
                               object_ids=None):
    """
    Computes the Multi-View 2D Projection Loss (GIoU) defined in MVAT.
    
    Args:
        pred_3d_bboxes (torch.Tensor): Predicted 3D boxes of shape (N, 7) or (N, 9).
            Format: [x, y, z, l, w, h, yaw, ...]
            NOTE: These should be repeated/expanded to match the number of 2D views. 
            If Object A is seen in 5 frames, pred_3d_bboxes should contain Object A 5 times.
        gt_2d_bboxes (torch.Tensor): Ground truth 2D boxes of shape (N, 4).
            Format: [x1, y1, x2, y2]
        lidar2img (torch.Tensor): Projection matrices of shape (N, 4, 4).
            Transforms points from the 3D box coordinate system (usually LiDAR) to Image.
        img_shapes (torch.Tensor): Image dimensions (N, 2) [height, width] or (N, 3).
            Used to clip projected boxes.
        object_ids (torch.Tensor, optional): Tensor of shape (N,) indicating which 
            unique object each row belongs to. Used to strictly follow Eq. 2 
            (average over views per object first). If None, standard mean is used.

    Returns:
        torch.Tensor: The calculated loss.
    """
    
    # 1. Get 8 corners of the 3D bounding boxes in LiDAR/World coords
    # shape: (N, 8, 3)
    corners_3d = get_corners_from_bboxes_3d(pred_3d_bboxes)

    # 2. Project corners to 2D image plane
    # Add homogeneous coord: (N, 8, 4)
    num_boxes = corners_3d.shape[0]
    ones = torch.ones(num_boxes, 8, 1, device=corners_3d.device)
    corners_3d_hom = torch.cat([corners_3d, ones], dim=2)

    # Apply projection matrix: (N, 4, 4) x (N, 8, 4)^T -> (N, 4, 8) -> (N, 8, 4)
    # We use bmm (batch matrix multiplication)
    corners_img_hom = torch.bmm(lidar2img, corners_3d_hom.permute(0, 2, 1)).permute(0, 2, 1)

    # Perspective division (x/w, y/w)
    # Clamp w to avoid division by zero or negative depths (points behind camera)
    eps = 1e-5
    w = corners_img_hom[..., 2:3].clamp(min=eps)
    x_img = corners_img_hom[..., 0:1] / w
    y_img = corners_img_hom[..., 1:2] / w

    # 3. Get Axis-Aligned 2D Bounding Box from projected corners
    # Shape: (N, 1)
    min_x = torch.min(x_img, dim=1)[0]
    min_y = torch.min(y_img, dim=1)[0]
    max_x = torch.max(x_img, dim=1)[0]
    max_y = torch.max(y_img, dim=1)[0]

    # 4. Clip to image boundaries (Crucial for training stability)
    H, W = img_shapes[:, 0], img_shapes[:, 1]
    min_x = min_x.clamp(min=0, max=W.unsqueeze(1))
    max_x = max_x.clamp(min=0, max=W.unsqueeze(1))
    min_y = min_y.clamp(min=0, max=H.unsqueeze(1))
    max_y = max_y.clamp(min=0, max=H.unsqueeze(1))

    # Form the projected box: [x1, y1, x2, y2]
    pred_2d_bboxes = torch.cat([min_x, min_y, max_x, max_y], dim=1)

    # 5. Calculate GIoU
    loss = 1.0 - calculate_giou(pred_2d_bboxes, gt_2d_bboxes)

    # 6. Aggregation 
    if object_ids is not None:
        # We need to average loss per object_id, then average over objects
        unique_ids = torch.unique(object_ids)
        per_object_losses = []
        for uid in unique_ids:
            mask = (object_ids == uid)
            # "1/|Fj| * sum(...)" part of Eq 2
            per_object_losses.append(loss[mask].mean())
        
        return torch.stack(per_object_losses).mean()
    
    return loss

class MultiView2DProjectionLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, 
                pred_3d_bboxes, 
                gt_2d_bboxes, 
                lidar2img, 
                img_shapes, 
                object_ids=None,
                **kwargs):
        
        loss = multi_view_projection_loss(
            pred_3d_bboxes,
            gt_2d_bboxes,
            lidar2img,
            img_shapes,
            object_ids=object_ids,
            reduction=self.reduction,
            avg_factor=None
        )
        return self.loss_weight * loss


def get_corners_from_bboxes_3d(bboxes_3d):
    """
    Generates 8 corners from 3D bounding boxes.
    Assumes box format: [x, y, z, l, w, h, yaw] (LiDAR coords).
    """
    x, y, z = bboxes_3d[:, 0], bboxes_3d[:, 1], bboxes_3d[:, 2]
    l, w, h = bboxes_3d[:, 3], bboxes_3d[:, 4], bboxes_3d[:, 5]
    yaw = bboxes_3d[:, 6]

    # 1. Create canonical corners (centered at 0,0,0)
    x_corners = l / 2 * torch.tensor([1, 1, -1, -1, 1, 1, -1, -1], device=bboxes_3d.device)
    y_corners = w / 2 * torch.tensor([1, -1, -1, 1, 1, -1, -1, 1], device=bboxes_3d.device)
    z_corners = h / 2 * torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], device=bboxes_3d.device)
    
    # Shape: (N, 3, 8)
    corners = torch.stack([x_corners, y_corners, z_corners], dim=1)

    # 2. Rotate
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    # Rotation matrix around Z axis
    R = torch.stack([c, -s, torch.zeros_like(c),
                     s,  c, torch.zeros_like(c),
                     torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], dim=1).reshape(-1, 3, 3)
    
    corners = torch.bmm(R, corners) # (N, 3, 8)

    # 3. Translate (Add center)
    center = torch.stack([x, y, z], dim=1).unsqueeze(2) # (N, 3, 1)
    corners = corners + center

    return corners.permute(0, 2, 1) # Return (N, 8, 3)

def calculate_giou(pred_boxes, gt_boxes):
    """
    Calculates Generalized IoU for 2D boxes.
    Boxes: [x1, y1, x2, y2]
    """
    # 1. Intersection
    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # 2. Areas
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    union = pred_area + gt_area - intersection + 1e-7
    iou = intersection / union

    # 3. Smallest Enclosing Box (C)
    c_x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    c_y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    c_x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    c_y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])

    c_area = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0) + 1e-7

    # 4. GIoU
    giou = iou - (c_area - union) / c_area
    return giou