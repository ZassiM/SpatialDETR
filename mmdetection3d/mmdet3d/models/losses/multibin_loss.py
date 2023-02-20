# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def multibin_loss(pred_orientations, gt_orientations, num_dir_bins=4):
    """Multi-Bin Loss.

    Args:
        pred_orientations(torch.Tensor): Predicted local vector
            orientation in [axis_cls, head_cls, sin, cos] format.
            shape (N, num_dir_bins * 4)
        gt_orientations(torch.Tensor): Corresponding gt bboxes,
            shape (N, num_dir_bins * 2).
        num_dir_bins(int, optional): Number of bins to encode
            direction angle.
            Defaults: 4.

    Return:
        torch.Tensor: Loss tensor.
    """
    cls_losses = 0
    reg_losses = 0
    reg_cnt = 0
    for i in range(num_dir_bins):
        # bin cls loss
        cls_ce_loss = F.cross_entropy(
            pred_orientations[:, (i * 2):(i * 2 + 2)],
            gt_orientations[:, i].long(),
            reduction='mean')
        # regression loss
        valid_mask_i = (gt_orientations[:, i] == 1)
        cls_losses += cls_ce_loss
        if valid_mask_i.sum() > 0:
            start = num_dir_bins * 2 + i * 2
            end = start + 2
            pred_offset = F.normalize(pred_orientations[valid_mask_i,
                                                        start:end])
            gt_offset_sin = torch.sin(gt_orientations[valid_mask_i,
                                                      num_dir_bins + i])
            gt_offset_cos = torch.cos(gt_orientations[valid_mask_i,
                                                      num_dir_bins + i])
            reg_loss = \
                F.l1_loss(pred_offset[:, 0], gt_offset_sin,
                          reduction='none') + \
                F.l1_loss(pred_offset[:, 1], gt_offset_cos,
                          reduction='none')

            reg_losses += reg_loss.sum()
            reg_cnt += valid_mask_i.sum()

        return cls_losses / num_dir_bins + reg_losses / reg_cnt


@LOSSES.register_module()
class MultiBinLoss(nn.Module):
    """Multi-Bin Loss for orientation.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        loss_weight (float, optional): The weight of loss. Defaults
            to 1.0.
    """

    def __init__(self, reduction='none', loss_weight=1.0):
        super(MultiBinLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, num_dir_bins, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            num_dir_bins (int): Number of bins to encode direction angle.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * multibin_loss(
            pred, target, num_dir_bins=num_dir_bins, reduction=reduction)
        return loss
