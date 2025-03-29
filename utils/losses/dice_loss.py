"""
File: utils/losses/dice_loss.py
Description: Implementation of Dice Loss, a loss function commonly used for segmentation tasks 
             to measure overlap between predicted and target segmentation maps.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.

    Args:
        epsilon (float): Small constant to avoid division by zero.
        reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
    """
    def __init__(self, epsilon=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        
    def forward(self, logits, targets):
        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

        # Calculate the intersection and union
        dims = (2, 3)  # Height and width dimensions
        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs + targets_one_hot, dims)

        # Compute the Dice coefficient
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)

        # Compute the Dice loss
        loss = 1 - dice.mean(dim=1)

        # Apply the specified reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss




