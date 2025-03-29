"""
File: utils/losses/focal_loss.py
Description: Implementation of Focal Loss, a loss function designed to address class imbalance 
             in classification tasks by down-weighting well-classified examples.
Author: Kevin Ferreira
Date: 18 December 2024
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.

    Args:
        alpha (list or float): Class weights. If a list, it should have a weight for each class.
        gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
        reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
    """
    def __init__(self, alpha=[0.8, 1.2, 1], gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Compute class probabilities using softmax
        probs = F.softmax(logits, dim=1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float().permute(0, 3, 1, 2)
        targets_probs   = torch.sum(probs * targets_one_hot, dim=1)

        # Calculate probabilities for the target classes
        focal_weight = (1 - targets_probs) ** self.gamma

        # Compute the focal weight
        if isinstance(self.alpha, list):  
            alpha_t = torch.tensor(self.alpha).to(logits.device)[targets]
        else:
            alpha_t = self.alpha
        
        # Compute the focal loss
        loss = -alpha_t * focal_weight * torch.log(targets_probs + 1e-15)  

        # Apply the specified reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: 
            return loss