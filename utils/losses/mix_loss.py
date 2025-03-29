"""
File: utils/losses/mix_loss.py
Description: Implements mixed loss functions combining Dice, Cross-Entropy, and Topological losses.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import torch.nn as nn
from .dice_loss import DiceLoss
from .topological_loss_maxpool import TopologicalLossMaxPool
 
class MixLoss(nn.Module):
    """
    A mixed loss function that combines Dice Loss, Cross-Entropy Loss, and optionally Topological Loss.

    Args:
        compute_topo (bool): Flag to decide whether to include the topological loss in the final loss computation.

    Methods:
        forward(logits, targets): Computes the loss, including Dice Loss, Cross-Entropy Loss, and optionally Topological Loss.
    """
    def __init__(self, reduction):
        super(MixLoss, self).__init__()
        self.dice_loss = DiceLoss(reduction=reduction)
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.topo_loss = TopologicalLossMaxPool(reduction=reduction)
        self.compute_topo = False
    def forward(self, logits, targets):
        loss = 0
        loss += self.dice_loss(logits, targets)
        loss += self.cross_entropy(logits, targets)
        if self.compute_topo:
            loss += self.topo_loss(logits, targets)
        return loss

class DiceTopoLoss(nn.Module):
    """
    A loss function that combines Dice Loss and Topological Loss.

    Args:
        compute_topo (bool): Flag to decide whether to include the topological loss in the final loss computation.

    Methods:
        forward(logits, targets): Computes the loss, including Dice Loss and optionally Topological Loss.
    """
    def __init__(self, reduction):
        super(DiceTopoLoss, self).__init__()
        self.dice_loss = DiceLoss(reduction=reduction)
        self.topo_loss = TopologicalLossMaxPool(reduction=reduction)
        self.compute_topo = False

    def forward(self, logits, targets):
        loss = 0
        loss += self.dice_loss(logits, targets)
        if self.compute_topo:
            loss += self.topo_loss(logits, targets)
        return loss

class CeTopoLoss(nn.Module):
    """
    A loss function that combines Cross-Entropy Loss and Topological Loss.

    Args:
        compute_topo (bool): Flag to decide whether to include the topological loss in the final loss computation.

    Methods:
        forward(logits, targets): Computes the loss, including Cross-Entropy Loss and optionally Topological Loss.
    """
    def __init__(self, reduction):
        super(CeTopoLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.topo_loss = TopologicalLossMaxPool(reduction=reduction)
        self.compute_topo = False

    def forward(self, logits, targets):
        loss = 0
        loss += self.cross_entropy(logits, targets)
        if self.compute_topo:
            loss += self.topo_loss(logits, targets)
        return loss
    
class CeDiceLoss(nn.Module):
    """
    A loss function that combines Cross-Entropy Loss and Dice Loss.

    Methods:
        forward(logits, targets): Computes the loss, including Cross-Entropy Loss and Dice Loss.
    """
    def __init__(self, reduction):
        super(CeDiceLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.dice_loss = DiceLoss(reduction=reduction)

    def forward(self, logits, targets):
        loss = 0
        loss = self.cross_entropy(logits, targets) + self.dice_loss(logits, targets)
        return loss