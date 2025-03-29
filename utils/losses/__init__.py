"""
File: utils/losses/__init__.py
Description:  Loss Initialization Script 
Author: Kevin Ferreira
Date: 18 December 2024
"""

# -----------------------------------------------------------------------------
# This script provides a function to initialize different loss functions for  
# medical image segmentation tasks.
#
# Supported Losses:
# - 'cross_entropy': A Cross Entropy Loss.
# - 'dice_loss': A Dice Loss function.
# - 'focal_loss': A Focal Loss function.
# - 'mix_loss': A combination of Dice Loss, CrossEntropy Loss, and Topological Loss.
# - 'dice_topo_loss': A combination of Dice Loss and Topological Loss.
# - 'ce_topo_loss': A combination of CrossEntropy Loss and Topological Loss.
# -----------------------------------------------------------------------------

from .mix_loss import MixLoss, DiceTopoLoss, CeTopoLoss, CeDiceLoss
from .focal_loss import FocalLoss
from .dice_loss import DiceLoss
from torch.nn import CrossEntropyLoss
from .topological_loss_maxpool import TopologicalLossMaxPool
#from .topological_loss_paper import TopologicalLossPaper

def init_loss(loss_name, reduction="mean"):
    """
    Initializes and returns the specified loss function for segmentation tasks.

    Args:
        loss_name (str): The name of the loss function to initialize. Options are:
                         'cross_entropy', 'dice_loss', 'focal_loss', 'mix_loss', 
                         'dice_topo_loss', 'ce_topo_loss'.
        reduction (str): Specifies the reduction to apply to the output ('mean', 'sum', 'none').
    Returns:
        loss (nn.Module): The initialized loss function based on the provided configuration.
    """
    # Initialize CrossEntropy Loss
    if loss_name.lower() == 'cross_entropy':
        return CrossEntropyLoss(reduction=reduction)

    # Initialize Dice Loss
    elif loss_name.lower() == 'dice_loss':
        return DiceLoss(reduction=reduction)

    # Initialize Focal Loss
    elif loss_name.lower() == 'focal_loss':
        return FocalLoss(reduction=reduction)
    
    # Initialize Topo Loss 
    elif loss_name.lower() == 'topo_loss':
        loss = TopologicalLossMaxPool(reduction=reduction)
        return loss
    
    # Initialize Topo Loss Paper
    # elif loss_name.lower() == 'topo_loss_paper':
    #     loss = TopologicalLossPaper(reduction=reduction)
    #     return loss

    # Initialize MixLoss (Dice + CrossEntropy + TopoLoss)
    elif loss_name.lower() == 'mix_loss':
        loss = MixLoss(reduction=reduction)
        return loss

    # Initialize DiceTopoLoss (Dice + TopoLoss)
    elif loss_name.lower() == 'dice_topo_loss':
        loss = DiceTopoLoss(reduction=reduction)
        return loss

    # Initialize CeTopoLoss (CrossEntropy + TopoLoss)
    elif loss_name.lower() == 'ce_topo_loss':
        loss = CeTopoLoss(reduction=reduction)
        return loss
    
    # Initialize CeTopoLoss (CrossEntropy + Dice)
    elif loss_name.lower() == 'ce_dice_loss':
        loss = CeDiceLoss(reduction=reduction)
        return loss

    else:
        raise ValueError(f"Loss function '{loss_name}' is not supported. Available options are 'cross_entropy', 'dice_loss', 'focal_loss', 'mix_loss', 'dice_topo_loss', or 'ce_topo_loss'.")
