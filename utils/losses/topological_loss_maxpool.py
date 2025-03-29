"""
File: utils/losses/topological_loss_maxpool.py
Description: This file defines the TopologicalLossMaxPool class, a custom loss function that incorporates topological data analysis into deep learning models. 
             It leverages iterative dilation and erosion operations to compute a topological loss based on the persistence diagrams (barcodes) of predicted and target maps. 
Author: Kevin Ferreira
Date: 18 December 2024
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class TopologicalLossMaxPool(nn.Module):
    def __init__(self, max_iter=5, kernel_size=5, reduction='mean'):
        """
        Initializes the TopologicalLossGPU class.
        
        Args:
            max_iter (int): Maximum number of iterations for topological operations (erosion and dilation).
            kernel_size (int): Size of the kernel for dilation and erosion operations.
            reduction (str): Specifies the method for reducing the loss ('mean', 'sum', or 'none').
        """
        super(TopologicalLossMaxPool, self).__init__()
        self.max_iter = max_iter
        self.kernel_size = kernel_size
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Computes the topological loss between predictions and targets.

        Args:
            predictions (torch.Tensor): Model's output with shape (batch_size, num_classes, H, W).
            targets (torch.Tensor): Ground truth labels with shape (batch_size, H, W).
        
        Returns:
            torch.Tensor: Calculated topological loss based on persistence barcodes of predictions and targets.
        """
        # Convert predictions to binary maps using Gumbel softmax and targets to one-hot encoding
        binary_pred_maps   = F.gumbel_softmax(predictions, hard=True, dim=1)
        binary_target_maps = F.one_hot(targets, num_classes=predictions.size(1)).float().permute(0, 3, 1, 2)

        # Compute barcodes for predictions and targets
        barcodes_pred   = self.compute_barcodes(binary_pred_maps)
        barcodes_target = self.compute_barcodes(binary_target_maps)

        # Compute loss
        losses = self.topological_loss(barcodes_pred, barcodes_target, binary_pred_maps.device)

        # Apply reduction method and return the loss
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none': 
            return losses
        else:
            return losses, barcodes_pred
    
    def erode(self, image):
        """
        Performs erosion on binary maps.

        Args:
            image (torch.Tensor): Binary maps (batch_size, num_classes, H, W).

        Returns:
            torch.Tensor: Eroded binary maps.
        """
        kernel = torch.ones((image.size(1), 1, self.kernel_size, self.kernel_size), dtype=image.dtype, device=image.device)
        eroded = F.conv2d(image, kernel, padding=self.kernel_size // 2, groups=image.size(1))
        return (eroded == self.kernel_size ** 2).float()
    
    def dilate(self, image):
        """
        Performs dilation on binary maps.

        Args:
            image (torch.Tensor): Binary maps (batch_size, num_classes, H, W).

        Returns:
            torch.Tensor: Dilated binary maps.
        """
        kernel = torch.ones((image.size(1), 1, self.kernel_size, self.kernel_size), dtype=image.dtype, device=image.device)
        dilated = F.conv2d(image, kernel, padding=self.kernel_size // 2, groups=image.size(1))
        return (dilated > 0).float()

    def compute_barcodes(self, binary_maps):
        """
        Computes persistence barcodes for the given binary maps through iterative dilation and erosion.

        Args:
            binary_maps (torch.Tensor): Tensor of binary maps with shape (batch_size, num_classes, H, W).
        
        Returns:
            dict: A dictionary containing persistence barcodes for each component of the maps.
        """
        B, C, H, W = binary_maps.shape
        barcodes = {b: {c: {} for c in range(C)} for b in range(B)}
        previous_components = {b: {c: {} for c in range(C)} for b in range(B)}
        
        current_maps = binary_maps.clone()
        # First image
        barcodes, previous_components = self._compute_barcodes_iterative(current_maps, barcodes, previous_components, self.max_iter//2)
        # Erosion step
        for t in range(0, self.max_iter//2):
            current_maps = self.erode(current_maps)
            barcodes, previous_components = self._compute_barcodes_iterative(current_maps, barcodes, previous_components, self.max_iter//2+t+1)
        
        # Dilatation step
        current_maps = binary_maps.clone()
        for t in range(0, self.max_iter//2):
            current_maps = self.dilate(current_maps)
            barcodes, previous_components = self._compute_barcodes_iterative(current_maps, barcodes, previous_components, self.max_iter//2-t-1)
        return barcodes
    

    def _compute_barcodes_iterative(self, binary_maps, barcodes, previous_components, iteration):
        """
        Iteratively computes persistence barcodes for binary maps through erosion or dilation.
        
        Args:
            binary_maps (torch.Tensor): Tensor of binary maps, shape (batch_size, num_classes, H, W).
            barcodes (dict): Existing persistence barcodes.
            previous_components (dict): Previously identified components.
            iteration (int): Current iteration index.
        
        Returns:
            dict: Updated barcodes and previous components after the iteration.
        """
        new_components = self.find_components_max_pool(binary_maps)
        for b, components_class in new_components.items():
            for c, components in components_class.items():
                for i, component in components.items():
                    match = self.match_component(component, previous_components[b][c])
                    if match is None:
                        # New component
                        key = len(previous_components[b][c])
                        barcodes[b][c][key] = torch.zeros(self.max_iter, device=binary_maps.device)
                        barcodes[b][c][key][iteration] = component.sum()
                        previous_components[b][c][key] = component
                    else:
                        # Existing component
                        barcodes[b][c][match][iteration] = component.sum()
        return barcodes, previous_components

    def match_component(self, comp, previous_components):
        """
        Matches the current component to a previously identified one based on overlap.

        Args:
            comp (torch.Tensor): Current component to match.
            previous_components (dict): Dictionary of previously identified components.
        
        Returns:
            int or None: Index of the matched component or None if no match is found.
        """
        best_id, best_union = None, 0
        for comp_id, prev in previous_components.items():
            union = (comp * prev).sum()
            if union>best_union: 
                best_id, best_union = comp_id, union
        return best_id

    def find_components_max_pool(self, binary_maps):
        """
        Finds connected components in binary maps using max pooling.
        
        Args:
            binary_maps (torch.Tensor): Binary maps, shape (batch_size, num_classes, H, W).
        
        Returns:
            dict: Dictionary of components identified using max pooling.
        """
        new_components = {b: {c: {} for c in range(binary_maps.size(1))} for b in range(binary_maps.size(0))}
        B, C, H, W = binary_maps.shape

        prev_labels = torch.arange(H*W, device=binary_maps.device).view(H, W).float().repeat(B, C, 1, 1) * binary_maps
        while True:
            next_labels = F.max_pool2d(prev_labels, kernel_size=3, stride=1, padding=1) *  binary_maps
            if torch.abs(next_labels - prev_labels).sum() == 0.:
                break
            prev_labels = next_labels

        # Find unique labels and their indices
        for b in range(B):
            for c in range(C):
                components = []
                labels = torch.unique(next_labels[b, c]).clone()
                for label in labels:
                    if label != 0:
                        mask = (next_labels[b, c] == label).float()
                        component_binary = mask * next_labels[b, c] * 1/torch.max(next_labels[b, c])
                        components.append(component_binary)
                components.sort(key=lambda x: x.sum(), reverse=True)
                new_label = 1
                for comp in components:
                    new_components[b][c][new_label] = comp
                    new_label += 1
        return new_components

    def topological_loss(self, barcode_pred, barcode_target, device):
        """
        Calculates the topological loss based on barcode distances.

        Args:
            barcode_pred (dict): Predicted persistence barcodes.
            barcode_target (dict): Ground truth persistence barcodes.
            device (torch.device): The device on which the tensors are located.
        
        Returns:
            torch.Tensor: Calculated loss based on the topological distance between predicted and target barcodes.
        """
        losses = torch.zeros(len(barcode_pred), device=device)
        for b, bbp in barcode_pred.items():
            for c, bcp in bbp.items():
                for i, comp_p in bcp.items():
                    comp_t = barcode_target[b][c].get(i, None)
                    comp_p_max = comp_p.max()
                    if comp_t is None:
                        losses[b] = losses[b] + torch.sum(((comp_p/comp_p_max) ** 2))
                    else:
                        losses[b] = losses[b] + torch.sum(((comp_t/comp_p_max - comp_p/comp_p_max) ** 2))
        return losses
    