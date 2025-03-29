"""
File: utils/losses/topological_loss.py
Description: Implementation of Topological Loss, which incorporates persistent homology
             for topological structure preservation in segmentation tasks.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import time 

class TopologicalLoss(nn.Module):
    """
    Topological Loss for segmentation tasks with an emphasis on preserving topological structures.

    Args:
        max_iter (int): Number of iterations for erosion/dilation.
        kernel_size (int): Size of the kernel for erosion/dilation operations.
        weight_components (float): Weight for topological components in the loss.
        reduction (str): Specifies the reduction to apply to the output ('mean', 'sum', 'none').
        with_dfs (bool): Use Depth First Search (DFS) for component finding if True, otherwise use Connected Component Labeling (CCL).
    """
    def __init__(self, max_iter=5, kernel_size=3, weight_components=1.0, reduction='none', with_dfs = True):
        super(TopologicalLoss, self).__init__()
        self.max_iter = max_iter
        self.kernel_size = kernel_size
        self.weight_components = weight_components
        self.reduction = reduction
        self.with_dfs = with_dfs
        self.time = {"topological_loss": 0, "dfs":0, "ccl": 0, "loop_comp": 0, "match_component":0}
        self.images = {}
    
    def forward(self, predictions, targets):
        # Compute the predicted maps using softmax
        prob_map = F.softmax(predictions, dim=1) 
        predicted_classes = prob_map.argmax(dim=1)  

        # Compute binary maps for each class
        binary_pred_maps = self._to_binary_maps(predicted_classes, predictions.size(1))
        binary_target_maps = self._to_binary_maps(targets, predictions.size(1))

        # Compute barcodes for predictions and targets
        barcodes_pred = self.compute_barcodes(binary_pred_maps)
        barcodes_target = self.compute_barcodes(binary_target_maps)

        # Compute loss
        losses = self.topological_loss(barcodes_pred, barcodes_target, binary_pred_maps.device)

        # Apply the specified reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none': 
            return losses
        else:
            return losses, barcodes_pred, self.images
    
    def _to_binary_maps(self, class_map, num_classes):
        """
        Converts a class map to binary maps for each class.

        Args:
            class_map (torch.Tensor): Class predictions (batch_size, H, W).
            num_classes (int): Number of classes.

        Returns:
            torch.Tensor: Binary maps (batch_size, num_classes, H, W).
        """
        return (class_map.unsqueeze(1) == torch.arange(num_classes, device=class_map.device).view(1, -1, 1, 1)).float()

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
        Computes barcodes for binary maps using iterative erosion and dilation.

        Args:
            binary_maps (torch.Tensor): Binary maps (batch_size, num_classes, H, W).

        Returns:
            dict: Barcodes for each class in each batch.
        """
        barcodes = {b: {c: {} for c in range(binary_maps.size(1))} for b in range(binary_maps.size(0))}
        previous_components = {b: {c: {} for c in range(binary_maps.size(1))} for b in range(binary_maps.size(0))}
        
        current_maps = binary_maps.clone()
        # First image
        self.images[self.max_iter//2] = current_maps
        barcodes, previous_components = self._compute_barcodes_iterative(current_maps, barcodes, previous_components, self.max_iter//2)
        
        # Erosion step
        for t in range(0, self.max_iter//2):
            current_maps = self.erode(current_maps)
            self.images[self.max_iter//2+t+1] = current_maps
            barcodes, previous_components = self._compute_barcodes_iterative(current_maps, barcodes, previous_components, self.max_iter//2+t+1)
        
        # Dilatation step
        current_maps = binary_maps.clone()
        for t in range(0, self.max_iter//2):
            current_maps = self.dilate(current_maps)
            self.images[self.max_iter//2-t-1] = current_maps
            barcodes, previous_components = self._compute_barcodes_iterative(current_maps, barcodes, previous_components, self.max_iter//2-t-1)
        return barcodes

    def _compute_barcodes_iterative(self, binary_maps, barcodes, previous_components, iteration):
        """
        Iteratively computes barcodes during erosion/dilation.

        Args:
            binary_maps (torch.Tensor): Current binary maps.
            barcodes (dict): Existing barcodes.
            previous_components (dict): Previously identified components.
            iteration (int): Current iteration index.

        Returns:
            tuple: Updated barcodes and previous components.
        """
        for b in range(binary_maps.size(0)):
            for c in range(binary_maps.size(1)):
                components = self.find_components_dfs(binary_maps[b, c]) if self.with_dfs else self.find_components_ccl(binary_maps[b, c])
                for component in components:
                    match = self.match_component(component, previous_components[b][c])
                    if match is None:
                        # New component
                        key = len(previous_components[b][c])
                        barcodes[b][c][key] = torch.zeros(self.max_iter, device=binary_maps.device)
                        barcodes[b][c][key][iteration] = len(component)
                        previous_components[b][c][key] = set(component)
                    else:
                        # Existing component
                        barcodes[b][c][match][iteration] = len(component)
        return barcodes, previous_components

    def match_component(self, comp, previous_components):
        """
        Matches a component to previously identified components.

        Args:
            comp (set): Current component pixels.
            previous_components (dict): Previously identified components.

        Returns:
            int: ID of the matching component, or None if no match is found.
        """
        t1 = time.time()
        comp_set = set(comp)
        best_id, best_union = None, 0
        for comp_id, prev_set in previous_components.items():
            union = len(comp_set & prev_set)
            if union>best_union and union > (0.5 * min(len(comp_set), len(prev_set))):
                best_id, best_union = comp_id, union
        t2 = time.time()
        self.time["match_component"] += t2-t1
        return best_id

    def find_components_dfs(self, binary_map):
        """
        Finds connected components in a binary map using Depth First Search (DFS).

        Args:
            binary_map (torch.Tensor): Binary map (H, W).

        Returns:
            list: List of connected components.
        """
        visited = torch.zeros_like(binary_map, dtype=torch.bool)
        components = []
        def dfs(x, y):
            t1d = time.time()
            stack = [(x, y)]
            comp = []
            while stack:
                cx, cy = stack.pop()
                if not (0 <= cx < binary_map.size(0) and 0 <= cy < binary_map.size(1)) or visited[cx, cy] or binary_map[cx, cy] == 0:
                    continue
                visited[cx, cy] = True
                comp.append((cx, cy))
                stack.extend([(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)])
            t2d = time.time()
            self.time["dfs"] += t2d-t1d
            return comp
        t1 = time.time()

        while True:
            indices = torch.nonzero((binary_map == 1) & (~visited), as_tuple=False)
            if indices.size(0) == 0:
                break
            i, j = indices[0]
            component = dfs(i.item(), j.item())
            if component:
                components.append(component)
        components.sort(key=lambda x: len(x), reverse=True)
        t2 = time.time()
        self.time["loop_comp"] += t2-t1
        return components
    
    def find_components_ccl(self, binary_map):
        """
        Finds connected components in a binary map using Connecting Component Labelling (CCL).

        Args:
            binary_map (torch.Tensor): Binary map (H, W).

        Returns:
            list: List of connected components.
        """
        t1 = time.time()
        labels = torch.zeros_like(binary_map, dtype=torch.int32) 
        current_label = 0  
        equivalences = {}
        for i, j in torch.nonzero(binary_map == 1, as_tuple=False):
            neighbors = []
            if i > 0 and labels[i - 1, j] > 0:
                neighbors.append(labels[i - 1, j])
            if j > 0 and labels[i, j - 1] > 0:
                neighbors.append(labels[i, j - 1])
            if len(neighbors)>0:
                min_label = min(neighbors)
                labels[i, j] = min_label
                for label in neighbors:
                    if label != min_label:
                        equivalences[label.item()] = min_label.item()
            else:
                current_label += 1
                labels[i, j] = current_label

        def find(label):
            while label in equivalences:
                label = equivalences[label]
            return label
        label_set = set()
        for key in equivalences:
            equivalences[key] = find(equivalences[key])
        for key in equivalences:
            labels[labels == key] = equivalences[key]
            label_set.add(equivalences[key])

        t2 = time.time()
        components = []
        for i in label_set:
            comp = [tuple(x) for x in torch.nonzero(labels == i, as_tuple=False).tolist()]
            components.append(comp)
        components.sort(key=lambda x: len(x), reverse=True)
        t3 = time.time()
        self.time["ccl"] += t2-t1
        self.time["loop_comp"] += t3-t2
        return components


    def topological_loss(self, barcode_pred, barcode_target, device):
        """
        Calculates the topological loss based on barcode distances.

        Args:
            barcode_pred (dict): Barcodes for predictions.
            barcode_target (dict): Barcodes for targets.
            device (torch.device): Device to use for loss calculation.

        Returns:
            torch.Tensor: Topological loss values.
        """
        t1 = time.time()
        losses = torch.zeros(len(barcode_pred), device=device)
        for b, bbp in barcode_pred.items():
            for c, bcp in bbp.items():
                for i, comp_p in bcp.items():
                    comp_t = barcode_target[b][c].get(i, None)
                    losses[b] += torch.abs(comp_p/comp_p.sum() - comp_t/comp_t.sum()).sum() if comp_t is not None else torch.abs(comp_p/comp_p.sum()).sum()
        t2 = time.time()
        self.time["topological_loss"] += t2-t1
        return losses
    