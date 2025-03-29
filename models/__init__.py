"""
File: models/__init__.py
Description: Model Initialization Script for UNet Variants
Author: Kevin Ferreira
Date: 16 December 2024
"""

# -----------------------------------------------------------------------------
# This script provides a function to initialize different variants of the UNet 
# model for medical image segmentation tasks. The function `init_model` allows 
# flexibility in selecting different model architectures, skip connection types, 
# bottleneck types, and other hyperparameters.
#
# Supported Models:
# - 'unet_2d': A standard 2D UNet model.
# - 'unet_3d': A 3D UNet model designed to work with volumetric data.
# - 'unet_2_5d': A hybrid 2.5D UNet model that uses 2D convolutions with context 
#               from multiple slices at once.
# - 'unet_3d_to_2d': A specialized model for converting 3D volumetric input into 
#                    2D representations for segmentation tasks.
# -----------------------------------------------------------------------------

from models.unet_2_5d import UNet2_5D 
from models.unet_3d_to_2d import UNet3DTo2D 
from models.unet import UNet

def init_model(name_model, skip_connection_type, bottleneck_type, num_classes, slice_of_interest, 
               filter_1 = 16, depth = 6, dropout = [0.1], n_slices = 9):
    """
    Initialize the specified UNet model variant with the given configuration parameters.
    
    Args:
        name_model (str): The name of the model architecture to initialize. 
                           Options are 'unet_2d', 'unet_3d', 'unet_2_5d', 'unet_3d_to_2d'.
        skip_connection_type (str): The type of skip connection to use ('attention_softmax', 
                                      'attention_sigmoid', 'mean', 'mid').
        bottleneck_type (str): The type of bottleneck mechanism ('attention' or 'transformer').
        num_classes (int): The number of output classes for the segmentation task.
        slice_of_interest (int): The slice index (for 2D or 3D context).
        filter_1 (int): The number of filters for the first convolution layer (default 16).
        depth (int): The depth of the UNet (number of downsampling layers).
        dropout (list or float): The dropout probability for regularization (default [0.1]).
        n_slices (int): Number of slices to use for 2.5D or 3D models (default 9).
        
    Returns:
        model (nn.Module): The initialized model based on the provided configuration.
    """
    # List of supported names
    supported_models           = ['unet_2d', 'unet_3d', 'unet_2_5d', 'unet_3d_to_2d']
    supported_skip_connections = ['none','attention_softmax', 'attention_sigmoid',
                                  'mean', 'attention_softmax_mean', 'attention_softmax_mean', 
                                  'mid', 'attention_sigmoid_mid', 'attention_sigmoid_mid']
    supported_bottlenecks      = ['none', 'attention', 'transformer']
    
    # Check validity of arguments
    if name_model.lower() not in supported_models:
        raise ValueError(f"Unsupported model name '{name_model}'. Supported models are: {', '.join(supported_models)}")
    if skip_connection_type not in supported_skip_connections:
        raise ValueError(f"Unsupported skip connection type '{skip_connection_type}'. Supported types are: {', '.join(supported_skip_connections)}")
    if bottleneck_type not in supported_bottlenecks:
        raise ValueError(f"Unsupported bottleneck type '{bottleneck_type}'. Supported types are: {', '.join(supported_bottlenecks)}")
    

    if name_model.lower() == 'unet_2d':
        return UNet(num_classes, filter_1=filter_1, depth=depth, dropout=dropout, 
                    slice_of_interest=slice_of_interest, n_slices=n_slices, 
                    skip_connection_type=skip_connection_type, bottleneck_type=bottleneck_type, is_3d=False)
    
    elif name_model.lower() == 'unet_3d':
        return UNet(num_classes, filter_1=filter_1, depth=depth, dropout=dropout, 
                    slice_of_interest=slice_of_interest, n_slices=n_slices, 
                    skip_connection_type=skip_connection_type, bottleneck_type=bottleneck_type, is_3d=True)
    
    elif name_model.lower() == 'unet_2_5d':
        return UNet2_5D(num_classes, filter_1 = filter_1, depth = depth, dropout = dropout, 
                        slice_of_interest = slice_of_interest, n_slices = n_slices, use_transformer= "transformer" in bottleneck_type)
    
    elif 'unet_3d_to_2d' in  name_model.lower():
        return UNet3DTo2D(num_classes, filter_1 = filter_1, depth = depth, dropout = dropout, 
                          slice_of_interest = slice_of_interest, n_slices = n_slices, 
                          skip_connection_type = skip_connection_type, bottleneck_type = bottleneck_type)
