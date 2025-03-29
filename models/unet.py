"""
File: models/unet.py
Description: U-Net architecture with attention, residual, and transformer blocks supporting both 2D and 3D.
Author: Kevin Ferreira
Date: 16 September 2024
"""

import torch.nn as nn
from models.helper_blocks import ResBlock, UpBlock, TransformerBlock, AttentionBlock, SelfAttention

class UNet(nn.Module):
    """
    U-Net 3D architecture for multi-slice image processing with attention and residual blocks.

    Args:
        num_classes (int): The number of output classes for segmentation.
        filter_1 (int): The number of filters in the first convolution layer.
        depth (int): The depth of the network (number of blocks in the encoder).
        dropout (list): List of dropout probabilities for each block.
        slice_of_interest (int): The slice index around which 3D information is utilized.
        n_slices (int): The number of slices used in the model (typically 3 or 5).
        skip_connection_type (str): Specify the skip connection type (e.g., "attention").
        bottleneck_type (str): Type of bottleneck to use, can be "attention" or "transformer".
        is_3d (bool): Flag to use 3D convolutions. Default is False (2D).
    """
    def __init__(self, num_classes, filter_1=16, depth=6, dropout=[0.3], 
                 slice_of_interest=4, n_slices=9, 
                 skip_connection_type='none', bottleneck_type='none', is_3d=False):
        super(UNet, self).__init__()
        # Initialize model parameters and configuration
        self.slice_of_interest = slice_of_interest
        self.n_slices          = n_slices
        self.is_3d             = is_3d
        self.checkpoint        = self._initialize_checkpoint()

        # Define dropout layers dynamically based on depth
        if len(dropout) == 2:
            dropout = [dropout[0] + i * (dropout[1] - dropout[0]) / (depth - 1) for i in range(depth)]
        else:
            dropout = [dropout[0] for _ in range(depth)]

        # Create encoder blocks with dynamic in/out channels
        params_block = self._create_encoder_blocks(filter_1, depth)
        self.encoder = nn.ModuleList()  
        for d in range(depth-1):
            self.encoder.append(ResBlock(params_block[d][0], params_block[d][1], 
                                         stride=params_block[d][2], dropout_prob=dropout[d], is_3d=is_3d))
        
        # Bottleneck
        self.bottleneck = None
        if "transformer" in bottleneck_type:
            print("With attention transformer bottleneck")
            self.bottleneck = nn.Sequential(
                SelfAttention(params_block[-1][0], is_3d=is_3d),
                TransformerBlock(params_block[-1][0]*2, 16, params_block[-1][0]*4, dropout=dropout[-1])
                )
        elif "attention" in bottleneck_type:
            print("With attention bottleneck")
            self.bottleneck = SelfAttention(params_block[-1][0], is_3d=is_3d)
        else:
            self.bottleneck = ResBlock(params_block[-1][0], params_block[-1][1], 
                                       stride=params_block[-1][2], dropout_prob=dropout[-1], is_3d=is_3d)

        # Skip connections
        self.skip_connections = nn.ModuleList([SkipConnection(skip_connection_type, params_block[d][1], params_block[d][1], 
                                                              params_block[d][1]//2, dropout[d], is_3d=is_3d) for d in range(depth-2)])
        self.skip_connections.append(SkipConnection(skip_connection_type, params_block[-1][1], params_block[-2][1], 
                                                    params_block[-2][1]//2, dropout[d], is_3d=is_3d))

        # Decoder
        self.decoder = nn.ModuleList()
        self.decoder.append(UpBlock(params_block[-1][1]+params_block[-2][1], params_block[-2][0], 
                                    dropout_prob=dropout[-2], stride=params_block[-2][2], is_3d=is_3d, padding=0))
        for d in range(3, depth): 
            self.decoder.append(UpBlock(params_block[-d][1]+params_block[-d][1], params_block[-d][0], 
                                          dropout_prob=dropout[-d], stride = params_block[-d][2], is_3d=is_3d))
        self.decoder.append(UpBlock(params_block[0][1]+params_block[0][1], 3, dropout_prob=dropout[0], 
                                    stride = params_block[0][2], is_3d=is_3d))        
        
        self.final_conv = nn.Conv3d(3, num_classes, kernel_size = 1) if is_3d else nn.Conv2d(3, num_classes, kernel_size = 1)
        self._initialize_weights()

    def _initialize_checkpoint(self):
        """
        Initializes a checkpoint dictionary to save model state and training configurations.
        """
        return {
            'name': 'Unet3D' if self.is_3d else 'Unet2D',
            'model_state_dict': self.state_dict(),
            'EPOCHS': 0,
            'LR': 0,
            'VAL_SIZE': 0,
            'TEST_SIZE': 0,
            'BATCH_SIZE': 0,
            'results': {
                'train_losses': [],
                'val_losses': [],
            }
        }
    
    def _create_encoder_blocks(self, filter_1, depth):
        """
        Creates the blocks for the encoder by defining the in_channels, out_channels,
        and stride for each convolutional layer.

        Args:
            filter_1 (int): Initial filter size.
            depth (int): Depth of the model.
        
        Returns:
            List of tuples representing the configuration for each block.
        """
        in_channels, out_channels = 1, filter_1
        params_block = []
        for d in range(depth-1):
            stride = 2 if (depth-d) <= self.n_slices**0.5+2 else 1
            params_block.append((in_channels, out_channels, stride))
            in_channels, out_channels = out_channels, 2*out_channels
        params_block.append((params_block[depth-2][1], 2*params_block[depth-2][1], 1))
        return params_block
    
    def _initialize_weights(self):
        for m in self.modules():

            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]  
                fan_out = m.out_channels * m.kernel_size[0] * m.kernel_size[1]  
                if fan_in > fan_out:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]* m.kernel_size[2]
                fan_out = m.out_channels * m.kernel_size[0] * m.kernel_size[1]* m.kernel_size[2]
                if fan_in > fan_out:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.is_3d:
            x = x[:,:,:, :, int(self.slice_of_interest - self.n_slices//2) : (self.slice_of_interest + self.n_slices//2+ self.n_slices%2)]
        else:
            x = x[:,:,:, :,int(self.slice_of_interest)]
        encoded = []
        for layer in self.encoder:
            x = layer(x)
            encoded.append(x)
        out = self.bottleneck(encoded[-1])
        for i, layer in enumerate(self.decoder):
            skip = self.skip_connections[-i-1](out, encoded[-i-1])
            out  = layer(out, skip)
        out = self.final_conv(out)
        if self.is_3d:
            out = out[:,:,:, :, self.slice_of_interest]            
        return out

class SkipConnection(nn.Module):
    """
    Skip connection layer with optional attention mechanism.

    Args:
        skip_connection_type (str): Specify the skip connection type (e.g., "attention").
        F_g (int): Number of input channels of the first input.
        F_l (int): Number of input channels of the second input.
        F_int (int): Number of intermediate channels.
        dropout_prob (float): Dropout probability for regularization.        
    """
    def __init__(self, skip_connection_type, F_g, F_l, F_int, dropout_prob, is_3d):
        super(SkipConnection, self).__init__()
        self.attention = None
        if "attention" in skip_connection_type:
            activation_type = "softmax" if "softmax" in skip_connection_type else "sigmoid"
            print(f"With Attention {activation_type}")
            self.attention = AttentionBlock(F_g, F_l, F_int, dropout_prob, activation_type, is_3d=is_3d)
    def forward(self, g, x):
        if self.attention:
            x = self.attention(g, x)
        out = x
        return out
