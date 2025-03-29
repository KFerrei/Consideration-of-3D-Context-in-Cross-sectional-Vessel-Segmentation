"""
File: models/unet_3d_to_2d.py
Description: 3D-to-2D U-Net architecture with attention, residual, and transformer blocks for multi-slice image processing.
Author: Kevin Ferreira
Date: 16 December 2024
"""

import torch
import torch.nn as nn
from models.helper_blocks import ResBlock, UpBlock, AttentionBlockHybrid, TransformerBlock, SelfAttention

class UNet3DTo2D(nn.Module):
    """
    U-Net architecture designed to process multi-slice 3D images and convert the 3D representations into 2D outputs. 
    The model leverages 3D convolutions, attention mechanisms, residual blocks, and transformer blocks to process
    multi-slice input data and extract meaningful features for segmentation tasks.
    
    Args:
        num_classes (int): The number of output classes for segmentation.
        filter_1 (int): The number of filters in the first convolution layer.
        depth (int): The depth of the network (number of blocks in the encoder).
        dropout (list): List of dropout probabilities for each block.
        slice_of_interest (int): The slice index around which 3D information is utilized.
        n_slices (int): The number of slices used in the model (typically 3 or 5).
        skip_connection_type (str): Type of skip connection ('attention', 'trainable', 'mid').
        bottleneck_type (str): Type of bottleneck to use ('transformers' or 'attention').
    """
    def __init__(self, num_classes, filter_1 = 16, depth = 6, dropout = [0.3], 
                 slice_of_interest = 4, n_slices = 9, skip_connection_type = 'none', bottleneck_type = 'none'):
        super(UNet3DTo2D, self).__init__()
        # Initialize model parameters
        self.n_slices = n_slices
        self.slice_of_interest = slice_of_interest
        self.checkpoint = self._initialize_checkpoint()

        # Define dropout layers dynamically based on depth
        if len(dropout) == 2:
            dropout = [dropout[0] + i * (dropout[1] - dropout[0]) / (depth - 1) for i in range(depth)]
        else:
            dropout = [dropout[0] for i in range(depth)]

        # Create encoder blocks with dynamic in/out channels
        params_block = self._create_encoder_blocks(filter_1, depth)

        # Encoder
        self.encoder = nn.ModuleList()  
        for d in range(depth-1):
            self.encoder.append(ResBlock(params_block[d][0], params_block[d][1], stride=params_block[d][2], dropout_prob=dropout[d], is_3d = True))

        # Bottleneck
        self.bottleneck = None
        if "transformer" in bottleneck_type:
            print("With attention transformer bottleneck")
            self.bottleneck = nn.Sequential(
                SelfAttention(params_block[-1][0]),
                TransformerBlock(params_block[-1][0]*2, 16, params_block[-1][0]*4, dropout=dropout[-1])
                )
        elif "attention" in bottleneck_type:
            print("With attention bottleneck")
            self.bottleneck = SelfAttention(params_block[-1][0])
        else:
            self.bottleneck = ResBlock(params_block[-1][0], params_block[-1][1], 
                                       stride=params_block[-1][2], dropout_prob=dropout[-1])
            
        # Skip connections
        sizes = [n_slices if (depth - i) > (n_slices ** 0.5 + 2)
                 else int(n_slices / 2**(i - depth + int(n_slices**0.5 + 2) + 1)) + 1
                 for i in range(depth)]
        self.skip_connections = nn.ModuleList([SkipConnection(sizes[d], skip_connection_type, 
                       params_block[d][1], params_block[d][1], params_block[d][1]//2, dropout[d]) for d in range(depth-2)])
        self.skip_connections.append(SkipConnection(sizes[-2], skip_connection_type, 
                       params_block[-1][1], params_block[-2][1], params_block[-2][1]//2, dropout[d]))

        # Decoder
        self.decoder = nn.ModuleList()
        self.decoder.append(UpBlock(params_block[-1][1]+params_block[-2][1], params_block[-2][0], dropout_prob=dropout[-2], stride = params_block[-2][2]))
        for d in range(3, depth): 
            self.decoder.append(UpBlock(params_block[-d][1]+params_block[-d][1], params_block[-d][0], dropout_prob=dropout[-d], stride = params_block[-d][2]))
        self.decoder.append(UpBlock(params_block[0][1]+params_block[0][1], 3, dropout_prob=dropout[0], stride = params_block[0][2]))        
        
        self.final_conv = nn.Conv2d(3, num_classes, kernel_size = 1)
        self._initialize_weights()

    def _initialize_checkpoint(self):
        """
        Initializes a checkpoint dictionary to save model state and training configurations.
        """
        return {
            'name': 'UNet3DTo2D',
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
    
    def forward(self, x):
        x = x[:,:,:, :, int(self.slice_of_interest - self.n_slices//2) : (self.slice_of_interest + self.n_slices//2+ self.n_slices%2)]
        encoded = []
        for layer in self.encoder:
            x = layer(x)
            encoded.append(x)
        # Decoding 
        out = self.bottleneck(encoded[-1].squeeze(4))
        for i, layer in enumerate(self.decoder):
            skip = self.skip_connections[-i-1](out, encoded[-i-1])
            out  = layer(out, skip)
        # Final layer 
        out = self.final_conv(out)
        return out

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

class SkipConnection(nn.Module):
    """
    Skip connection module that allows different types of skip connection mechanisms such as 
    attention-based, mean-based, or mid-slice methods. 

    Args:
        size (int): The size of the input (number of slices).
        skip_connection (str): Type of skip connection ('attention', 'mean', 'mid', or 'trainable').
        F_g (int): Number of channels in the first input tensor.
        F_l (int): Number of channels in the second input tensor.
        F_int (int): Intermediate number of channels for attention.
        dropout_prob (float): Dropout probability to be used in attention mechanism.
    """
    def __init__(self, size, skip_connection, F_g, F_l, F_int, dropout_prob):
        super(SkipConnection, self).__init__()
        self.attention = None
        if "attention" in skip_connection:
            activation_type = "softmax" if "softmax" in skip_connection else "sigmoid"
            print(f"With Attention {activation_type}")
            self.attention = AttentionBlockHybrid(F_g, F_l, F_int, dropout_prob, activation_type)
        
        if "mean" in skip_connection:
            self.type_output = WeightedMean(size)
        elif "mid" in skip_connection:
            self.type_output = MidSlice(size)
        else:
            self.type_output = TrainableWeightedMean(size)
    def forward(self, g, x):
        if self.attention:
            x = self.attention(g, x)
        out = self.type_output(x)
        return out

class TrainableWeightedMean(nn.Module):
    """
    A trainable weighted mean skip connection mechanism where the weights are learned during training.
    """
    def __init__(self, size):
        super(TrainableWeightedMean, self).__init__()
        print("Trainable Weighted Mean")
        self.weights = nn.Parameter(torch.ones(size), requires_grad=True)
    def forward(self, out):
        norm_weights  = torch.softmax(self.weights, dim=0)
        weighted_mean = torch.sum(out * norm_weights, dim=4)
        return weighted_mean

class WeightedMean(nn.Module):
    """
    A fixed weighted mean skip connection mechanism, where the weights are predefined and do not change during training.
    The weights are calculated such that the center slice gets the highest weight.
    """
    def __init__(self, size):
        super(WeightedMean, self).__init__()
        print("Weighted Mean")
        self.size    = int(size)
        self.midsize = size // 2
        self.is_even = (size % 2 == 0)
        self.weights = self.calculate_weights()

    def calculate_weights(self):
        weights       = torch.zeros(self.size)
        weights[self.midsize] = 1.0
        if self.is_even:
            weights[self.midsize - 1] = 1.0  
        factor = 0.5
        for offset in range(1, self.midsize + 1):
            if self.midsize + offset < self.size:
                weights[self.midsize + offset] = (factor ** offset)
            if self.midsize - offset >= 0:
                weights[self.midsize - offset] = (factor ** offset)
        weights /= weights.sum()
        return weights
    
    def forward(self, out):
        weights       = self.weights.to(out.device).view(1, 1, 1, 1, -1)
        weighted_mean = (out * weights).sum(dim=-1)
        return weighted_mean

class MidSlice(nn.Module):
    """
    A mid-slice skip connection mechanism that extracts the slice from the middle of the input tensor.
    """
    def __init__(self, size):
        super(MidSlice, self).__init__()
        print("Mid Slice")
        self.midsize  = int(size//2)
    def forward(self, out):
        out = out[:,:,:, :, self.midsize]
        return out
    