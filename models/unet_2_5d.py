"""
File: models/unet_2_5d.py
Description: 2.5D U-Net architecture with attention, residual, and transformer blocks.
Author: Kevin Ferreira
Date: 16 September 2024
"""

import torch.nn as nn
from models.helper_blocks import ResBlock, UpBlock, AttentionSlice, TransformerBlock

class UNet2_5D(nn.Module):
    """
    UNet 2.5D architecture for multi-slice 3D image processing. It uses 2D convolutions
    on individual slices with attention mechanisms and residual connections to improve performance.

    Args:
        num_classes (int): The number of output classes for segmentation.
        filter_1 (int): The number of filters in the first convolution layer.
        depth (int): The depth of the network (number of blocks in the encoder).
        dropout (list): List of dropout probabilities for each block.
        slice_of_interest (int): The slice index around which 3D information is utilized.
        n_slices (int): The number of slices used in the model (typically 3 or 5).
    """
    def __init__(self, num_classes, filter_1=16, depth=6, dropout=[0.3], slice_of_interest=4, n_slices=3, use_transformer=False):
        super(UNet2_5D, self).__init__()

        # Initialize model parameters and configuration
        self.slice_of_interest = slice_of_interest
        self.n_slices          = n_slices
        self.checkpoint        = self._initialize_checkpoint()

        # Define dropout layers dynamically based on depth
        if len(dropout) == 2:
            dropout = [dropout[0] + i * (dropout[1] - dropout[0]) / (depth - 1) for i in range(depth)]
        else:
            dropout = [dropout[0] for i in range(depth)]

        # Create encoder blocks with dynamic in/out channels
        params_block = self._create_encoder_blocks(filter_1, depth, dropout)

        # Encoder (downward path)
        self.downs = nn.ModuleList([DownBlock(params_block, dropout) for _ in range(n_slices)])
        
        # Attention layers to focus on relevant features
        self.attentions = nn.ModuleList([AttentionSlice(params_block[-1][1]) for _ in range(n_slices)])
        
        # Optionally include Transformer Block
        if self.use_transformer:
            self.transformer = TransformerBlock(params_block[-1][1] * n_slices, 16, params_block[-1][1] * 4, dropout=dropout[-1])
        else:
            self.transformer = None

        # Decoder (upward path) to recover the image resolution
        self.decoder = nn.ModuleList()
        self.decoder.append(UpBlock(params_block[-1][1]*2+params_block[-2][1], params_block[-2][0], dropout_prob=dropout[-2], scale_factor = params_block[-2][2]))
        for d in range(3, depth): 
            self.decoder.append(UpBlock(params_block[-d][1]+params_block[-d][1], params_block[-d][0], dropout_prob=dropout[-d], scale_factor = params_block[-d][2]))
        self.decoder.append(UpBlock(params_block[0][1]+params_block[0][1], 3, dropout_prob=dropout[0], scale_factor = params_block[0][2]))        

        # Final convolution to get the output segmentation map
        self.final_conv = nn.Conv2d(3, num_classes, kernel_size=1)

        # Initialize network weights
        self._initialize_weights()
    
    def _initialize_checkpoint(self):
        """
        Initializes a checkpoint dictionary to save model state and training configurations.
        """
        return {
            'name': 'UNet2_5D',
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
        params_block = []
        in_channels, out_channels = 1, filter_1
        for _ in range(depth - 1):
            params_block.append((in_channels, out_channels, 2))
            in_channels, out_channels = out_channels, 2 * out_channels
        params_block.append((in_channels, out_channels, 1))
        return params_block
    
    def _initialize_weights(self):
        for m in self.modules():

            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_in  = m.in_channels * m.kernel_size[0] * m.kernel_size[1]  
                fan_out = m.out_channels * m.kernel_size[0] * m.kernel_size[1]  
                if fan_in > fan_out:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                fan_in  = m.in_channels * m.kernel_size[0] * m.kernel_size[1]* m.kernel_size[2]
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
        x_encoded= []
        for i in range(self.n_slices):
            out = x[:, :,:,:, self.slice_of_interest-self.n_slices//2+i]
            out = self.downs[i](out)
            x_encoded.append(out)
        outs = []
        for i in range(self.n_slices):
            outs.append(self.attentions[i](x_encoded[self.n_slices//2][-1], x_encoded[i][-1]))
        out = sum(outs)
        if self.transformer is not None:
            out = self.transformer(out)
        for i, layer in enumerate(self.decoder):
            out  = layer(out, x_encoded[self.n_slices//2][-i-2])
        return self.final_conv(out)
    
class DownBlock(nn.Module):
    """
    Encoder block for downsampling the input image. Each block consists of a series of
    residual blocks with convolutional strides and optional dropout.

    Args:
        params_block (list): List of tuples representing the configuration of each residual block.
        dropout (list): Dropout values for each residual block.
    """
    def __init__(self, params_block, dropout):
        super(DownBlock, self).__init__()
        self.encoder = nn.ModuleList()  
        for d in range(len(params_block)):
            self.encoder.append(ResBlock(params_block[d][0], params_block[d][1], stride=params_block[d][2], dropout_prob=dropout[d]))
    def forward(self, x):
        encoded = []
        for layer in self.encoder:
            x = layer(x)
            encoded.append(x)
        return encoded
 

