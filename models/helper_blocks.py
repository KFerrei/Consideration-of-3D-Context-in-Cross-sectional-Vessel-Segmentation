"""
File: models/building_blocks.py
Description: Core building blocks for 2D and 3D deep learning models, including convolutional, residual, 
             attention, and transformer-based modules.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# DoubleConv Module
class DoubleConv(nn.Module):
    """
    Double convolutional block consisting of two consecutive convolutions, 
    each followed by BatchNorm, Dropout, and PReLU.
    Supports both 2D and 3D convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dropout_prob (float): Dropout probability for regularization. Default is 0.1.
        is_3d (bool): Flag to use 3D convolutions. Default is False (2D).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_prob=0.1, is_3d=False):
        super(DoubleConv, self).__init__()
        Conv      = nn.Conv3d if is_3d else nn.Conv2d
        BatchNorm = nn.BatchNorm3d if is_3d else nn.BatchNorm2d

        self.conv1    = Conv(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn1      = BatchNorm(out_channels)
        self.dropout1 = nn.Dropout3d(dropout_prob) if is_3d else nn.Dropout(dropout_prob)
        self.prelu1   = nn.PReLU()

        self.conv2    = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn2      = BatchNorm(out_channels)
        self.dropout2 = nn.Dropout3d(dropout_prob) if is_3d else nn.Dropout(dropout_prob)
        self.prelu2   = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.dropout1(self.bn1(self.conv1(x))))
        x = self.prelu2(self.dropout2(self.bn2(self.conv2(x))))
        return x

# ConvStride Module
class ConvStride(nn.Module):
    """
    Convolutional block with a stride, followed by BatchNorm, Dropout, and PReLU.
    Supports both 2D and 3D convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution. Default is 2.
        dropout_prob (float): Dropout probability for regularization. Default is 0.1.
        is_3d (bool): Flag to use 3D convolutions. Default is False (2D).
    """
    def __init__(self, in_channels, out_channels, stride=2, dropout_prob=0.1, is_3d=False):
        super(ConvStride, self).__init__()
        Conv      = nn.Conv3d if is_3d else nn.Conv2d
        BatchNorm = nn.BatchNorm3d if is_3d else nn.BatchNorm2d

        self.conv_stride = Conv(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn          = BatchNorm(out_channels)
        self.dropout     = nn.Dropout3d(dropout_prob) if is_3d else nn.Dropout(dropout_prob)
        self.prelu       = nn.PReLU() 

    def forward(self, x):
        out = self.prelu(self.dropout(self.bn(self.conv_stride(x))))
        return out

# Residual Block with Convolutional Shortcut Module
class ResBlock(nn.Module):
    """
    Residual block with optional down-sampling (via stride), including a convolutional shortcut.
    Supports both 2D and 3D convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution. Default is 2.
        dropout_prob (float): Dropout probability for regularization. Default is 0.1.
        is_3d (bool): Flag to use 3D convolutions. Default is False (2D).
    """
    def __init__(self, in_channels, out_channels, stride=2, dropout_prob=0.1, is_3d=False):
        super(ResBlock, self).__init__()
        self.conv_stride   = ConvStride(in_channels, out_channels, stride=stride, dropout_prob=dropout_prob, is_3d=is_3d)
        self.double_conv   = DoubleConv(out_channels, out_channels, dropout_prob=dropout_prob, is_3d=is_3d)
        self.residual_conv = ConvStride(in_channels, out_channels, stride=stride, dropout_prob=dropout_prob, is_3d=is_3d) 
    
    def forward(self, x):
        residual = self.residual_conv(x)
        out      = self.double_conv(self.conv_stride(x))
        out     += residual
        return out

# Upsampling Block
class UpBlock(nn.Module):
    """
    Upsampling block using transposed convolutions with skip connections.
    Supports both 2D and 3D convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution. Default is 2.
        dropout_prob (float): Dropout probability for regularization. Default is 0.1.
        is_3d (bool): Flag to use 3D convolutions. Default is False (2D).
        padding (int): Padding reactification only when is_3d = True.
    """
    def __init__(self, in_channels, out_channels, stride=2, dropout_prob=0.1, is_3d=False, padding=1):
        super(UpBlock, self).__init__()
        ConvTranspose = nn.ConvTranspose3d if is_3d else nn.ConvTranspose2d
        BatchNorm     = nn.BatchNorm3d if is_3d else nn.BatchNorm2d

        padding       = (stride//2, stride//2, max(0, stride//2 - padding)) if is_3d else stride//2
        self.upconv   = ConvTranspose(in_channels, out_channels, kernel_size=3, 
                                      stride=stride, padding=1, output_padding=padding)
        self.bn1      = BatchNorm(out_channels)
        self.dropout1 = nn.Dropout3d(dropout_prob) if is_3d else nn.Dropout(dropout_prob)
        self.prelu1   = nn.PReLU()

        self.conv     = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) if is_3d else \
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2      = BatchNorm(out_channels)
        self.dropout2 = nn.Dropout3d(dropout_prob) if is_3d else nn.Dropout(dropout_prob)
        self.prelu2   = nn.PReLU()

    def forward(self, x, skip_connection):
        x    = torch.cat((x, skip_connection), dim=1)
        x    = self.prelu1(self.dropout1(self.bn1(self.upconv(x))))
        out  = self.prelu2(self.dropout2(self.bn2(self.conv(x))))
        out += x
        return out

# Attention Hybrid Block Module
class AttentionBlockHybrid(nn.Module):
    """
    Hybrid Attention Block combining attention mechanisms for both global and local feature maps from skip connections.
    Supports both 2D and 3D convolutions.
    
    Args:
        F_g (int): Number of feature maps for global input.
        F_l (int): Number of feature maps for local input.
        F_int (int): Number of intermediate channels.
        dropout_prob (float): Dropout probability for regularization.
        activation_type (str): Type of activation function ('sigmoid' or 'softmax').
    """
    def __init__(self, F_g, F_l, F_int, dropout_prob, activation_type='sigmoid'):
        super(AttentionBlockHybrid, self).__init__()
        # Global feature transformation
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
            nn.Dropout2d(dropout_prob),
        )
        # Local feature transformation
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
            nn.Dropout3d(dropout_prob),
        )
        # Attention map (psi) generation
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Dropout3d(dropout_prob),
        )
        self.activation_type = activation_type
        self.prelu = nn.PReLU()
        self.activation_type = activation_type
        # Output transformation
        self.out = nn.Sequential(
            nn.Conv3d(F_l, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_l),
            nn.Dropout3d(dropout_prob),
        )
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        g1 = g1.unsqueeze(4)
        psi = self.prelu(g1 + x1)
        psi = self.psi(psi) 
        if self.activation_type == 'sigmoid':
            psi = torch.sigmoid(psi) 
        elif self.activation_type == 'softmax':
            psi = nn.functional.softmax(psi, dim=1)
        return self.out(x * psi)

# Attention Block Module
class AttentionBlock(nn.Module):
    """
    Attention block applies attention mechanisms to input tensors and focuses on important features.
    Supports both 2D and 3D convolutions.
    
    Args:
        F_g (int): Number of input channels of the first input.
        F_l (int): Number of input channels of the second input.
        F_int (int): Number of intermediate channels. 
        dropout_prob (float): Dropout probability for regularization.
        activation_type (string): Activation type after the attention mechanism.
        is_3d (bool): Flag to indicate if the input is 3D. Default is False (2D).
    """
    def __init__(self, F_g, F_l, F_int, dropout_prob, activation_type='sigmoid', is_3d=False):
        super(AttentionBlock, self).__init__()
        Conv          = nn.Conv3d if is_3d else nn.Conv2d
        BatchNorm     = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
        
        self.W_g = nn.Sequential(
            Conv(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm(F_int),
            nn.Dropout3d(dropout_prob) if is_3d else nn.Dropout(dropout_prob),
        )
        self.W_x = nn.Sequential(
            Conv(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm(F_int),
            nn.Dropout3d(dropout_prob) if is_3d else nn.Dropout(dropout_prob),
        )
        self.psi = nn.Sequential(
            Conv(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm(1),
            nn.Dropout3d(dropout_prob) if is_3d else nn.Dropout(dropout_prob),
        )
        self.activation_type = activation_type
        self.prelu = nn.PReLU()
        self.activation_type = activation_type

        self.out = nn.Sequential(
            Conv(F_l, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm(F_l),
            nn.Dropout3d(dropout_prob) if is_3d else nn.Dropout(dropout_prob),
        )
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.prelu(g1 + x1)
        psi = self.psi(psi) 
        if self.activation_type == 'sigmoid':
            psi = torch.sigmoid(psi) 
        elif self.activation_type == 'softmax':
            psi = nn.functional.softmax(psi, dim=1)
        return self.out(x * psi)

# Attention Slice Module
class AttentionSlice(nn.Module):
    """
    Attention Slice applies attention mechanisms to input tensors and focuses on important features.
    Supports both 2D and 3D convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        dropout_prob (float): Dropout probability for regularization.
        inter_channels (int, optional): Number of intermediate channels. Defaults to in_channels // 2.
        is_3d (bool): Flag to indicate if the input is 3D. Default is False (2D).
    """
    def __init__(self, in_channels, dropout_prob=0.1, inter_channels=None, is_3d=False):
        super(AttentionSlice, self).__init__()
        
        # Set default intermediate channels if not provided
        if inter_channels is None:
            inter_channels = in_channels // 2
        if inter_channels == 0:
            inter_channels = 1

        self.is_3d = is_3d

        self.theta = self._conv_layer(in_channels, inter_channels, dropout_prob)
        self.phi = self._conv_layer(in_channels, inter_channels, dropout_prob)
        self.g = self._conv_layer(in_channels, inter_channels, dropout_prob)
        self.W = self._conv_layer(inter_channels, in_channels, dropout_prob)
    def _conv_layer(self, in_channels, out_channels, dropout_prob):
        """
        Helper function to create convolution layers with batch normalization and dropout.
        """
        if self.is_3d:
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(out_channels),
                nn.Dropout3d(dropout_prob)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(dropout_prob)
            )
    def forward(self, x1, x2):
        if self.is_3d:
            batch_size, C, H, W, D= x1.size()
        else:
            batch_size, C, H, W = x1.size()
            D = 1 
        theta_x1 = self.theta(x1).view(batch_size, -1, D * H * W).permute(0, 2, 1)
        phi_x2 = self.phi(x2).view(batch_size, -1, D * H * W)
        g_x2 = self.g(x2).view(batch_size, -1, D * H * W).permute(0, 2, 1)

        # Attention mechanism: calculate the attention map
        attention_map = torch.matmul(theta_x1, phi_x2)
        attention_map = F.softmax(attention_map, dim=-1)
        # Apply attention to the g tensor
        out = torch.matmul(attention_map, g_x2).permute(0, 2, 1).contiguous()

        # Reshape and apply the W convolution
        out = out.view(batch_size, -1, H, W, D).squeeze(4)
        out = self.W(out)
        # Concatenate the output with the original x1 tensor (skip connection)
        out = torch.cat((out, x1), dim=1)
        return out

# Self Attention Module
class SelfAttention(nn.Module):
    """
    Self Attention Slice applies attention mechanisms to input tensors and focuses on important features.
    Supports both 2D and 3D convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        dropout_prob (float): Dropout probability for regularization.
        inter_channels (int, optional): Number of intermediate channels. Defaults to in_channels // 2.
        is_3d (bool): Flag to indicate if the input is 3D. Default is False (2D).
    """
    def __init__(self, in_channels, dropout_prob=0.1, inter_channels=None, is_3d=False):
        super(SelfAttention, self).__init__()
        self.attention = AttentionSlice(in_channels, dropout_prob=dropout_prob, inter_channels=inter_channels, is_3d=is_3d)
    def forward(self, x):
        return self.attention(x, x)
    
# Transformer Block Module
class TransformerBlock(nn.Module):
    """
    Transformer-based block with multi-head self-attention and feedforward layers.
    Supports both 2D and 3D inputs.
    
    Args:
        dim (int): Dimension of the input.
        num_heads (int): Number of attention heads.
        mlp_dim (int): Dimension of the feedforward layer.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        s = x.shape 
        x = x.flatten(2).permute(2, 0, 1)
        
        attn_output, _ = self.attention(x, x, x)
        x  = x + attn_output  
        x  = self.norm1(x)

        mlp_output = self.mlp(x)
        x  = x + mlp_output
        x  = self.norm2(x) 

        return x.permute(1, 2, 0).view(*s)
    
