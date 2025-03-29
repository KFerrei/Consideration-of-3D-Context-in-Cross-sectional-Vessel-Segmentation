"""
Full assembly baseline UNet
@Author: Kevin Ferreira
@Date: 10/09/2024
@Version: 1.0
"""
import torch.nn as nn
from models.helper_blocks import ResBlock, UpBlock

class UNet2D(nn.Module):
    def __init__(self, num_classes, slice_of_interest):
        super(UNet2D, self).__init__()
        self.checkpoint = {'name': 'UNet2D',
                           'model_state_dict': self.state_dict(),
                           'EPOCHS': 0,
                           'LR': 0,
                           'VAL_SIZE': 0,
                           'TEST_SIZE': 0,
                           'results': {
                               'train_losses': [],
                               'val_losses': [],
                               }
                          }
        self.slice_of_interest = slice_of_interest
        #Encoder 
        self.res1 = ResBlock(1, 16)
        self.res2 = ResBlock(16, 32)
        self.res3 = ResBlock(32, 64)
        self.res4 = ResBlock(64, 128)
        self.res5 = ResBlock(128, 256)
        self.res6 = ResBlock(256, 512, stride = 1)
        
        #Decoder
        self.up1 = UpBlock(512+256, 128)
        self.up2 = UpBlock(128+128, 64)
        self.up3 = UpBlock(64+64, 32)
        self.up4 = UpBlock(32+32, 16)
        self.up5 = UpBlock(16+16, 3)

        #Final
        self.final_conv = nn.Conv2d(3, num_classes, kernel_size = 1)
        self.initialize_weights()
    def initialize_weights(self):
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
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]** m.kernel_size[2]
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
        x = x[:,:,:, :, self.slice_of_interest]
        out1 = self.res1(x)
        out2 = self.res2(out1)
        out3 = self.res3(out2)
        out4 = self.res4(out3)
        out5 = self.res5(out4)
        out6 = self.res6(out5)

        out = self.up1(out6, out5)
        out = self.up2(out, out4)
        out = self.up3(out, out3)
        out = self.up4(out, out2)
        out = self.up5(out, out1)

        out = self.final_conv(out)
        return out