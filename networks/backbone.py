import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F     
from torchvision.models import vgg19
from layers import *
from networks.rrdb import RRDBNet

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)
    
class UpSampleBlock(nn.Module):
    """
    Upsample using pixelshuffle
    """
    
    def __init__(self, in_filters, out_filters, bias=False, non_linearity=True):
        super(UpSampleBlock, self).__init__()

        def block(in_filters, out_filters, bias, non_linearity):
            layers = [nn.Conv2d(in_filters, out_filters, 3, 1, 1, bias=bias)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            layers += [nn.Conv2d(out_filters, out_filters, 1, 1, 0, bias=bias)]
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            layers += [nn.Conv2d(out_filters, 1, 1, 1, 0, bias=bias)]
            layers += [nn.Sigmoid()]
            return nn.Sequential(*layers)
        
        self.layer = block(in_filters, out_filters, bias, non_linearity)

    def forward(self, x):
        return self.layer(x)
    
class UpSampleBlock_mask(nn.Module):
    """
    Upsample using pixelshuffle
    """
    
    def __init__(self, in_filters, out_filters, bias=False, non_linearity=True):
        super(UpSampleBlock_mask, self).__init__()

        def block(in_filters, out_filters, bias, non_linearity):
            layers = [nn.Conv2d(in_filters, out_filters, 3, 1, 1, bias=bias)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            layers += [nn.Conv2d(out_filters, out_filters, 1, 1, 0, bias=bias)]
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            layers += [nn.Conv2d(out_filters, 2, 1, 1, 0, bias=bias)]
            layers += [nn.Sigmoid()]
            return nn.Sequential(*layers)
        
        self.layer = block(in_filters, out_filters, bias, non_linearity)

    def forward(self, x):
        return self.layer(x)
        
    
class DownSampleBlock(nn.Module):
    """
    Downsample using stride-2 convolution
    """
    
    def __init__(self, filters, bias=True, non_linearity=False):
        super(DownSampleBlock, self).__init__()
        
        def block(filters, bias, non_linearity):
            layers = [nn.Conv2d(filters, filters, 4, 2, 1, bias=bias), nn.Conv2d(filters, filters, 1, 1, bias=bias)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            return nn.Sequential(*layers)
        
        self.layer = block(filters, bias, non_linearity)

    def forward(self, x):
        return self.layer(x)
    
    
class CompressBlock(nn.Module):
    """
    Compress the features to vectors
    """
    
    def __init__(self, filters, step=7, latent_size=512, bias=False, non_linearity=False):
        super(CompressBlock, self).__init__()
        
        self.compress = nn.Sequential(nn.Conv2d(filters, step+1, 3, 1, 1, bias=bias))
        
        self.linear = nn.Linear(16, latent_size)

    def forward(self, x): # Nx64x4x4
        # print(x.shape)
        returns = []
        x = self.compress(x) # Nx8x4x4
        # print(x.shape)
        returns.append(x[:, 0])
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) # Nx8x16
        # print(x.shape)
        x = self.linear(x) # Nx8x512       
        for i in range(1, x.shape[1]):
            returns.append(x[:, i]) # each Nx512
        return returns


class DenseResidualBlock(nn.Module):
    """
    Based on: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class RRDBNet(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16):
        super(RRDBNet, self).__init__()

        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        feats = self.conv2(x)
        return feats

class Encoder(nn.Module):
    def __init__(self, img_channels=3, base_filters=64, latent_features=128,  num_res_blocks=4):
        super(Encoder, self).__init__()
            
        self.rrdbnet = RRDBNet(img_channels, base_filters, num_res_blocks) # 64 channels
        self.down1 = DownSampleBlock(base_filters)
        self.down2 = DownSampleBlock(base_filters)
        self.down3 = DownSampleBlock(base_filters)
        self.down4 = DownSampleBlock(base_filters)
        self.down5 = DownSampleBlock(base_filters)
        self.down6 = DownSampleBlock(base_filters)
        self.compress = CompressBlock(base_filters, latent_features)
            
    def forward(self, img):
        f0 = self.rrdbnet(img) # f0: 64x64x64
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2) # f3: 64x4x4
        f4 = self.down4(f3)
        f5 = self.down5(f4)
        f6 = self.down6(f5)
        c = self.compress(f6)
        return [f6, f5, f4, f3, f2, f1, f0], c

class Decoder(nn.Module):
    def __init__(self, img_channels=3, base_filters=64, latent_features=128):
        super(Decoder, self).__init__()
        
        self.up1 = UpSampleBlock(576, base_filters) # in 64x32x32 out 64x64x64
        self.up2 = UpSampleBlock(320, base_filters) # in 128x64x64 out 64x128x128
        self.up3 = UpSampleBlock(192, base_filters) # in 128x128x128 out 64x256x256
        self.up4 = UpSampleBlock(128, base_filters) # in 128x128x128 out 64x256x256
        
    def forward(self, feats, codes):
        self.outputs = {}

        out1 = self.up1(torch.cat((feats[0], codes[-4]), 1))
        # print(feats[0].shape, codes[-4].shape)
        out2 = self.up2(torch.cat((feats[1], codes[-3]), 1))
        # print(feats[1].shape, codes[-3].shape)
        out3 = self.up3(torch.cat((feats[2], codes[-2]), 1))
        # print(feats[2].shape, codes[-2].shape)
        out4 = self.up4(torch.cat((feats[3], codes[-1]), 1))
        # print(feats[3].shape, codes[-1].shape)
        
        self.outputs[("disp", 0)] = out4
        self.outputs[("disp", 1)] = out3
        self.outputs[("disp", 2)] = out2
        self.outputs[("disp", 3)] = out1

        return self.outputs
    
class Decoder_mask(nn.Module):
    def __init__(self, img_channels=3, base_filters=64, latent_features=128):
        super(Decoder_mask, self).__init__()
        
        self.up1 = UpSampleBlock_mask(64, base_filters) 
        self.up2 = UpSampleBlock_mask(320, base_filters)
        self.up3 = UpSampleBlock_mask(192, base_filters) 
        self.up4 = UpSampleBlock_mask(128, base_filters) 
        
    def forward(self, feats):
        self.outputs = {}
        out1 = self.up1(feats)

        return out1

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, norm=None):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, norm=None):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if norm == 'instance':
                layers.append(nn.InstanceNorm2d(out_filters))
            if norm == 'batch':
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, norm),
            *discriminator_block(64, 128, norm),
            *discriminator_block(128, 256, norm),
            *discriminator_block(256, 512, norm),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        """Concatenate image and condition image by channels to produce input"""
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=16):
        super(UNetEncoder, self).__init__()
        self.base_channels = base_channels
        
        # Downsample blocks
        self.down1 = self.conv_block(in_channels, base_channels)      # 480x480 -> 240x240
        self.down2 = self.conv_block(base_channels, base_channels*2)  # 240x240 -> 120x120
        self.down3 = self.conv_block(base_channels*2, base_channels*4) # 120x120 -> 60x60
        self.down4 = self.conv_block(base_channels*4, base_channels*8) # 60x60 -> 30x30
        self.down5 = self.conv_block(base_channels*8, base_channels*16) # 30x30 -> 15x15

        # Final 5x5 convolution to output 3x3x(16*base_channels)
        self.final_conv = nn.Conv2d(base_channels*16, base_channels*16, kernel_size=5, stride=5, padding=0, groups=base_channels*16)

    def conv_block(self, in_channels, out_channels):
        """Downsample block with 4x2x1 convolution, followed by 1x1x0 convolution."""
        return nn.Sequential(
            # First conv with stride=2 to downsample
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            # Second conv 3x3 (same resolution)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            # 1x1 convolution to reduce complexity
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        )
    
    def forward(self, x):
        # Downsampling
        x1 = self.down1(x)  # 480 -> 240
        x2 = self.down2(x1) # 240 -> 120
        x3 = self.down3(x2) # 120 -> 60
        x4 = self.down4(x3) # 60 -> 30
        x5 = self.down5(x4) # 30 -> 15

        # Apply the final 5x5 convolution to get 3x3 spatial output
        x_compressed = self.final_conv(x5)

        # Return the downsampled feature maps and the final compressed 3x3 projection
        return x1, x2, x3, x4, x5, x_compressed

class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1, base_channels=16):
        super(UNetDecoder, self).__init__()
        
        # Special upsampling from 3x3 -> 15x15 with a 5x5 transposed convolution
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*16, base_channels*16, kernel_size=5, stride=5, padding=0, groups=base_channels*16),  # Upscale 3x3 -> 15x15
            nn.BatchNorm2d(base_channels*16),
            nn.LeakyReLU(0.2)
        )
        # only encoder and decoder
        # Upsample blocks (with concatenation), in_channels = 2 * base_channels due to concatenation
        # self.up4 = self.up_conv_block(base_channels*32, base_channels*8)  # 15x15 -> 30x30
        # self.up3 = self.up_conv_block(base_channels*16, base_channels*4)  # 30x30 -> 60x60
        # self.up2 = self.up_conv_block(base_channels*8, base_channels*2)   # 60x60 -> 120x120
        # self.up1 = self.up_conv_block(base_channels*4, base_channels)   # 120x120 -> 240x240
        # Final upsampling to 480x480
        # self.final_up = nn.Sequential(
        #     nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),  # 240x240 -> 480x480
        #     nn.BatchNorm2d(base_channels),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(base_channels, out_channels, 1, 1, 0)
        # )

        # set latentbank
        self.up4 = self.up_conv_block(base_channels*48, base_channels*8)  # 15x15 -> 30x30
        self.up3 = self.up_conv_block(base_channels*24, base_channels*4)  # 30x30 -> 60x60
        self.up2 = self.up_conv_block(base_channels*12, base_channels*2)   # 60x60 -> 120x120
        self.up1 = self.up_conv_block(base_channels*6, base_channels)   # 120x120 -> 240x240
        
        # Final upsampling to 480x480
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(base_channels*3, base_channels, kernel_size=4, stride=2, padding=1),  # 240x240 -> 480x480
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, out_channels, 1, 1, 0)
        )

        self.dispconv1 = Conv3x3(base_channels*4, 1)
        self.dispconv2 = Conv3x3(base_channels*2, 1)
        self.dispconv3 = Conv3x3(base_channels, 1)
        
        self.sigmoid = nn.Sigmoid()

    def up_conv_block(self, in_channels, out_channels):
        """Upsample block with transposed convolution and concatenation."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),  # Upsample by 2
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 3x3 conv to refine features
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x1, x2, x3, x4, x5, x_compressed, code):
        self.outputs = {}
        # Start with compressed 3x3x(16*base_channels)
        x = self.up5(x_compressed)  # 3x3 -> 15x15
        
        # Concatenate encoder output and upsampled output, then upsample
        x = torch.cat([x, x5, code[0]], dim=1)  # Concatenate along the channel dimension
        x = self.up4(x)  # 15x15 -> 30x30

        x = torch.cat([x, x4, code[1]], dim=1) # Concatenate
        x = self.up3(x)  # 30x30 -> 60x60
        output_1 = self.sigmoid(self.dispconv1(x))
        # print(output_1.shape)

        x = torch.cat([x, x3, code[2]], dim=1)  # Concatenate
        x = self.up2(x)  # 60x60 -> 120x120
        output_2 = self.sigmoid(self.dispconv2(x))
        # print(output_2.shape)

        x = torch.cat([x, x2, code[3]], dim=1)  # Concatenate
        x = self.up1(x)  # 120x120 -> 240x240
        output_3 = self.sigmoid(self.dispconv3(x))
        # print(output_3.shape)

        # Final upsample to output size 480x480
        x = torch.cat([x, x1, code[4]], dim=1) 
        output_4 = self.sigmoid(self.final_up(x))  # 240x240 -> 480x480
        # print(output_4.shape)

        self.outputs[("disp", 0)] = output_4
        self.outputs[("disp", 1)] = output_3
        self.outputs[("disp", 2)] = output_2
        self.outputs[("disp", 3)] = output_1

        return self.outputs
    
class RRDBEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=16):
        super(RRDBEncoder, self).__init__()
        self.base_channels = base_channels
        
        # Downsample blocks
        self.down1 = nn.Sequential(
            self.conv_block(in_channels, base_channels),      # 480x480 -> 240x240
            RRDBNet(channels=base_channels, filters=base_channels, num_res_blocks=2)
        )
        self.down2 = nn.Sequential(
            self.conv_block(base_channels, base_channels*2),  # 240x240 -> 120x120
            RRDBNet(channels=base_channels*2, filters=base_channels*2, num_res_blocks=2)
        )
        self.down3 = nn.Sequential(
            self.conv_block(base_channels*2, base_channels*4), # 120x120 -> 60x60
            RRDBNet(channels=base_channels*4, filters=base_channels*4, num_res_blocks=2)
        )
        self.down4 = self.conv_block(base_channels*4, base_channels*8) # 60x60 -> 30x30

        self.down5 = self.conv_block(base_channels*8, base_channels*16) # 30x30 -> 15x15

        # Final 5x5 convolution to output 3x3x(16*base_channels)
        self.final_conv = nn.Conv2d(base_channels*16, base_channels*16, kernel_size=5, stride=5, padding=0, groups=base_channels*16)

    def conv_block(self, in_channels, out_channels):
        """Downsample block with 4x2x1 convolution, followed by 1x1x0 convolution."""
        return nn.Sequential(
            # First conv with stride=2 to downsample
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        # Downsampling
        x1 = self.down1(x)  # 480 -> 240
        x2 = self.down2(x1) # 240 -> 120
        x3 = self.down3(x2) # 120 -> 60
        x4 = self.down4(x3) # 60 -> 30
        x5 = self.down5(x4) # 30 -> 15

        # Apply the final 5x5 convolution to get 3x3 spatial output
        x_compressed = self.final_conv(x5)

        # Return the downsampled feature maps and the final compressed 3x3 projection
        return x1, x2, x3, x4, x5, x_compressed
