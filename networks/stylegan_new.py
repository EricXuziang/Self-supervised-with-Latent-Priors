import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch
import torch.nn as nn
import sys
sys.path.append('/well/rittscher/users/ycr745/monodepth2_revised/networks')
from rrdb import RRDBNet

from math import sqrt


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)
    
class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
    

class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None
    
class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply

class AdaIN(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)
        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out
    
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
    
class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        if noise is not None:
            return image + self.weight * noise
        else:
            return image

class StyledConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim=512, upsample=False, kernel_size=4, stride=2, padding=1, groups=1):
        super().__init__()

        if upsample:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
                EqualConv2d(in_channel, out_channel, kernel_size=3, padding=1),
                Blur(out_channel),
            )
        else:
            self.conv1 = EqualConv2d(in_channel, out_channel, kernel_size=3, padding=1)

        self.noise1 = equal_lr(NoiseInjection(out_channel))  # Noise injection after the first convolution
        self.adain1 = AdaIN(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.noise2 = equal_lr(NoiseInjection(out_channel))  # Noise injection after the second convolution
        self.adain2 = AdaIN(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        # First convolution block with noise injection
        out = self.conv1(input)
        out = self.noise1(out, noise)  # Inject noise here
        out = self.lrelu1(self.adain1(out, style))  # AdaIN and activation

        # Second convolution block with noise injection
        out = self.conv2(out)
        out = self.noise2(out, noise)  # Inject noise here
        out = self.lrelu2(self.adain2(out, style))  # AdaIN and activation

        return out




class StyledGenerator(nn.Module):
    def __init__(self, base_channels=16, style_dim=256, noise_shape=[15, 30, 60, 120, 240, None]):
        super().__init__()
        
        # Generator blocks (upscale progressively from 3x3 to 480x480)
        self.rrdb_blocks = nn.ModuleList([
            RRDBNet(channels=base_channels*16, filters=base_channels*16, num_res_blocks=2),
            RRDBNet(channels=base_channels*8, filters=base_channels*8, num_res_blocks=2),
            RRDBNet(channels=base_channels*4, filters=base_channels*4, num_res_blocks=2),
            RRDBNet(channels=base_channels*2, filters=base_channels*2, num_res_blocks=1),
            RRDBNet(channels=base_channels, filters=base_channels, num_res_blocks=1),
            RRDBNet(channels=base_channels, filters=base_channels, num_res_blocks=1),
        ])

        self.progression = nn.ModuleList([
                StyledConvBlock(base_channels*16, base_channels*16, style_dim, upsample=True, kernel_size=5, stride=5, padding=0, groups=base_channels*16), # 3x3 -> 15x15
                StyledConvBlock(base_channels*16, base_channels*8, style_dim, upsample=True), # 15x15 -> 30x30
                StyledConvBlock(base_channels*8, base_channels*4, style_dim, upsample=True),  # 30x30 -> 60x60
                StyledConvBlock(base_channels*4, base_channels*2, style_dim, upsample=True),  # 60x60 -> 120x120
                StyledConvBlock(base_channels*2, base_channels, style_dim, upsample=True),  # 120x120 -> 240x240
                StyledConvBlock(base_channels, base_channels, style_dim, upsample=True),   # 240x240 -> 480x480
        ])
        
        # Optional: Add initial noise constant input for generator mode
        self.initial_constant = nn.Parameter(torch.randn(1, base_channels*16, 3, 3))
        self.noise_shape = noise_shape

        layers = [PixelNorm()]
        for i in range(len(noise_shape)//2):
            layers.append(EqualLinear(style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.style = nn.Sequential(*layers)

        self.phase_out = nn.ModuleList(
            [
                EqualConv2d(base_channels*8, 1, 1),
                EqualConv2d(base_channels*4, 1, 1),
                EqualConv2d(base_channels*2, 1, 1),
                EqualConv2d(base_channels, 1, 1),
                EqualConv2d(base_channels, 1, 1),
            ]
        )

        # below will be trainable during laten-bank mode
        self.fusions = nn.ModuleList([
            EqualLinear(base_channels, style_dim),
            EqualLinear(base_channels*2, style_dim),
            EqualLinear(base_channels*4, style_dim),
            EqualLinear(base_channels*8, style_dim),
            EqualLinear(base_channels*16, style_dim),
        ])

        self.process_compress = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 1, 1, 0)
        )

    def forward(self, x_proj=None, x1=None, x2=None, x3=None, x4=None, x5=None, mode='generator', random_noise=None, current_phase=4):
        batch_size = random_noise.shape[0] if random_noise is not None else x_proj.shape[0]

        if mode == 'generator':
            # Start with a constant noise input and progressively upscale
            out = self.initial_constant.repeat(batch_size, 1, 1, 1)  # Start with 3x3 constant
            styles = self.style(random_noise)  # Use random noise as style input for AdaIN layers
            
            # Prepare a list of noise inputs for each block in the progression
            noise_list = []
            for shape in self.noise_shape:
                if shape is not None:
                    noise_list.append(torch.randn(batch_size, 1, shape, shape).cuda())
                else:
                    noise_list.append(None)

            # Progressively upscale from 3x3 to 480x480 using noise
            total_out_levels = len(self.phase_out)
            out_level = 1 + current_phase
            for i, conv in enumerate(self.progression):
                out = self.rrdb_blocks[i](conv(out, styles, noise_list[i]))  # Use random noise in generator mode
                if i >= out_level:
                    break
            out = self.phase_out[min(current_phase, total_out_levels-1)](out)
            
            return out

        elif mode == 'latent_bank':
            # Latent bank mode: start with x_proj from the encoder, and progressively merge x1, x2, x3, x4
            out = self.process_compress(x_proj)  # Start with the 3x3 projection from the encoder
            styles = [nn.Flatten()(F.adaptive_avg_pool2d(x, (1, 1))) for x in [x1, x2, x3, x4, x5]]
            for i, fusion in enumerate(self.fusions):
                styles[i] = self.style(fusion(styles[i]))
            # Progressively upscale from 3x3 to 480x480 without noise
            out_codes = []
            take_code = 5
            for i, conv in enumerate(self.progression):
                if i > 0:
                    # print(conv(out, styles[-i], noise=None))
                    out = self.rrdb_blocks[i](conv(out, styles[-i], noise=None))  # No noise injection in latent bank mode
                else:
                    style = nn.Flatten()(F.adaptive_avg_pool2d(out, (1, 1)))
                    out = self.rrdb_blocks[i](conv(out, style, noise=None))  # No noise injection in latent bank mode
                if i < take_code:
                    out_codes.append(out)
            return out_codes

       

    

def disp_to_depth(disp, min_depth=1e-3, max_depth=20):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    # disp = torch.sigmoid(disp)
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    
    # return scaled_disp, depth
    return scaled_disp, depth
    


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, norm='instance'):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, norm=None):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if norm == 'instance':
                layers.append(nn.InstanceNorm2d(out_filters))
            if norm == 'batch':
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 16, norm),
            *discriminator_block(16, 32, norm),
            *discriminator_block(32, 64, norm),
            *discriminator_block(64, 128, norm),
            *discriminator_block(128, 256, norm),
            *discriminator_block(256, 256, norm),
        )

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 1, 1, 1)
        )

    def forward(self, img_A):
        """Concatenate image and condition image by channels to produce input"""
        img_A = F.interpolate(img_A, size=(256, 256), mode='bilinear', align_corners=False)
        return self.pooling(self.model(img_A)).squeeze(-1).squeeze(-1)
    