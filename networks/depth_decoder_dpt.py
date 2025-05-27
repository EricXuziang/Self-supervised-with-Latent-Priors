# Copyright © NavInfo Europe 2023.

import torch.nn as nn
import sys
sys.path.append('/well/rittscher/users/ycr745/monodepth2_revised/networks')

# from .transformer_utils import _make_scratch, Interpolate, FeatureFusionBlock_custom
from transformer_utils import _make_scratch, Interpolate, FeatureFusionBlock_custom

class DepthDecoderDpt(nn.Module):
    def __init__(self,
                 num_ch_enc=(96, 192, 384, 768),
                 scales=range(4),
                 features=256,
                #  features=96,
                 use_bn=True,
                 num_output_channels=1):

        super(DepthDecoderDpt, self).__init__()

        assert features in [64, 80, 96, 128, 256], 'Please choose transformer_features from [64, 80, 96, 128, 256]'

        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_output_channels = num_output_channels

        self.scratch = _make_scratch(
            self.num_ch_enc,
            out_shape=features,
            groups=1,
            expand=False
        )

        refinenets = []
        for i in range(4):
            refinenets.append(
                FeatureFusionBlock_custom(
                    features,
                    nn.ReLU(False),
                    deconv=False,
                    bn=use_bn,
                    expand=False,
                    align_corners=True
                )
            )
        self.refinenets = nn.Sequential(*refinenets)

        self.heads = nn.ModuleDict()
        for scale in scales:
            # Set head_convs = 2; heads = same
            self.heads[str(scale)] = nn.Sequential(
                nn.Conv2d(
                    features,
                    32,
                    kernel_size=3, stride=1, padding=1
                ),
                # nn.ReLU(True),
                nn.LeakyReLU(0.2),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(32, self.num_output_channels, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )

    def forward(self, input_features):
        self.outputs = {}
        
        for scale in self.scales[-1::-1]:
            # print(f"Scale {scale}, input feature stats: {input_features[scale].min()}, {input_features[scale].max()}, {input_features[scale].mean()}")
            layer_rn = self.scratch.layers_rn[scale](input_features[scale])
            # print(f"Scale {scale}, layer_rn stats: {layer_rn.min()}, {layer_rn.max()}, {layer_rn.mean()}")
            if scale == 3:
                path = self.refinenets[scale](layer_rn)
            else:
                path = self.refinenets[scale](path, layer_rn)
            # print(f"Scale {scale}, path stats after refinenet: {path.min()}, {path.max()}, {path.mean()}")

            self.outputs[("disp", scale)] = self.heads[str(scale)](path)

            # 梯度检测
            # if scale == 0:
            #     disp_pre_sigmoid = self.heads[str(scale)](path)[:-1]
            #     print(f"Scale {scale}, disp_pre_sigmoid stats: {disp_pre_sigmoid.min()}, {disp_pre_sigmoid.max()}, {disp_pre_sigmoid.mean()}")
            #     disp = self.heads[str(scale)](path)
            #     print(f"Scale {scale}, disp stats after sigmoid: {disp.min()}, {disp.max()}, {disp.mean()}")
            
                # for i, block in enumerate(self.scratch.layers_rn):
                #     if block.weight.grad is not None:
                #         print(f"scratch.layers_rn block {i}, Conv1 grad stats: {block.weight.grad.min()}, {block.weight.grad.max()}")
                
                # if self.refinenets[scale].resConfUnit1.conv2.weight.grad is not None:
                #     print(f"resConfUnit1.conv2 block, grad stats: {self.refinenets[scale].resConfUnit1.conv2.weight.grad.min()}, {self.refinenets[scale].resConfUnit1.conv2.weight.grad.max()}")
                # if self.refinenets[scale].resConfUnit2.conv2.weight.grad is not None:
                #     print(f"resConfUnit2.conv2 block, grad stats: {self.refinenets[scale].resConfUnit2.conv2.weight.grad.min()}, {self.refinenets[scale].resConfUnit2.conv2.weight.grad.max()}")
                
                # if self.refinenets[scale].resConfUnit2.bn2.weight.grad is not None:
                #     print(f"resConfUnit2.bn2 block, grad stats: {self.refinenets[scale].resConfUnit2.bn2.weight.grad.min()}, {self.refinenets[scale].resConfUnit2.bn2.weight.grad.max()}")
                # if self.refinenets[scale].resConfUnit1.skip_add.weight.grad is not None:
                #     print(f"resConfUnit1.skip_add block, grad stats: {self.refinenets[scale].resConfUnit1.skip_add.weight.grad.min()}, {self.refinenets[scale].resConfUnit1.skip_add.weight.grad.max()}")
                # if self.refinenets[scale].resConfUnit2.skip_add.weight.grad is not None:
                #     print(f"resConfUnit2.skip_add block, grad stats: {self.refinenets[scale].resConfUnit2.skip_add.weight.grad.min()}, {self.refinenets[scale].resConfUnit2.skip_add.weight.grad.max()}")
                
                # if self.refinenets[scale].out_conv.weight.grad is not None:
                #     print(f"out_conv block, grad stats: {self.refinenets[scale].out_conv.weight.grad.min()}, {self.refinenets[scale].out_conv.weight.grad.max()}")

                # head_conv1 = self.heads[str(scale)][0]  
                # head_conv2 = self.heads[str(scale)][3]

                # if head_conv1.weight.grad is not None:
                #     print(f"Scale {scale}, Conv1 grad stats: {head_conv1.weight.grad.min()}, {head_conv1.weight.grad.max()}")
                # if head_conv2.weight.grad is not None:
                #     print(f"Scale {scale}, Conv2 grad stats: {head_conv2.weight.grad.min()}, {head_conv2.weight.grad.max()}")
        
        
            # print(scale)
            # print(self.outputs[("disp", scale)].shape)
        return self.outputs