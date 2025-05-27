# Copyright © NavInfo Europe 2023.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import math
import random
import torch.nn.functional as F
from timm.utils import  ModelEma
import sys
sys.path.append('/well/rittscher/users/ycr745/monodepth2_revised/networks')
from transformers_deit import *
from transformer_utils import trunc_normal_
from timm.models.layers.helpers import to_2tuple

from timm.models.layers import  DropPath

# from .transformers_deit import *
# from .transformer_utils import trunc_normal_

def EM(x, stage_num=10, k=3):
    #input: x:[b,c,n]
    #output: mu:[b,k,c]  z:[b,n,k]
    def l2norm(inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
    def Kernel(x, mu):
        x_t = x.permute(0, 2, 1)  # b * n * c
        z = kappa * torch.bmm(x_t, mu)  # b * n * k
        return z
    mu = torch.Tensor(1, x.size(1), k).to(x.device)
    mu.normal_(0, math.sqrt(2. / k))
    mu = l2norm(mu, dim=1)
    kappa = 40.0
    b = x.shape[0]
    mu = mu.repeat(b, 1, 1)  # b * c * k
    with torch.no_grad():
        for i in range(stage_num):
            # E STEP:
            z = Kernel(x, mu)
            z = F.softmax(z, dim=2)  # b * n * k
            # M STEP:
            z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
            mu = torch.bmm(x, z_)  # b * c * k
            mu = l2norm(mu, dim=1)
    z = Kernel(x, mu)
    z = F.softmax(z, dim=2)  # b * n * k
    z = z / (1e-6 + z.sum(dim=1, keepdim=True))
    mu = mu.permute(0, 2, 1)  # b * k * c
    return mu, z

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attn_graph(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_abs = attn
        attn = attn.softmax(dim=-1)
        attn_softmax = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_abs, attn_softmax


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_attn_graph(self,x):
        y,attn_abs,attn_softmax=self.attn.get_attn_graph(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x,attn_abs,attn_softmax

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def load_interpolated_encoder(checkpoint, model, num_input_images=1):
    #checkpoint_model = checkpoint['model']
    if num_input_images > 1:
        checkpoint['encoder.patch_embed.proj.weight'] = torch.cat(
            [checkpoint['encoder.patch_embed.proj.weight']] * num_input_images, 1
        ) / num_input_images
    state_dict = model.state_dict()
    for k in ['encoder.head.weight', 'encoder.head.bias', 'encoder.head_dist.weight', 'encoder.head_dist.bias']:
        if k in checkpoint and k in state_dict and checkpoint[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint['encoder.pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.encoder.patch_embed.num_patches
    num_extra_tokens = model.encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    #orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    orig_size = (
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                checkpoint["height"] // model.encoder.patch_embed.patch_size[0]),
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                        checkpoint["width"] // model.encoder.patch_embed.patch_size[0])
    )
    # height (== width) for the new position embedding
    #new_size = int(num_patches ** 0.5)
    new_size = (model.encoder.patch_embed.img_size[0] // model.encoder.patch_embed.patch_size[0],
                model.encoder.patch_embed.img_size[1] // model.encoder.patch_embed.patch_size[1])
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, #size=(new_size, new_size),
        size=new_size, mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint['encoder.pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint, strict=False)


def load_interpolated_pose_encoder(checkpoint, model, size, num_input_images=1):

    width, height = size

    # if num_input_images > 1:
    #     checkpoint['encoder.patch_embed.proj.weight'] = torch.cat(
    #         [checkpoint['encoder.patch_embed.proj.weight']] * num_input_images, 1
    #     ) / num_input_images
    state_dict = model.state_dict()

    for k in ['encoder.head.weight', 'encoder.head.bias', 'encoder.head_dist.weight', 'encoder.head_dist.bias']:
        if k in checkpoint and k in state_dict and checkpoint[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint['encoder.pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.encoder.patch_embed.num_patches
    num_extra_tokens = model.encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    #orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    orig_size = (
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                height // model.encoder.patch_embed.patch_size[0]),
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                        width // model.encoder.patch_embed.patch_size[0])
    )
    # height (== width) for the new position embedding
    #new_size = int(num_patches ** 0.5)
    new_size = (model.encoder.patch_embed.img_size[0] // model.encoder.patch_embed.patch_size[0],
                model.encoder.patch_embed.img_size[1] // model.encoder.patch_embed.patch_size[1])
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, #size=(new_size, new_size),
        size=new_size, mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint['encoder.pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint, strict=False)


class TransformerEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, transformer_model, pretrained, img_size=(224, 224), num_input_images=1, mim_strategy=''):
        super().__init__()

        # decoder based on num_input_images.
        # Inspired by DPT for depth and by MD2 for pose
        assert num_input_images in [1, 2], 'The num_input_images to transformers is either 1 (depth) or 2 (pose)'
        self.num_input_images = num_input_images
        if self.num_input_images == 2:
            self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        else:
            self.num_ch_enc = np.array([96, 192, 384, 768])

        self.num_ch_enc[1:] *= 4

        if num_input_images > 1:
            self.register_buffer('img_mean',
                                 torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view(1, -1, 1, 1))
            self.register_buffer('img_std',
                                 torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        else:
            self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
            self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        transformers = {
            "deit-base": deit_base_patch16_224,
            "deit-small": deit_small_patch16_224,
            "deit-tiny": deit_tiny_patch16_224,
        }

        # self.num_input_images == 1
        self.hooks = [2, 5, 8, 11]

        if transformer_model not in transformers:
            transformer_model = 'deit-base'
            # raise ValueError("{} is not a valid transformer model".format(transformer_model))

        self.encoder = transformers[transformer_model](img_size=img_size,
                                                       pretrained=pretrained,
                                                       num_input_images=num_input_images)
        self.reassemble = nn.ModuleList()

        for i, num_ch_enc in enumerate(self.num_ch_enc):
            modules = [
                Transpose(1, 2),
                nn.Unflatten(2,
                             torch.Size([img_size[0] // self.encoder.patch_embed.patch_size[0],
                                         img_size[1] // self.encoder.patch_embed.patch_size[0]])),
                nn.Conv2d(
                    in_channels=self.encoder.embed_dim,
                    out_channels=num_ch_enc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            ]
            if self.num_input_images == 1:
                if i < len(self.num_ch_enc) - 1:
                    if (len(self.num_ch_enc) - i - 2) != 0:
                        modules.append(
                            nn.ConvTranspose2d(
                                in_channels=num_ch_enc,
                                out_channels=num_ch_enc,
                                kernel_size=2 ** (len(self.num_ch_enc) - i - 2),
                                stride=2 ** (len(self.num_ch_enc) - i - 2),
                                padding=0,
                                bias=True,
                                dilation=1,
                                groups=1
                            )
                        )
                else:
                    modules.append(
                        nn.Conv2d(
                            in_channels=num_ch_enc,
                            out_channels=num_ch_enc,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=True,
                            dilation=1,
                            groups=1
                        )
                    )
            else:
                if i < len(self.num_ch_enc) - 1:
                    modules.append(
                        nn.ConvTranspose2d(
                            in_channels=num_ch_enc,
                            out_channels=num_ch_enc,
                            kernel_size=2 ** (len(self.num_ch_enc) - i - 2),
                            stride=2 ** (len(self.num_ch_enc) - i - 2),
                            padding=0,
                            bias=True,
                            dilation=1,
                            groups=1
                        )
                    )
            self.reassemble.append(nn.Sequential(*modules))

        if any([strategy != '' for strategy in mim_strategy]):
            self.encoder.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim))
            trunc_normal_(self.encoder.mask_token, std=.02)

        self.encoder.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim))
        trunc_normal_(self.encoder.mask_token, std=.02)

        # self.cls_token_new = nn.Parameter(torch.zeros(1, 1, self.encoder.patch_embed.embed_dim))
        # self.pos_embed_new = nn.Parameter(torch.zeros(1, self.encoder.patch_embed.num_patches + 1, self.encoder.patch_embed.embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            Block(self.encoder.patch_embed.embed_dim, num_heads = 16, mlp_ratio = 4, qkv_bias=True,  norm_layer = nn.LayerNorm)
            for i in range(24)])
        
    def forward(self, input_image, mask=None, model_ema = None, mode='train'):
        self.features = []
        x = (input_image - self.img_mean) / self.img_std

        if self.num_input_images == 2:
            # feats = self.encoder(x)
            B = x.shape[0]
            x = self.encoder.patch_embed(x)

            if mask is not None:
                B, L, _ = x.shape
                mask_token = self.encoder.mask_token.expand(B, L, -1)
                w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
                x = x * (1 - w) + mask_token * w

            cls_tokens = self.encoder.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.encoder.pos_embed
            x = self.encoder.pos_drop(x)

            for blk in self.encoder.blocks:
                x = blk(x)

            x = self.encoder.norm(x)
            # Set read to ignore
            feats = x[:, 1:]

            for i in range(len(self.num_ch_enc)):
                if i < len(self.num_ch_enc) - 1:
                    self.features.append(
                        self.reassemble[i](feats)
                    )
                else:
                    self.features.append(
                        nn.functional.interpolate(
                            self.reassemble[i](feats),
                            scale_factor=2 ** (len(self.num_ch_enc) - i - 2),
                            mode='bilinear',
                            align_corners=True
                        )
                    )
        else:
            if mode == 'train':
                B = x.shape[0]
                # print('1',x.shape)
                x = self.encoder.patch_embed(x)

                # print('x',x.shape)
                # print('self.encoder.cls_token',self.encoder.cls_token.shape)
                # print('self.encoder.pos_embed',self.encoder.pos_embed.shape)
                # print('self.cls_token_new',self.cls_token_new.shape)
                # print('self.pos_embed_new',self.pos_embed_new.shape)
                x_input = x + self.encoder.pos_embed[:, 1:, :]

                x_masked, mask, ids_restore = self.strategy_masking_eff(x_input, mask_ratio = 0.2, model_ema = model_ema, epoch = 300)

                B, L, _ = x.shape
                mask_token = self.encoder.mask_token.expand(B, L, -1)
                w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
                x = x * (1 - w) + mask_token * w
                
                cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                # print(x.shape)
                x = x + self.encoder.pos_embed
                # print(x.shape)

            elif mode == 'test':
                B = x.shape[0]
                x = self.encoder.patch_embed(x)
                # print(x.shape)
                cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.encoder.pos_embed
                # print(x.shape)

            # if mask is not None:
            #     B, L, _ = x.shape
            #     mask_token = self.encoder.mask_token.expand(B, L, -1)
            #     w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            #     x = x * (1 - w) + mask_token * w

            # cls_tokens = self.encoder.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            # x = torch.cat((cls_tokens, x), dim=1)
            # print(x.shape)
            # x = x + self.encoder.pos_embed


            x = self.encoder.pos_drop(x)
            # print('3',x.shape)

            for hook, blk in enumerate(self.encoder.blocks):
                x = blk(x)
                if hook in self.hooks:
                    ind = self.hooks.index(hook)
                    if self.num_input_images == 2:
                        if ind < len(self.num_ch_enc) - 1:
                            self.features.append(
                                self.reassemble[ind](x[:, 1:])
                            )
                        else:
                            self.features.append(
                                nn.functional.interpolate(
                                    self.reassemble[ind](x[:, 1:]),
                                    scale_factor=2 ** (len(self.num_ch_enc) - ind - 2),
                                    mode='bilinear',
                                    align_corners=True
                                )
                            )
                    else:
                        if hook != self.hooks[-1]:
                            self.features.append(self.reassemble[ind](x[:, 1:]))

            x = self.encoder.norm(x)

            feats = x[:, 1:]
            # print(self.reassemble[-1](feats).shape)
            self.features.append(self.reassemble[-1](feats))

        return self.features

    def strategy_masking_eff(self, x, mask_ratio,model_ema=None,epoch=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        low = 10
        high = 40
        total_epoch = 300.0
        with torch.no_grad():
            if epoch is None:
                alpha=random.random()*0.5
            else:
                alpha =  math.pow(1.0/total_epoch*(epoch+1),0.5)


            cluster_idx=(high-low)*(total_epoch-epoch)/total_epoch+low
            cluster_num=[int(cluster_idx),int(cluster_idx+2)]
            parts_num=cluster_num[1]+1
            scale_factor=1
            tk_parts_dic={}
            N, L, D = x.shape  # batch, length, dim
            H,W=int(math.sqrt(L)),int(math.sqrt(L))
            len_keep = int(L * (1 - mask_ratio))


            tk_parts_dic=self.unsupervised_token_classification(x,cluster_num=cluster_num,model_ema=model_ema)

            # grid noise
            H = W = int(math.sqrt(L))
            noise_grid = torch.rand(2, 2, device=x.device)
            noise = torch.ones((L), device=x.device)
            for i in range(0, L):
                noise[i] = noise_grid[(i//W)%2, i%2]
            noise = noise.repeat(N, 1)

            parts_noise_asign=torch.rand(N, parts_num, device=x.device)
            parts_noise=torch.gather(parts_noise_asign, dim=1, index=tk_parts_dic)

            noise=(1-alpha)*noise+alpha*parts_noise
            noise=noise.detach()


            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)



        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def unsupervised_token_classification(self,x,cluster_num=[5,10],model_ema=None):
        with torch.no_grad():
            # attn_abs_set = model_ema.ema.get_features(x)
            attn_abs_set = self.get_features(x)
            cluster_num=random.randint(cluster_num[0],cluster_num[1])
            meta, meta_map = EM(attn_abs_set, stage_num=15, k=cluster_num)
            # serial num starts from 0
            classified=torch.max(meta_map,dim=-1)[1]

            return classified
    
    def get_features(self,input):
        with torch.no_grad():
            input = input.clone().detach()
            x = input

            # append cls token
            cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # apply Transformer blocks
            attn_abs_set = None
            for blk in self.blocks:
                x, attn_abs, attn_softmax = blk.get_attn_graph(x)
                if attn_abs_set is None:
                    attn_abs_set=attn_softmax
                else:
                    attn_abs_set=attn_abs_set+attn_softmax

            attn_abs_set=attn_abs_set[:,:,1:,1:]
            attn_abs_set=attn_abs_set.sum(dim=1)

            return attn_abs_set
