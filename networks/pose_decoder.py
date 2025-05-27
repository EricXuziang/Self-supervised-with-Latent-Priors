# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1) # for mu
        self.convs[("pose", 3)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1) # for var

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
    
    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features


        out = self.relu(self.convs[("pose", 0)](out))
        out = self.relu(self.convs[("pose", 1)](out))
        out_mu = self.convs[("pose", 2)](out)
        out_var = self.convs[("pose", 3)](out)
        
        # out = out.mean(3).mean(2)

        # out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)
        # axisangle = out[..., :3]
        # translation = out[..., 3:]
        out_mu = out_mu.mean(3).mean(2) 
        out_var = out_var.mean(3).mean(2)
        out_mu = 0.01 * out_mu.view(-1, self.num_frames_to_predict_for, 1, 6)
        out_var = 0.01 * out_mu.view(-1, self.num_frames_to_predict_for, 1, 6)
        # axisangle = out_mu[..., :3]
        # translation = out_mu[..., 3:]

        return out_mu, out_var