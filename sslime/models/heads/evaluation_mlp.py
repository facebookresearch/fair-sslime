#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from sslime.core.config import config as cfg
from sslime.models.heads.mlp import MLP


class Eval_MLP(nn.Module):
    def __init__(self, in_channels, dims):
        super(Eval_MLP, self).__init__()
        self.channel_bn = nn.BatchNorm2d(
            in_channels,
            eps=cfg.MODEL.HEAD.BATCHNORM_EPS,
            momentum=cfg.MODEL.HEAD.BATCHNORM_MOMENTUM,
        )

        self.clf = MLP(dims)

    def forward(self, batch):
        out = self.channel_bn(batch)
        out = torch.flatten(out, start_dim=1)
        out = self.clf(out)
        return out
