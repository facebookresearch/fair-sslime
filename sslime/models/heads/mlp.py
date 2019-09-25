#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch.nn as nn

from sslime.core.config import config as cfg


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        layers = []
        last_dim = dims[0]
        for dim in dims[1:-1]:
            layers.append(nn.Linear(last_dim, dim))
            if cfg.MODEL.HEAD.APPLY_BATCHNORM:
                layers.append(
                    nn.BatchNorm1d(
                        dim,
                        eps=cfg.MODEL.HEAD.BATCHNORM_EPS,
                        momentum=cfg.MODEL.HEAD.BATCHNORM_MOMENTUM,
                    )
                )
            layers.append(nn.ReLU(inplace=True))
            last_dim = dim

        layers.append(nn.Linear(last_dim, dims[-1]))

        self.clf = nn.Sequential(*layers)

    def forward(self, batch):
        out = self.clf(batch)
        return out
