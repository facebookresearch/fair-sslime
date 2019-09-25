#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torchvision.models as models

from sslime.utils.utils import Flatten, parse_out_keys_arg


class ResNet50(nn.Module):
    """Wrapper for TorchVison ResNet50 Model
    This was needed to remove the final FC Layer from the ResNet Model"""

    def __init__(self):
        super(ResNet50, self).__init__()
        model = models.resnet50()
        conv1 = nn.Sequential(model.conv1, model.bn1, model.relu)

        self._feature_blocks = nn.ModuleList(
            [
                conv1,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool,
                Flatten(1),
            ]
        )

        self.all_feat_names = [
            "conv1",
            "res1",
            "res2",
            "res3",
            "res4",
            "res5",
            "res5avg",
            "flatten",
        ]

        assert len(self.all_feat_names) == len(self._feature_blocks)

    def forward(self, x, out_feat_keys=None):
        out_feat_keys, max_out_feat = parse_out_keys_arg(
            out_feat_keys, self.all_feat_names
        )
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat + 1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        return out_feats
