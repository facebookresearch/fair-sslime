#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn

from sslime.core.config import config as cfg
from sslime.models.heads import HEADS
from sslime.models.trunks import TRUNKS
from sslime.models.trunks.feature_extractor import FeatureExtractorModel


class BaseImageSSLModel(nn.Module):
    def __init__(self):
        super(BaseImageSSLModel, self).__init__()
        if cfg.MODEL.FEATURE_EVAL_MODE:
            self.trunk = FeatureExtractorModel()
        else:
            self.trunk = TRUNKS[cfg.MODEL.TRUNK.TYPE]()
        self.clf_heads = nn.ModuleList()
        for (head_type, kwargs) in cfg.MODEL.HEAD.PARAMS:
            self.clf_heads.append(HEADS[head_type](**kwargs))

    def forward(self, batch):
        feats = self.trunk(batch)
        out = []
        for feat, head in zip(feats, self.clf_heads):
            out.append(head(feat))
        return out
