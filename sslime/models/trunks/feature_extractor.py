#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import torch
import torch.nn as nn

from sslime.core.config import config as cfg
from sslime.models.trunks import TRUNKS
from sslime.utils.checkpoint import get_model_trunk_from_state

POOL_OPS = {
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
    "AvgPool2d": nn.AvgPool2d,
    "MaxPool2d": nn.MaxPool2d,
}
logger = logging.getLogger(__name__)


class FeatureExtractorModel(nn.Module):
    def __init__(self):
        super(FeatureExtractorModel, self).__init__()

        assert (
            cfg.MODEL.TRUNK.TYPE in TRUNKS
        ), "Unknown base model type: {}. Supported Model Types: {}".format(
            cfg.MODEL.TRUNK.TYPE, list(TRUNKS.keys())
        )
        self.base_model = TRUNKS[cfg.MODEL.TRUNK.TYPE]()

        if cfg.CHECKPOINT.FEATURE_EXTRACTOR_PARAMS:
            self.base_model.load_state_dict(
                get_model_trunk_from_state(cfg.CHECKPOINT.FEATURE_EXTRACTOR_PARAMS)
            )
            logger.info(
                "Checkpoint loaded from: {}".format(
                    cfg.CHECKPOINT.FEATURE_EXTRACTOR_PARAMS
                )
            )

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.feature_pool_ops = nn.ModuleList(
            [
                POOL_OPS[pool_ops](*args)
                for (pool_ops, args) in cfg.MODEL.TRUNK.LINEAR_FEAT_POOL_OPS
            ]
        )

    def forward(self, batch):
        feats = self.base_model(batch, cfg.MODEL.EVAL_FEATURES)
        assert len(feats) == len(
            self.feature_pool_ops
        ), "Mismatch between number of features returned by base model ({}) and number of Linear Feature Pooling Ops ({})".format(
            len(feats), len(self.feature_pool_ops)
        )
        out = []
        for feat, op in zip(feats, self.feature_pool_ops):
            feat = op(feat)
            if cfg.MODEL.TRUNK.SHOULD_FLATTEN:
                feat = torch.flatten(feat, start_dim=1)
            out.append(feat)
        return out

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.base_model.eval()
        return self
