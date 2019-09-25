#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import torch.nn as nn
import torch.optim as optim

from sslime.core.config import config as cfg

logger = logging.getLogger(__name__)
OPTIMIZERS = {"sgd": optim.SGD}


def group_params_by_decay(model):
    group_decay = []
    group_no_decay = []
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, conv_types):
            group_decay.append(m.weight)
            if m.bias is not None:
                if cfg.OPTIMIZER.REGULARIZE_BIAS:
                    group_decay.append(m.bias)
                else:
                    group_no_decay.append(m.bias)
        elif isinstance(m, bn_types):
            if m.weight is not None:
                if cfg.OPTIMIZER.REGULARIZE_BATCHNORM:
                    group_decay.append(m.weight)
                else:
                    group_no_decay.append(m.weight)
            if m.bias is not None:
                if cfg.OPTIMIZER.REGULARIZE_BATCHNORM and cfg.OPTIMIZER.REGULARIZE_BIAS:
                    group_decay.append(m.bias)
                else:
                    group_no_decay.append(m.bias)

    return group_decay, group_no_decay


def get_optimizer(model):
    assert (
        cfg.OPTIMIZER.TYPE in OPTIMIZERS
    ), "Unsupported Criterion {}. Currently supported are: {}".format(
        cfg.OPTIMIZER.TYPE, list(OPTIMIZERS.keys())
    )

    group_decay, group_no_decay = group_params_by_decay(model)

    trainable_params = [params for params in model.parameters() if params.requires_grad]
    group_decay = [params for params in group_decay if params.requires_grad]
    group_no_decay = [params for params in group_no_decay if params.requires_grad]
    assert len(trainable_params) == len(group_decay) + len(group_no_decay)

    optimizer_grouped_parameters = [
        {"params": group_decay, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY},
        {"params": group_no_decay, "weight_decay": 0.0},
    ]
    optimizer = OPTIMIZERS[cfg.OPTIMIZER.TYPE](
        optimizer_grouped_parameters,
        lr=cfg.OPTIMIZER.BASE_LR,
        momentum=cfg.OPTIMIZER.MOMENTUM,
        dampening=cfg.OPTIMIZER.DAMPENING,
        nesterov=cfg.OPTIMIZER.NESTEROV,
    )

    logger.info(optimizer)
    logger.info("Traininable params: {}".format(len(trainable_params)))
    logger.info(
        "Regularized Parameters: {}. Unregularized Parameters {}".format(
            len(group_decay), len(group_no_decay)
        )
    )

    return optimizer
