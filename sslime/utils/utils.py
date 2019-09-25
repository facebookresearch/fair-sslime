#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import collections.abc
import datetime
import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from sslime.core.config import config as cfg

logger = logging.getLogger(__name__)


def get_mean_image(crop_size):
    img = Image.fromarray(128 * np.ones((crop_size, crop_size, 3), dtype=np.uint8))
    return img


class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        return torch.flatten(feat, start_dim=self.dim)


def is_pos_int(number):
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return ((cur_epoch + 1) % cfg.TRAINER.EVAL_PERIOD) == 0 or (
        cur_epoch + 1
    ) == cfg.TRAINER.MAX_EPOCH


def is_log_iter(cur_iter):
    """Determines if Stats should be logger at current iteration."""
    return ((cur_iter + 1) % cfg.TRAINER.LOG_ITER_PERIOD) == 0


def log_train_stats(i_epoch, iter, max_iter, optimizer, meters, keys=("loss", "top_1")):
    output = [
        f"Epoch {i_epoch + 1}/{cfg.TRAINER.MAX_EPOCHS}",
        f"Iter {iter + 1}/{max_iter}",
    ]
    lrs = [round(param_group["lr"], 7) for param_group in optimizer.param_groups]
    output.append(f"Lr: {lrs}")
    for meter in meters:
        if meter.name in keys:
            if isinstance(meter.value, collections.abc.Sequence):
                output.append(f"{meter.name}: {[round(val, 3) for val in meter.value]}")
            else:
                output.append(f"{meter.name}: {round(meter.value, 3)}")
        elif isinstance(meter.value, dict):
            m_val = meter.value
            for key in keys:
                if key in m_val:
                    if isinstance(m_val[key], collections.abc.Sequence):
                        output.append(f"{key}: {[round(val, 3) for val in m_val[key]]}")
                    else:
                        output.append(f"{key}: {round(m_val[key], 3)}")

    logger.info(" | ".join(output))


def log_post_epoch_timer_stats(train_timer, test_timer, i_epoch):
    eta = train_timer.average_time * (cfg.TRAINER.MAX_EPOCHS - i_epoch)
    eta += (
        test_timer.average_time
        * (cfg.TRAINER.MAX_EPOCHS - i_epoch)
        / cfg.TRAINER.EVAL_PERIOD
    )
    curr_time = train_timer.diff + (test_timer.diff if is_eval_epoch(i_epoch) else 0)
    logger.info(
        "Epoch Time: {}, ETA Time: {}".format(
            datetime.timedelta(seconds=curr_time), datetime.timedelta(seconds=eta)
        )
    )


def parse_out_keys_arg(out_feat_keys, all_feat_names):
    """
    Checks if all out_feature_keys are mapped to a layer in the model.
    Ensures no duplicate features are requested.
    Returns the last layer to forward pass through for efficiency.
    Adapted from (https://github.com/gidariss/FeatureLearningRotNet)
    """

    # By default return the features of the last layer / module.
    out_feat_keys = [all_feat_names[-1]] if out_feat_keys is None else out_feat_keys

    if len(out_feat_keys) == 0:
        raise ValueError("Empty list of output feature keys.")
    for f, key in enumerate(out_feat_keys):
        if key not in all_feat_names:
            raise ValueError(
                "Feature with name {0} does not exist. Existing features: {1}.".format(
                    key, all_feat_names
                )
            )
        elif key in out_feat_keys[:f]:
            raise ValueError("Duplicate output feature key: {0}.".format(key))

    # Find the highest output feature in `out_feat_keys
    max_out_feat = max(all_feat_names.index(key) for key in out_feat_keys)

    return out_feat_keys, max_out_feat
