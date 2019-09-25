#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn

from sslime.core.config import config as cfg
from sslime.criterion.cross_entropy_multiple_output_single_target import (
    CrossEntropyMultipleOutputSingleTargetLoss,
)

CRITERION = {
    "cross_entropy": nn.CrossEntropyLoss,
    "cross_entropy_multiple_output_single_target": CrossEntropyMultipleOutputSingleTargetLoss,
}


def get_criterion():
    assert (
        cfg.TRAINER.CRITERION in CRITERION
    ), "Unsupported Criterion {}. Currently supported are: {}".format(
        cfg.TRAINER.CRITERION, list(CRITERION.keys())
    )
    return CRITERION[cfg.TRAINER.CRITERION]()
