#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.optim as optim

from sslime.core.config import config as cfg


SCHEDULERS = {"step": optim.lr_scheduler.StepLR}


def get_scheduler(optimizer):
    assert (
        cfg.SCHEDULER.TYPE in SCHEDULERS
    ), "Unsupported Criterion {}. Currently supported are: {}".format(
        cfg.SCHEDULER.TYPE, list(SCHEDULERS.keys())
    )
    scheduler = SCHEDULERS[cfg.SCHEDULER.TYPE](
        optimizer, step_size=cfg.SCHEDULER.STEP_SIZE, gamma=cfg.SCHEDULER.GAMMA
    )
    return scheduler
