#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import torch

from sslime.core.config import config as cfg
from sslime.meters import METERS


logger = logging.getLogger(__name__)


def generic_eval_loop(val_loader, model, i_epoch):
    model.eval()
    eval_meters = [
        METERS[meter](**cfg.TRAINER.EVAL_METERS[meter])
        for meter in cfg.TRAINER.EVAL_METERS
    ]

    for batch in val_loader:
        batch["data"] = torch.cat(batch["data"]).cuda()
        batch["label"] = torch.cat(batch["label"]).cuda()
        with torch.no_grad():
            out = model(batch["data"])
            for meter in eval_meters:
                meter.update(out, batch["label"])

    logger.info("Epoch: {}. Validation Stats".format(i_epoch + 1))
    for meter in eval_meters:
        logger.info(meter)
