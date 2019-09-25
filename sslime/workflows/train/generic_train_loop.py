#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from tqdm import tqdm

import torch

from sslime.core.config import config as cfg
from sslime.meters import METERS
from sslime.utils.utils import is_log_iter, log_train_stats

logger = logging.getLogger(__name__)


def generic_train_loop(train_loader, model, criterion, optimizer, scheduler, i_epoch):
    model.train()
    train_meters = [
        METERS[meter](**cfg.TRAINER.TRAIN_METERS[meter])
        for meter in cfg.TRAINER.TRAIN_METERS
    ]
    for i_batch, batch in enumerate(tqdm(train_loader)):
        batch["data"] = torch.cat(batch["data"]).cuda()
        batch["label"] = torch.cat(batch["label"]).cuda()
        out = model(batch["data"])
        loss = criterion(out, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for meter in train_meters:
                meter.update(out, batch["label"])

        if is_log_iter(i_batch):
            log_train_stats(
                i_epoch, i_batch, len(train_loader), optimizer, train_meters
            )

    scheduler.step()
    logger.info(f"Epoch: {i_epoch + 1}. Train Stats")
    for meter in train_meters:
        logger.info(meter)
