#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch
import torch.nn as nn
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import DataLoader

import sslime.utils.checkpoint as checkpoint
from sslime.criterion import get_criterion
from sslime.core.config import config as cfg, print_cfg
from sslime.data.datasets.ssl_dataset import GenericSSLDataset
from sslime.models.SSLModel import BaseImageSSLModel
from sslime.optimizers import get_optimizer
from sslime.schedulers import get_scheduler
from sslime.utils.timer import Timer
from sslime.utils.utils import is_eval_epoch, log_post_epoch_timer_stats
from sslime.workflows.eval import EVAL_LOOPS
from sslime.workflows.train import TRAIN_LOOPS


# create the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


class Trainer:
    """TrainLoop contains the core training loop logic."""

    def __init__(self):
        self.train_loop = TRAIN_LOOPS[cfg.TRAINER.TRAIN_LOOP]
        self.eval_loop = EVAL_LOOPS[cfg.TRAINER.EVAL_LOOP]

    def train(self):
        """
        Perform a training run.
        """
        print_cfg()
        logger.info("System config:\n{}".format(get_pretty_env_info()))

        model = BaseImageSSLModel()
        criterion = get_criterion()
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer)

        logger.info(model)

        start_epoch = 0
        if cfg.TRAINER.AUTO_RESUME and checkpoint.has_checkpoint():
            last_checkpoint = checkpoint.get_last_checkpoint()
            checkpoint_epoch = checkpoint.load_checkpoint(
                last_checkpoint, model, optimizer, scheduler
            )
            logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
            if not cfg.TRAINER.RESET_START_EPOCH:
                start_epoch = checkpoint_epoch + 1

        if torch.cuda.is_available():
            if len(cfg.GPU_IDS) > 1 or (
                len(cfg.GPU_IDS) == 0 and torch.cuda.device_count() > 1
            ):
                num_gpus = (
                    len(cfg.GPU_IDS) if cfg.GPU_IDS else torch.cuda.device_count()
                )
                model = nn.DataParallel(
                    model, device_ids=(cfg.GPU_IDS if cfg.GPU_IDS else None)
                )
                cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE * num_gpus
                cfg.TEST.BATCH_SIZE = cfg.TEST.BATCH_SIZE * num_gpus
            elif len(cfg.GPU_IDS) == 1:
                torch.cuda.set_device(cfg.GPU_IDS[0])

            model.cuda()

        train_dataset = GenericSSLDataset("TRAIN")
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.TRAINER.NUM_WORKERS,
            drop_last=True,
        )

        if cfg.TRAINER.EVAL_MODEL:
            val_dataset = GenericSSLDataset("TEST")
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.TRAINER.NUM_WORKERS,
                drop_last=True,
            )

        train_timer = Timer()
        test_timer = Timer()

        logger.info("=> Training model...")
        for i_epoch in range(start_epoch, cfg.TRAINER.MAX_EPOCHS):
            train_timer.tic()
            self.train_loop(
                train_loader, model, criterion, optimizer, scheduler, i_epoch
            )
            train_timer.toc()
            if checkpoint.is_checkpoint_epoch(i_epoch):
                checkpoint.save_checkpoint(model, optimizer, scheduler, i_epoch)
            if cfg.TRAINER.EVAL_MODEL and is_eval_epoch(i_epoch):
                test_timer.tic()
                self.eval_loop(val_loader, model, i_epoch)
                test_timer.toc()

            log_post_epoch_timer_stats(train_timer, test_timer, i_epoch)
