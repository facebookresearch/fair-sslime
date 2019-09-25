#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os

import torch

from sslime.core.config import config as cfg

logger = logging.getLogger(__name__)
_NAME_PREFIX = "model_epoch_"


def get_checkpoint(epoch):
    """Get the full path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(cfg.CHECKPOINT.DIR, name)


def is_checkpoint_epoch(cur_epoch):
    """Determines if a checkpoint should be saved on current epoch."""
    return (cur_epoch + 1) % cfg.CHECKPOINT.CHECKPOINT_PERIOD == 0


def get_last_checkpoint():
    names = os.listdir(cfg.CHECKPOINT.DIR) if os.path.exists(cfg.CHECKPOINT.DIR) else []
    names = [f for f in names if _NAME_PREFIX in f]
    assert len(names), "No checkpoints found in '{}'.".format(cfg.CHECKPOINT.DIR)
    name = sorted(names)[-1]
    return os.path.join(cfg.CHECKPOINT.DIR, name)


def has_checkpoint():
    """Determines if the given directory contains a checkpoint."""
    files = os.listdir(cfg.CHECKPOINT.DIR) if os.path.exists(cfg.CHECKPOINT.DIR) else []
    return any(_NAME_PREFIX in f for f in files)


def load_checkpoint(checkpoint_file, model, optimizer=None, scheduler=None):
    """Loads the checkpoint from the given file."""
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["epoch"]


def save_checkpoint(model, optimizer, scheduler, epoch):
    """Saves a checkpoint."""
    # Ensure that the checkpoint dir exists
    os.makedirs(cfg.CHECKPOINT.DIR, exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting
    sd = (
        model.module.state_dict()
        if len(cfg.GPU_IDS) > 1
        or (len(cfg.GPU_IDS) == 0 and torch.cuda.device_count() > 1)
        else model.state_dict()
    )
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    # Write the checkpoint
    checkpoint_file = get_checkpoint(epoch + 1)
    torch.save(checkpoint, checkpoint_file)
    logger.info("Saved checkpoint: {}".format(checkpoint_file))


def get_model_trunk_from_state(path):
    state_dict = torch.load(path)["model_state"]
    state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("trunk.")}
    return state_dict
